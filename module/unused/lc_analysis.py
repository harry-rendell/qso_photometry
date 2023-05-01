from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

class singlesurvey_prop():
    def __init__(self, band):
        self.band      = band
        self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}[band]

    def read_in(self, fname=None, reader=None, multi_proc=True, catalogue_of_properties=None):
        
        if multi_proc == True:
            pool = Pool(4)
            df_list = pool.map(reader, [1,2,3,4])
            df = pd.concat(df_list)
        
        elif multi_proc == False:
            datatype = {'catalogue': np.uint8, 'mag': np.float32,'magerr': np.float32, 'mjd': np.float64,'uid': np.uint32}
            df = pd.read_csv(fname, usecols = [0,1,2,3,4,5], index_col = 1, dtype = datatype)
        
        df = df[~df['oid'].isin(np.loadtxt(cfg.USER.W_DIR + 'analysis/ztf_oids_outside1arcsec.txt',dtype=np.uint64))]

        df['filtercode'] = df['filtercode'].str[-1]
        df = df[df['filtercode']==self.band]
        df = df.drop('filtercode',axis=1)
        
        # remove lightcurves with a single datapoint
        self.df = df[df.index.value_counts()>1]
        
        # Using pandas indicies might be better: pd.Index(dr2.uids_matched)
        self.uids_matched = np.sort(self.df.index.unique().astype('uint32'))
        uids_complete     = np.arange(1,526356+1, dtype = np.uint32)
        self.uids_missing = uids_complete[~np.isin(uids_complete,self.df.index.unique())]
        self.n_qsos       = len(self.uids_matched)
    
    def info(self):
        print('Number of qsos with lightcurves in {} band : {:,}'.format(self.band, self.n_qsos))
        print('Number of datapoints: {:,}'.format(len(self.df)))
        print('Max mag: {}, min mag: {}'.format(self.df['mag'].max(),self.df['mag'].min()))
            
    def plot_series(self, uids, axes = None, ms = 5, **kwargs):
        return_axes = False
        if type(uids) == int:
            if axes == None:
                fig, axes = plt.subplots(1,1,figsize = (13,5), **kwargs)
                return_axes = True
            single_obj = self.df.loc[uids].sort_values('mjd')
            axes.errorbar(single_obj.mjd, 
                        single_obj.mag,
                        yerr = single_obj.magerr, 
                        lw = 0.5, markersize = ms, marker = 'o', color = self.plt_color, label = 'ztf dr2 {}'.format(self.band))
            axes.invert_yaxis()
            axes.set(xlabel = 'mjd', ylabel = 'apparent mag')
        else:
            if axes == None:
                fig, axes = plt.subplots(len(uids),1,figsize = (13,2*len(uids)), **kwargs)
                plt.subplots_adjust(0)
                return_axes = True
            for uid, ax in zip(uids,axes.ravel()):
                single_obj = self.df.loc[uid].sort_values('mjd')
                ax.errorbar(single_obj.mjd, 
                            single_obj.mag,
                            yerr = single_obj.magerr, 
                            lw = 0.5, markersize = ms, marker = 'o', color = self.plt_color, label = 'ztf dr2 {}'.format(self.band))
                ax.invert_yaxis()
                ax.set(xlabel = 'mjd', ylabel = 'apparent mag')
        if return_axes is True:
            return fig, axes
        
    def group(self, keys = ['uid'], read_in = True):
        if read_in == True:
            if len(keys) == 1:
                self.df_grouped = pd.read_csv(cfg.USER.D_DIR + 'surveys/ztf/meta_data/ztfdr2_gb_uid_{}.csv'.format(self.band),index_col = 0)

        elif read_in == False:
            self.df_grouped = self.df.groupby(keys).agg({'mag':['mean','std','count'], 'magerr':'mean', 'mjd': ['min', 'max', np.ptp]})
            self.df_grouped.columns = ["_".join(x) for x in self.df_grouped.columns.ravel()]
 
    def struc_func(self, n_per_batch, n_batches, n_bins):
        """
        input: dataframe of all observations. Filter for mag_count>2. Split by uid. Apply fn where we calculate dtdm
        and sample given bin_probabilities * 0.01 so our array isnt too large. Append this to larger set. 
        output: total square sum and counts per bin
        """
        n_sample = n_per_batch*n_batches
        bin_edges = np.linspace(0,self.df_grouped['mjd_ptp'].max(),n_bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        # Need a better method of getting equal representation so that we have ncounts = const
        sub_uids = np.random.choice(self.df_grouped.index[(self.df_grouped['mag_count'] > 2)],n_sample)
        sub_df = self.df[['mjd','mag']].loc[sub_uids]
        sub_uids_split = np.split(sub_uids,n_batches)
        
        total_tss = 0
        total_counts = 0
        
        #define uids_batch
        for idx,uid_batch in enumerate(sub_uids_split): #- first bin by black hole mass or luminosity??
            dtdms_unsorted = np.empty((0,2))
            print('Batch {}: {}'.format(idx,uid_batch))
            for uid in uid_batch:
                mjd_mag = sub_df.loc[uid].values
        #         n = np.random.randint(len(mjd_mag))
        #         n = int(10*truncexpon.rvs(len(mjd_mag)/10))
                dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
                dtdm = dtdm[np.triu_indices(len(mjd_mag),1)]
                dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

                dtdms_unsorted = np.append(dtdms_unsorted,dtdm, axis = 0)
            print('dtdm array length: {}'.format(len(dtdms_unsorted)))
            # Do in batches of 1000 objects, calculate tss and counts, add to previous, and move on.
            # Give batches to separate cores for multi-threading
            tss, _, _ = binned_statistic(dtdms_unsorted[:,0],dtdms_unsorted[:,1],lambda x: np.sum(x**2),bins=bin_edges) #try median too
            counts, _, _ = binned_statistic(dtdms_unsorted[:,0],dtdms_unsorted[:,1],'count', bins = bin_edges) #try median too
            
            total_tss += tss
            total_counts += counts
            print('total counts: {}'.format(total_counts.sum()))
        
        return tss, counts, bin_edges, bin_centres
    
    def merge_with_catalogue(self,catalogue_of_properties):

        self.df    = self.df[self.df.index.isin(catalogue_of_properties.index)]

        # Recalculate which qsos we are missing and which we have, given a list (copy of code in self.read_in)
        self.uids_matched = np.sort(self.df.index.unique().astype('uint32'))
        uids_complete     = np.arange(1,526356+1, dtype = np.uint32)
        self.uids_missing = uids_complete[~np.isin(uids_complete,self.df.index.unique())]
        self.n_qsos       = len(self.uids_matched)
        
        self.properties = self.df_grouped.join(catalogue_of_properties, how = 'inner', on='uid')
        #calculate absolute magnitude
        self.properties['mag_abs_meaAn'] = self.properties['mag_mean'] - 5*np.log10(3.0/7.0*self.properties['redshift']*(10**9))
        
    def plot_property_distributions(self, prop_range_dict, n_width, n_bins = 250, separate_catalogues = True):
        m = -( -len(prop_range_dict) // n_width )
        fig, axes = plt.subplots(m, n_width,  figsize = (5*n_width,5*m))
        cat_label_dict = {1:'SDSS', 2:'PanSTARRS', 3:'ZTF'}
        for property_name, ax, in zip(prop_range_dict, axes.ravel()):
            self.properties[property_name].hist(bins = 250, ax=ax, alpha = 0.3, range=prop_range_dict[property_name]);
            ax.set(title = property_name)
            
    def plot_correlation(self, var1, var2):
        fig, ax = plt.subplots(figsize = (6,6))

		
class multisurvey_prop():
    def __init__(self, band):
        self.band      = band
        self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}[band]

    def read_in(self, reader, multi_proc = True, catalogue_of_properties = None):
        # Default to 4 cores
        if multi_proc == True:
            pool = Pool(4)
            df_list = pool.map(reader, [1,2,3,4])
            self.df = pd.concat(df_list)
        elif multi_proc == False:
            self.df = pd.read_csv('../lcs_merged/lc_{}.csv'.format(self.band), index_col = 5, dtype = {'catalogue': np.uint8, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, 'uid': np.uint32})
        
        #Would be good to add in print statments saying: 'lost n1 readings due to -9999, n2 to -ve errors etc'
        
        # Remove bad values from SDSS (= -9999) and large outliers (bad data)
        self.df = self.df[(self.df['mag'] < 22.5) & (self.df['mag'] > 15)]
        # Remove -ve errors (why are they there?) and readings with >0.5 error
        self.df = self.df[ (self.df['magerr'] > 0) & (self.df['magerr'] < 0.5)]
        
    def summary(self):
        
        # Check which qsos we are missing and which we have, given a list 
        self.idx_uid      = self.df.index.get_level_values('uid').unique()
        uids_complete     = pd.Index(np.arange(1,526356+1), dtype = np.uint32)
        self.uids_missing = uids_complete[~np.isin(uids_complete,self.df.index.get_level_values('uid').unique())]
        self.n_qsos       = len(self.idx_uid)
        self.idx_cat      = self.df.index.get_level_values('catalogue').unique()
        
        print('Number of qsos with lightcurves in {} band : {:,}'.format(self.band, self.n_qsos))
        print('Number of datapoints in:\nSDSS: {:,}\nPS: {:,}\nZTF: {:,}'.format(len(self.df.loc[pd.IndexSlice[:,1],:]), 
                                                                                 len(self.df.loc[pd.IndexSlice[:,2],:]),
                                                                                 len(self.df.loc[pd.IndexSlice[:,3],:])))

    def merge_with_catalogue(self,catalogue='dr12', remove_outliers=True, prop_range_any = {'MBH_MgII':(6,12), 'MBH_CIV':(6,12)}):
        
        if catalogue == 'dr12':
            prop_range_all = {'Mi':(-30,-20),'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48),'nEdd':(-3,0.5)}
            self.prop_range = {**prop_range_all, **prop_range_any}
            vac = pd.read_csv(cfg.USER.D_DIR + 'catalogues/SDSS_DR12Q_BH_matched.csv', index_col=16)
            vac = vac.rename(columns={'z':'redshift'});
        
        # This line is taking ages..?
        self.df    = self.df[self.df.index.get_level_values('uid').isin(vac.index)]
        
        # Recalculate which qsos we are missing and which we have, given a list (copy of code in self.summary)
        self.idx_uid      = self.df.index.get_level_values('uid').unique()
        uids_complete     = np.arange(1,526356+1, dtype = np.uint32)
        self.uids_missing = uids_complete[~np.isin(uids_complete,self.idx_uid)]
        self.n_qsos       = len(self.idx_uid)
        
        self.properties = self.df_grouped.join(vac, how = 'inner', on='uid')
        #calculate absolute magnitude
        self.properties['mag_abs_mean'] = self.properties['mag_mean'] - 5*np.log10(3.0/7.0*self.properties['redshift']*(10**9))
        if remove_outliers==True:
            # Here, the last two entries of the prop_range dictionary are included on an any basis (ie if either are within the range)
            mask_all = np.array([(bound[0] < self.properties[key]) & (self.properties[key] < bound[1]) for key, bound in prop_range_all.items()])
            mask_any  = np.array([(bound[0] < self.properties[key]) & (self.properties[key] < bound[1]) for key, bound in prop_range_any.items()])
            mask = mask_all.all(axis=0) & mask_any.any(axis=0)
            self.properties = self.properties[mask]
    
    def plot_series(self,uids,catalogue=None):
        fig, axes = plt.subplots(len(uids),1,figsize = (15,4*len(uids)))
        if catalogue is not None:
            self.df = self.df[self.df.catalogue == catalogue]
        for uid, ax in zip(uids,axes.ravel()):
            single_obj = self.df.loc[uid].sort_values('mjd')
            ax.errorbar(single_obj.mjd, 
                        single_obj.mag,
                        yerr = single_obj.magerr, 
                        lw = 0.2, markersize = 2, marker = 'o', color = self.plt_color)
            ax.invert_yaxis()
        
    def group(self, keys = ['uid'], read_in = True):
        if read_in == True:
            if len(keys) == 1:
                self.df_grouped = pd.read_csv(cfg.USER.D_DIR + 'merged/meta_data/df_gb_uid_{}.csv'.format(self.band),index_col = 0)
            elif len(keys) == 2:
                self.df_grouped = pd.read_csv(cfg.USER.D_DIR + 'merged/meta_data/df_gb_uid_cat_{}.csv'.format(self.band),index_col = [0,1])
            self.df_grouped['mjd_ptp_rf'] = self.df_grouped['mjd_ptp']/(1+self.df_grouped['z'])
        elif read_in == False:
            self.df_grouped = self.df.groupby(keys).agg({'mag':['mean','std','count'], 'magerr':'mean', 'mjd': ['min', 'max', np.ptp]})
            
            self.df_grouped.columns = ["_".join(x) for x in self.df_grouped.columns.ravel()]
                  
    def struc_func(self, n_per_batch, n_batches, bin_edges, catalogue = None):
        """
        input: dataframe of all observations. Filter for mag_count>2. Split by uid. Apply fn where we calculate dtdm
        and sample. Append this to larger set. 
        output:
        """
        n_sample = n_per_batch*n_batches
        
        if catalogue is not None:
            sub_uids = self.df_grouped[(self.df_grouped['mag_count'] > 2)].loc[pd.IndexSlice[:,1],:].index.get_level_values('uid').unique()
#             self.df_grouped.loc[pd.IndexSlice[:,catalogue],:].index.get_level_values('uid')[(self.df_grouped['mag_count'] > 2)]
        else:
            sub_uids = self.df_grouped.index.get_level_values('uid')[(self.df_grouped['mag_count'] > 2)]

        sub_uids = np.random.choice(sub_uids,n_sample)
        sub_df = self.df[['mjd','mag']].loc[sub_uids]
        sub_uids_split = np.split(sub_uids,n_batches)
        
        total_tss = 0
        total_counts = 0
        
        #define uids_batch
        for idx,uid_batch in enumerate(sub_uids_split): #- first bin by black hole mass or luminosity??
            dtdms_unsorted = np.empty((0,2))
            print('Batch {}: {}'.format(idx,uid_batch))
            for uid in uid_batch:
                mjd_mag = sub_df.loc[uid].values
        #         n = np.random.randint(len(mjd_mag))
        #         n = int(10*truncexpon.rvs(len(mjd_mag)/10))
                dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
                dtdm = dtdm[np.triu_indices(len(mjd_mag),1)]
                dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

                dtdms_unsorted = np.append(dtdms_unsorted,dtdm, axis = 0)
            print('dtdm array length: {:,}'.format(len(dtdms_unsorted)))
            # Do in batches of 1000 objects, calculate tss and counts, add to previous, and move on.
            # Give batches to separate cores for multi-threading
            tss, _, _ = binned_statistic(dtdms_unsorted[:,0],dtdms_unsorted[:,1],lambda x: np.sum(x**2),bins=bin_edges) #try median too
            counts, _, _ = binned_statistic(dtdms_unsorted[:,0],dtdms_unsorted[:,1],'count', bins = bin_edges)
            
            total_tss += tss
            total_counts += counts
            print('total counts: {:,}'.format(total_counts.sum()))
        
        return tss, counts
            
    def plot_property_distributions(self, prop_range_dict, n_width, n_bins = 250, separate_catalogues = True):
        m = -( -len(prop_range_dict) // n_width )
        fig, axes = plt.subplots(m, n_width,  figsize = (5*n_width,5*m))
        cat_label_dict = {1:'SDSS', 2:'PanSTARRS', 3:'ZTF'}
        for property_name, ax, in zip(prop_range_dict, axes.ravel()):
            if separate_catalogues == True:
                for cat, color in zip(self.cat_list,'krb'):
                    self.properties[self.properties.index.get_level_values('catalogue')==cat][property_name].hist(bins = n_bins, ax=ax, alpha = 0.3, color = color, label = cat_label_dict[cat], range=prop_range_dict[property_name]);
                ax.legend()
            elif separate_catalogues == False:
                self.properties[property_name].hist(bins = 250, ax=ax, alpha = 0.3, range=prop_range_dict[property_name]);
            else:
                print('Error, seperate_catalogues must be boolean')
            ax.set(title = property_name)
            
    def bounds(self,key,save=False):
        fig, ax = plt.subplots(1,1,figsize = (6,3))
        key = 'Lbol'
        z_score = (self.properties[key]-self.properties[key].mean())/self.properties[key].std()
        z_score.hist(bins = 200, ax=ax)
        bounds = [-6,-1.5,-1,-0.5,0,0.5,1,1.5,6]
        for i in range(len(bounds)-1):
            print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])&(dr.properties['mag_count']>2)).sum()))
        for bound in bounds:
            ax.axvline(x=bound, color = 'k')
        ax.set(xlabel=key)
        if save == True:
            fig.savefig('bins_{}.pdf'.format(key),bbox_inches='tight')
        return bounds
            

    def calc_dtdm(self, uids=None, n_bins_t = 1000, n_bins_m = 200, t_max=7600, t_spacing = 'log', m_spacing = 'log', read_in = False, key = None):
        """
        Take batch of qsos from a MBH bin. Calculate all dtdm for these qsos.
        Section these values into 19 large Δt bins with logarithmic spacing
        Within these large bins, bin Δt and Δm into 50 and 200 bins respectively.

        Parameters
        ----------
        uids : unique qso id
            list of qso uids whose lightcurves are to be used
        n_bins_t : total number of t bins


        Returns
        -------
        value : fullname
            descrip
        """
        if m_spacing == 'log':
            def calc_m_edges(n_bins_m, steepness):
                start = np.log10(steepness)
                stop = np.log10(steepness+3)
                return np.concatenate((-np.logspace(start,stop,int(n_bins_m/2+1))[:0:-1]+steepness,np.logspace(start,stop,int(n_bins_m/2+1))-steepness))
            m_bin_edges = calc_m_edges(200,0.2)
        elif m_spacing == 'lin':
            m_bin_edges = np.linspace(-3,3,201)

        if t_spacing == 'log':
            def calc_t_bins(t_max, n_bins_t=19 , steepness=10):
                start = np.log10(steepness)
                stop = np.log10(steepness+t_max)
                return np.logspace(start,stop,n_bins_t+1)-steepness
            t_bin_chunk = calc_t_bins(t_max = t_max, steepness = 1000)
        elif t_spacing == 'lin':
            t_bin_chunk = np.linspace(0,t_max,20)

        t_bin_edges = np.linspace(0,t_max,(n_bins_t+1))
        t_dict = dict(enumerate(['{0:1.0f}<t<{1:1.0f}'.format(t_bin_chunk[i],t_bin_chunk[i+1]) for i in range(len(t_bin_chunk)-1)]))
        dts_binned = np.zeros((19,n_bins_t))
        dms_binned = np.zeros((19,n_bins_m))
        m_bin_centres = (m_bin_edges[1:] + m_bin_edges[:-1])/2
        t_bin_chunk_centres = (t_bin_chunk[1:] + t_bin_chunk[:-1])/2


        if read_in != False:
            dms_binned = np.loadtxt(cfg.USER.W_DIR + 'analysis/dtdm/dms_binned_{}_{}.csv'.format(key,read_in), delimiter = ',')
    #         dts_binned = np.loadtxt(cfg.USER.W_DIR + 'analysis/dtdm/dts_binned_{}_{}.csv'.format(key,read_in), delimiter = ',')

        elif read_in == False:

            z = self.df_grouped[self.df_grouped['mag_count']>2]['z'].loc[uids]
            sub_df = self.df[['mjd','mag']].loc[uids]
            dtdms = [np.empty((0,2))]*19
            for uid,z in zip(uids,z.values):
                #maybe groupby then iterrows? faster?
                mjd_mag = sub_df.loc[uid].values

                # Rest frame - need to change time bins
                mjd_mag[:,0] /= (1+z)

                dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
                dtdm = dtdm[np.triu_indices(len(mjd_mag),1)]
                dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]
                idxs = np.digitize(dtdm[:,0], t_bin_chunk)-1
                for index in np.unique(idxs): #Can we vectorize this?
                    dtdms[index] = np.append(dtdms[index],dtdm[(idxs == index),:],axis = 0)

            print('now binning')
            for i in range(19):
                print('dtdm counts in {}: {:,}'.format(t_dict[i],len(dtdms[i])))
                dts_binned[i] += np.histogram(dtdms[i][:,0], t_bin_edges)[0]
                dms_binned[i] += np.histogram(dtdms[i][:,1], m_bin_edges)[0]

        return dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict
    

    def plot_sf_moments(self, key, bounds, save = False):
        fig, ax = plt.subplots(1,1,figsize = (16,8))
        fig2, axes2 = plt.subplots(2,2,figsize=(16,10))
        fig3, axes3 = plt.subplots(8,1,figsize = (16,50))
        label_range = {i:'{:.1f} < z < {:.1f}'.format(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)}
        label_moment = ['mean', 'std', 'skew_stand', 'kurt_stand']
        cmap = plt.cm.jet
        for i in range(8):
            dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.calc_dtdm(t_max = 3011, t_spacing='log', m_spacing='log', read_in=i+1, key=key)
            SF_n = (((m_bin_centres[:100]**2)*dms_binned[:,:100]).sum(axis=1)/dms_binned[:,:100].sum(axis=1))**0.5
            SF_p = (((m_bin_centres[100:]**2)*dms_binned[:,100:]).sum(axis=1)/dms_binned[:,100:].sum(axis=1))**0.5
            ax.plot(t_bin_chunk_centres, SF_p, label = label_range[i], lw = 0.5, marker = 'o', ls='-',  color = cmap(i/10))
            ax.plot(t_bin_chunk_centres, SF_n, label = label_range[i], lw = 0.5, marker = 'o', ls='--', color = cmap(i/10))
            ax.legend()
            ax.set(yscale='log', xscale='log')
            
            axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,:100].sum(axis=1), alpha = 0.5, label = '-ve',bins = 19)
            axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,100:].sum(axis=1), alpha = 0.5, label = '+ve',bins = 19)
            axes3[i].set(yscale='log')
            dms_binned_norm = np.zeros((19,200))
            moments = np.zeros(19)
            for j in range(19):
                dms_binned_norm[j],_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[j], density=True);
#                 print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
#                 print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
            moments = calc_moments(m_bin_centres,dms_binned_norm)


            for idx, ax2 in enumerate(axes2.ravel()):
                ax2.plot(t_bin_chunk_centres,moments[idx], lw = 0.5, marker = 'o', label = label_range[i], color = cmap(i/10.0));
        #         ax2.legend()
                ax2.set(xlabel='mjd',ylim=[None,None,[-1.5,0.5],[2.5,4.5]][idx])
                ax2.title.set_text(label_moment[idx])
        ax.set(xlabel='mjd',ylabel = 'structure function')
        if save == True:
            # fig.savefig('SF_{}.pdf'.format(key),bbox_inches='tight')
            fig2.savefig('moments_{}.pdf'.format(key),bbox_inches='tight')

    def plot_sf_ensemble(self, save = False):
        fig, ax = plt.subplots(1,1,figsize = (16,8))
        dms_binned_tot = np.zeros((8,19,200))
        for i in range(8):
            dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.calc_dtdm(t_max = 3011, t_spacing='log', m_spacing='log', read_in=i+1, key=key)
            dms_binned_tot[i] = dms_binned

        dms_binned_tot = dms_binned_tot.sum(axis=0)

        SF = (((m_bin_centres**2)*dms_binned_tot).sum(axis=1)/dms_binned_tot.sum(axis=1))**0.5
        ax.plot(t_bin_chunk_centres,SF, lw = 0.5, marker = 'o')
        ax.set(yscale='log',xscale='log')
        ax.set(xlabel='mjd',ylabel = 'structure function')
        if save == True:
            fig.savefig('SF_ensemble.pdf',bbox_inches='tight')
