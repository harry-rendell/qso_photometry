import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def reader(args):
	print('calling reader function')
	n_subarray, nrows = args
	return pd.read_csv('/disk1/hrb/python/data/surveys/ztf/calibStars/lc_{}.csv'.format(n_subarray), nrows = nrows, usecols=[0,2,3,4,5,7], index_col=['uid_s','filtercode'])

class star_survey():
	def __init__(self, survey):
		self.name = survey
#         self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}[band]

	def read_in_raw(self, nrows):
		print('status')
		if self.name == 'sdss':
			self.df = pd.read_csv('/disk1/hrb/python/data/surveys/sdss/calibStars/calibStarsSecondary.csv', sep=',', nrows = nrows, index_col='uid_s', usecols=[0,2,3,4,5,6,7,8,9,10,11,12])
            # Add columns for colors
            for b1, b2 in zip('gri','riz'):
                self.df[b1+'-'+b2] = self.df['mag_'+b1] - self.df['mag_'+b2]
            
        elif self.name == 'ps':
            cols = ['uid_s', 'filter', 'obsTime',# 'objID',
                    'psfFlux', 'psfFluxErr']
            dtype1 = {'uid_s': np.uint32, 'objID': np.uint64, 'obsTime': np.float64,'psfFlux': np.float32, 'psfFluxErr': np.float32}

            self.df = pd.read_csv('/disk1/hrb/python/data/surveys/ps/calibStars/PS_secondary_calibStars.csv', nrows = nrows, usecols = cols, dtype = dtype1)

            # Drop bad values
            self.df = self.df[self.df['psfFlux']!=0]

            self.df = self.df.rename(columns = {'obsTime': 'mjd', 'filter': 'filtercode'})
            self.df['mag'] = -2.5*np.log10(self.df['psfFlux']) + 8.90
            self.df['magerr'] = 1.086*self.df['psfFluxErr']/self.df['psfFlux']
            self.df = self.df.drop(['psfFlux','psfFluxErr'], axis = 1)
            self.df = self.df.set_index(['uid_s','filtercode'])
            
        elif self.name == 'ztf':
			print(__name__)
            if __name__ == '__main__':
                pool = Pool(4)
                try:
					print('attempting parallel reading')
                    df_list = pool.map(reader, [(0, nrows//4), (1, nrows//4), (2, nrows//4), (3, nrows//4)])
                except:
                    df_list = pool.map(reader, [(0, nrows), (1, nrows), (2, nrows), (3, nrows)])
                self.df = pd.concat(df_list)
				print('completed parallel reading')
        
	  
        else:
            print('Error, unrecognised survey')
            
    def apply_bounds(self, bounds):
        mask = np.array([self.df_pivot[key].between(bound[0],bound[1]) for key, bound in bounds.items()]).all(axis=0)
        self.df_pivot = self.df_pivot[mask]
        
    def apply_bounds_raw(self, bounds):
        mask = np.array([self.df[key].between(bound[0],bound[1]) for key, bound in bounds.items()]).all(axis=0)
        self.df = self.df[mask]
    
    def pivot(self, read_in=True):
        
        if self.name == 'sdss':
            if read_in:
                self.df_pivot = pd.read_csv('/disk1/hrb/python/data/surveys/{}/calibStars/gb.csv'.format(self.name), index_col='uid_s', dtype={'count':np.uint16}) 
            if not read_in:
                def stats(group):
                    """
                    Calculates group statistics. Only ~half of sdss have multiple observations so probably better off splitting into repeat and non repeat.
                    """
                    mag    = group[['mag_'   +b for b in 'griz']].values
                    mag_ps = group[['mag_ps_'+b for b in 'griz']].values
                    magerr = group[['magerr_'+b for b in 'griz']].values

                #     mean_airmass = group['airmass'].mean()
                    mean         = (mag*magerr**-2).sum(axis=0)/(magerr**-2).sum(axis=0)
                    mean_ps      = (mag_ps*magerr**-2).sum(axis=0)/(magerr**-2).sum(axis=0)
                    meanerr      = 1/(magerr**-2).sum(axis=0)
                    count        = len(mag[:,0])

                    mean_dict    = {'mean_'   +b: mean[i]    for i,b in enumerate('griz')}
                    meanps_dict  = {'mean_ps_'+b: mean_ps[i] for i,b in enumerate('griz')}
                    meanerr_dict = {'meanerr_'+b: meanerr[i] for i,b in enumerate('griz')}

                    mean_gr      = group['g-r'].mean()
                    mean_ri      = group['r-i'].mean()
                    mean_iz      = group['i-z'].mean()

                    meanerr_gr   = group['g-r'].std()
                    meanerr_ri   = group['r-i'].std()
                    meanerr_iz   = group['i-z'].std()

                    return {**mean_dict, **meanps_dict, **meanerr_dict, **{'count':int(count), 'mean_gr':mean_gr, 'meanerr_gr':meanerr_gr, 'mean_ri':mean_ri, 'meanerr_ri':meanerr_ri, 'mean_iz':mean_iz, 'meanerr_iz':meanerr_ri}}#, mean_airmass

                self.df_pivot = self.df.groupby('uid_s').apply(stats).apply(pd.Series)
        else:
            if read_in:
                grouped = pd.read_csv('/disk1/hrb/python/data/surveys/{}/calibStars/gb.csv'.format(self.name), index_col=['uid_s','filtercode'], dtype={'count':np.uint16}) 
                
            elif not read_in:
                print('creating uid chunks to assign to each core')
                uids = np.array_split(self.df.index,4)
                uid_0 = uids[0].unique()
                uid_1 = np.setdiff1d(uids[1].unique(),uid_0,assume_unique=True)
                uid_2 = np.setdiff1d(uids[2].unique(),uid_1,assume_unique=True)
                uid_3 = np.setdiff1d(uids[3].unique(),uid_2,assume_unique=True)
                print('assigning chunk to each core')
                if __name__ == '__main__':
                    pool = Pool(4)
                    df_list = pool.map(groupby_apply, [self.df.loc[uid_0],self.df.loc[uid_1],self.df.loc[uid_2],self.df.loc[uid_3]])
                    grouped = pd.concat(df_list)
#             self.grouped = grouped
            self.df_pivot = grouped.reset_index('filtercode').pivot(columns='filtercode', values=grouped.columns)
            self.df_pivot.columns = ["_".join(x) for x in self.df_pivot.columns.ravel()]
        
    def transform_to_ps(self, colors=None):
        # Transform original data using colors. Adds column '_ps' with color corrected data
        if self.name == 'ztf':
            self.df = self.df.merge(colors, how='left', on='uid_s') #merge colors onto ztf df
            self.df = self.df.reset_index('uid_s').set_index(['uid_s','filtercode'])       #add filtercode to index
            self.df['mag_ps'] = 0
            slidx = pd.IndexSlice[:, ['r','g']]
            self.df.loc[slidx,'mag_ps'] = (self.df.loc[slidx,'mag'] + self.df.loc[slidx,'clrcoeff']*(self.df.loc[slidx,'mean_gr']))
            slidx = pd.IndexSlice[:, 'i']
            self.df.loc[slidx,'mag_ps'] = (self.df.loc[slidx,'mag'] + self.df.loc[slidx,'clrcoeff']*(self.df.loc[slidx,'mean_ri']))

        if self.name == 'sdss':
            color_transf = pd.read_csv('color_transf_coef_to_ps.txt',index_col=0)
            x = self.df['g-r']
            for band in 'griz':
                a0, a1, a2, a3 = color_transf.loc[band].values
                # Convert to SDSS AB mags
                self.df['mag_ps_'+band] = self.df['mag_'+band] + a0 + a1*x + a2*(x**2) + a3*(x**3)

#         if self.name == 'sdss':
#             color_transf = pd.read_csv('color_transf_coef_to_ps.txt',index_col=0)
#             x = self.df_pivot['mean_g'] - self.df_pivot['mean_i']

#             for band in 'griz':
#                 a0, a1, a2, a3 = color_transf.loc[band].values
#                 # Convert to SDSS AB mags
#                 self.df_pivot['mean_ps_'+band] = self.df_pivot['mean_'+band] + a0 + a1*x + a2*(x**2) + a3*(x**3)

    def calculate_offset(self, other, bands='gri'):
        for b in bands:
            self.df_pivot[self.name+'-'+other.name+'_'+b]    = self.df_pivot['mean_'+b]    - other.df_pivot['mean_'+b]
            self.df_pivot[self.name+'\'-'+other.name+'\'_'+b] = self.df_pivot['mean_ps_'+b] - other.df_pivot['mean_ps_'+b]

    def calculate_pop_mask(self, band):
        mag = self.df_pivot['mean_'+band]
        self.masks = pd.DataFrame(data = {'17>mag':(mag<17),'17<mag<18':(17<mag)&(mag<18), '18<mag<19':(18<mag)&(mag<19), '19<mag<20':(19<mag)&(mag<20), 'mag>20':(mag>20)})
            
    def plot_offset_distrib(self, other):      
        fig, axes = plt.subplots(3,2,figsize=(23,13))
        bands = 'gri' #add this to argument?
        bins = 250
        for name, mask in self.masks.iteritems():
            data       = self.df_pivot.loc[mask, [self.name+'-'+other.name+'_'      +b for b in bands]]
            data_calib = self.df_pivot.loc[mask, [self.name+'\'-'+other.name+'\'_'+b for b in bands]]

            print('Offset for {} population: \n{}\n'.format(name, data.mean()))
            print('Offset for {} population: \n{}\n'.format(name, data_calib.mean()))

            data      .hist(bins=bins, range=(-0.5,0.5), alpha=0.5, label=name, ax=axes[:,0])
            data_calib.hist(bins=bins, range=(-0.5,0.5), alpha=0.5, label=name, ax=axes[:,1])
        for ax in axes.ravel():
            ax.set(yscale='log')
            ax.legend()
            
    def correlate_offset_color(self, other, band):

        fig, ax = plt.subplots(2,3,figsize=(21,14))
        xlim=[-0.5,1.6]
        ylim=[-0.3,0.2]
        for name, mask in self.masks.columns:
            for i, b1, b2 in zip(range(3),'gri','riz'):
                ax[0,i].scatter(self.df_pivot.loc[mask, b1+'-'+b2],  self.df_pivot.loc[mask, self.name+  '-'+other.name+  '_'+band], s=0.01, label = name)
                ax[1,i].scatter(self.df_pivot.loc[mask, b1+'-'+b2],  self.df_pivot.loc[mask, self.name+'\'-'+other.name+'\'_'+band], s=0.01, label = name)
            
                ax[0,i].set(xlabel=b1+'-'+b2, ylabel='offset ({}-{})'           .format(self.name, other.name), ylim=ylim, xlim=xlim)
                ax[1,i].set(xlabel=b1+'-'+b2, ylabel='calibrated offset ({}-{})'.format(self.name, other.name), ylim=ylim, xlim=xlim)
        for axis in ax.ravel():
            axis.legend(markerscale=50)