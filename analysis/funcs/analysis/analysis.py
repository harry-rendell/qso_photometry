from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import column

wdir = '/disk1/hrb/python/'
DTYPES = {'catalogue': np.uint8, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, 'uid': np.uint32, 'uid_s':np.uint32}

def calc_moments(bins,weights):
	"""
	Calculate mean and kurtosis
	"""
	x = bins*weights
	z = (x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis]
	return x.mean(axis=1), (z**4).mean(axis = 1) - 3

def reader(args):
	"""
	Reading function for multiprocessing
	"""
	n_subarray, basepath, ID = args
	return pd.read_csv(basepath+'lc_{}.csv'.format(n_subarray), 
					   comment='#',
					   index_col = ID,
					   dtype = DTYPES)

class analysis():
	def __init__(self, ID, obj, band):
		self.ID = ID
		self.obj = obj
		self.band = band
		self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}
		self.plt_color_bokeh = {'u':'magenta', 'g':'green', 'r':'red','i':'black','z':'blue'}
		self.marker_dict = {1:'^', 3:'v', 5:'D', 7:'s', 11:'o'}
		self.marker_dict_bokeh = {1:'triangle', 3:'inverted_triangle', 5:'diamond', 7:'square', 11:'circle'}
		self.survey_dict = {1:'SSS_r1', 3: 'SSS_r2', 5:'SDSS', 7:'PS1', 11:'ZTF'}
		self.coords = pd.read_csv(wdir+'data/catalogues/qsos/dr14q/dr14q_uid_coords.csv', index_col='uid')

	def read_in(self, multi_proc = True, catalogue_of_properties = None, redshift=True):
		"""
		Read in raw data

		Parameters
		----------
		reader : function
				used for reading data
		multi_proc : boolean
				True to use multiprocessesing
		catalogue_of_properties : dataframe
		"""
		
		# Default to 4 cores
		if multi_proc == True:
			# Use the path below for SSA
			# basepath = wdir+'data/merged/{}/{}_band/with_ssa/'
			basepath = wdir+'data/merged/{}/{}_band/'.format(self.obj, self.band)
			pool = Pool(4)
			df_list = pool.map(reader, [(i, basepath, self.ID) for i in range(4)])
			self.df = pd.concat(df_list).rename(columns={'mag_ps':'mag'})
			if self.df.index.name == None:
				self.df = self.df.set_index(self.ID)
		elif multi_proc == False:
			self.df = pd.read_csv(wdir+'data/merged/{}/lc_{}_{}.csv'.format(self.obj, self.band), comment='#', index_col = self.ID, dtype = DTYPES)

		# Would be good to add in print statments saying: 'lost n1 readings due to -9999, n2 to -ve errors etc'

		# Remove bad values from SDSS (= -9999) and large outliers (bad data)
		self.df = self.df[(self.df['mag'] < 25) & (self.df['mag'] > 15)]
		# Remove -ve errors (why are they there?) and readings with >0.5 error
		self.df = self.df[ (self.df['magerr'] > 0) & (self.df['magerr'] < 0.5)]
		# Remove objects with a single observation.
		self.df = self.df[self.df.index.duplicated(keep=False)]
		if redshift:
			# add on column for redshift. Use squeeze = True when reading in a single column.
			self.redshifts = pd.read_csv(wdir+'data/catalogues/qsos/dr14q/dr14q_uid_desig_z.csv', index_col=self.ID, usecols=[self.ID,'z'], squeeze=True).rename('redshift')
			self.df = self.df.join(self.redshifts, how = 'left', on=self.ID)
			self.df['mjd_rf'] = self.df['mjd']/(1+self.df['redshift'])

		self.df = self.df.sort_values([self.ID, 'mjd'])
		assert self.df.index.is_monotonic, 'Index is not sorted'

	def residual(self, corrections):
		"""
		Apply residual corrections, defined such that the corrected star photometry has a
			non-zero offset
		"""
		self.df = self.df.reset_index().set_index([self.ID,'catalogue'])
		for cat in corrections.keys():
			self.df.loc[pd.IndexSlice[:, cat], 'mag'] += corrections[cat]
		self.df = self.df.reset_index('catalogue')
		assert self.df.index.is_monotonic, 'Index is not sorted'

	def summary(self):
		"""
		Run to create the following attributes:
		self.idx_uid : array_like
				unique set of uids
		self.uids_missing : array_like
				uids of objects which are present in DR14Q but not in observations
		self.n_qsos : int
				number of objects for which we have observations
		self.idx_cat : array_like
				list of surveys which contribute to observations (sdss=1, ps=2, ztf=3)
		"""
		# Check which qsos we are missing and which we have, given a list
		uids_complete	 = pd.Index(np.arange(1,526356+1), dtype = np.uint32)
		
		self.idx_uid	  = self.df.index.unique()
		self.uids_missing = uids_complete[~np.isin(uids_complete,self.idx_uid)]
		self.n_qsos		   = len(self.idx_uid)
		self.idx_cat	  = self.df['catalogue'].unique()

		print('Number of qsos with lightcurve: {:,}'.format(self.n_qsos))
		print('Number of datapoints in:\nSDSS: {:,}\nPS: {:,}\nZTF: {:,}'.format((self.df['catalogue']==5).sum(),(self.df['catalogue']==7).sum(),(self.df['catalogue']==11).sum()))

	def group(self, keys = ['uid'], read_in = True, redshift=True, colors=True, survey = None):
		"""
		Group self.df by keys and apply {'mag':['mean','std','count'], 'magerr':'mean', 'mjd': ['min', 'max', np.ptp]}

		Add columns to self.df_grouped:
				redshift   : by joining vac['redshift'] along uid)
				mjd_ptp_rf : max ∆t in rest frame for given object (same as self.properties)

		Parameters
		----------
		keys : list of str
		read_in : boolean
		survey : str
		Default is None. If 'ZTF' then read in grouped_stats computed from ZTF data only. If 'SSS' then read in grouped_stats with plate data included.
		"""
		if read_in == True:
			if len(keys) == 1:
				self.df_grouped = pd.read_csv(wdir+'data/merged/{}/{}_band/grouped_stats_{}_{}.csv'.format(self.obj, self.band, self.band, survey), index_col=0, comment='#')

			elif len(keys) == 2:
				self.df_grouped = pd.read_csv(wdir+'data/merged/{}/{}_band/grouped_stats_cat_{}.csv'.format(self.obj, self.band, self.band),index_col = [0,1])
		elif read_in == False:
			# median_mag_fn/mean_mag_fn calculate mean/median magnitude by fluxes rather than mags themselves
			mean_mag_fn   = ('mean'  , lambda mag: -2.5*np.log10(np.mean  (10**(-(mag-8.9)/2.5))) + 8.9)
			median_mag_fn = ('median', lambda mag: -2.5*np.log10(np.median(10**(-(mag-8.9)/2.5))) + 8.9)

			self.df_grouped = self.df.groupby(keys).agg({'mag':[mean_mag_fn, median_mag_fn,'std','count'], 'magerr':'mean', 'mjd': ['min', 'max', np.ptp]})
			self.df_grouped.columns = ["_".join(x) for x in self.df_grouped.columns.ravel()]

		if redshift:
			if ~hasattr(self, 'redshifts'):
				# add on column for redshift. Use squeeze = True when reading in a single column.
				self.redshifts = pd.read_csv(wdir+'data/catalogues/qsos/dr14q/dr14q_uid_desig_z.csv', index_col=self.ID, usecols=[self.ID,'z'], squeeze=True).rename('redshift')
			
			self.df_grouped = self.df_grouped.merge(self.redshifts, on=self.ID)
			self.df_grouped['mjd_ptp_rf'] = self.df_grouped['mjd_ptp']/(1+self.df_grouped['redshift'])
		
		if colors:
			df_colors = pd.read_csv(wdir+'data/computed/{}/colors_sdss.csv'.format(self.obj), index_col=0)
			self.df_grouped = self.df_grouped.join(df_colors, on=self.ID, how='left')

	def merge_with_catalogue(self,catalogue='dr12_vac', remove_outliers=True, prop_range_any = {'MBH_MgII':(6,12), 'MBH_CIV':(6,12)}):
		"""
		Reduce self.df to intersection of self.df and catalogue.
		Compute summary() to reupdate idx_uid, uids_missing, n_qsos and idx_cat
		Create new DataFrame, self.properties, which is inner join of [df_grouped, vac] along uid.
		Add columns to self.properties:
				mag_abs_mean : mean absolute magnitude
				mjd_ptp_rf   : max ∆t in rest frame for given object

		Parameters
		----------
		catalogue : DataFrame
				value added catalogue to be used for analysis
		remove_outliers : boolean
				remove objects which have values outside range specified in prop_range_any
		prop_range_any : dict
				dictionary of {property_name : (lower_bound, upper_bound)}

		"""
		if catalogue == 'dr12_vac':
			# cols = z, Mi, L5100, L5100_err, L3000, L3000_err, L1350, L1350_err, MBH_MgII, MBH_CIV, Lbol, Lbol_err, nEdd, sdss_name, ra, dec, uid
			prop_range_all = {'Mi':(-30,-20),'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48),'nEdd':(-3,0.5)}
			self.prop_range = {**prop_range_all, **prop_range_any}
			vac = pd.read_csv(wdir+'data/catalogues/qsos/dr12q/SDSS_DR12Q_BH_matched.csv', index_col=self.ID)
			vac = vac.rename(columns={'z':'redshift_vac'});

		if catalogue == 'dr14_vac':
			# cols = ra, dec, uid, sdssID, plate, mjd, fiberID, z, pl_slope, pl_slope_err, EW_MgII_NA, EW_MgII_NA_ERR, FWHM_MgII_NA, FWHM_MgII_NA_ERR, FWHM_MgII_BR, FWHM_MgII_BR_ERR, EW_MgII_BR, EW_MgII_BR_ERR, MBH_CIV, MBH_CIV_ERR, MBH, MBH_ERR, Lbol

			prop_range_all = {'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48)}
			self.prop_range = {**prop_range_all, **prop_range_any}
			vac = pd.read_csv(wdir+'data/catalogues/qsos/dr14q/dr14q_spec_prop_matched.csv', index_col=self.ID)
			vac = vac.rename(columns={'z':'redshift_vac'});

		print(vac.index)
		self.df = self.df[self.df.index.isin(vac.index)]

		# Recalculate and print which qsos we are missing and which we have
		self.summary()
		self.properties = self.df_grouped.join(vac, how = 'inner', on=self.ID)
		self.vac = vac
		
		#calculate absolute magnitude
		self.properties['mag_abs_mean'] = self.properties['mag_mean'] - 5*np.log10(3.0/7.0*self.redshifts*(10**9))
		self.properties['mjd_ptp_rf']   = self.properties['mjd_ptp']/(1+self.redshifts)

		if remove_outliers:
			# Here, the last two entries of the prop_range dictionary are included on an any basis (ie if either are within the range)
			mask_all = np.array([(bound[0] < self.properties[key]) & (self.properties[key] < bound[1]) for key, bound in prop_range_all.items()])
			mask_any  = np.array([(bound[0] < self.properties[key]) & (self.properties[key] < bound[1]) for key, bound in prop_range_any.items()])
			mask = mask_all.all(axis=0) & mask_any.any(axis=0)
			self.properties = self.properties[mask]
	
	def merge_slopes():
		names = ['restframe_all','restframe_ztf']
		slopes = pd.concat([pd.read_csv(wdir+'data/catalogues{}/slopes_{}.csv'.format(self.obj, name), index_col=self.ID, usecols=[self.ID,'m_optimal']) for name in names], axis=1)
		slopes.columns = []
		self.properties = self.properties.join(slopes, how='left', on=self.ID)

		
	def bounds(self, key, bounds = np.array([-6,-1.5,-1,-0.5,0,0.5,1,1.5,6]), save=False):
		"""
		Compute z score of key for each object

		Parameters
		----------
		key : string
				property from VAC

		Returns
		-------
		bounds : array_like
				array of bounds to be used
		z_score : pandas.DataFrame
				z value of property column (value-mean / std) and original value
		self.bounds_values : values of property for each value in bounds
		mean : float 
			mean of the property of the total population
		std : float
			std  of the property of the total population
		ax : axes handle
		"""
		fig, ax = plt.subplots(1,1,figsize = (6,3))
		mean = self.properties[key].mean()
		std  = self.properties[key].std()
		z_score = (self.properties[key]-mean)/std
		self.properties['z_score'] = z_score
		z_score.hist(bins = 200, ax=ax)
		self.bounds_values = bounds * std + mean
		for i in range(len(bounds)-1):
			# print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])&(self.properties['mag_count']>2)).sum()))
			print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])).sum()))
		for bound in bounds:
			ax.axvline(x=bound, color = 'k')
		ax.set(xlabel=key)
		if save == True:
			fig.savefig('bins_{}.pdf'.format(key),bbox_inches='tight')
		bounds_tuple = list(zip(bounds[:-1],bounds[1:]))
		return bounds_tuple, self.properties[[key,'z_score']], self.bounds_values, mean, std, ax

	def save_dtdm_rf(self, uids, time_key):
		"""
		Save (∆t, ∆m) pairs from lightcurves. 
		dtdm defined as: ∆m = (m2 - m1), ∆t = (t2 - t1) where (t1, m1) is the first obs and (t2, m2) is the second obs.
		Thus a negative ∆m corresponds to a brightening of the object
		
		Parameters
		----------
		uids : array_like
			uids of objects to be used for calcuation
		time_key : str
			either mjd or mjd_rf for regular and rest frame respectively
		
		Returns
		-------
		df : DataFrame
			DataFrame(columns=[self.ID, 'dt', 'dm', 'de', 'dm2_de2', 'cat'])
		where
			dt : time interval between pair
			dm : magnitude difference between pair
			de : error on dm, calculated by combining individual errors in quadrature as sqrt(err1^2 + err2^2)
			dm2_de2 : dm^2 - de^2, representing the intrinsic variability once photometric variance has been removed
			cat : an ID representing which catalogue this pair was created from, calculated as survey_id_1*survey_id_2
				where survey_ids are: 
					1 = SSS_r1
					3 = SSS_r2
					5 = SDSS
					7 = PS1
					11 = ZTF
		"""
		sub_df = self.df[[time_key, 'mag', 'magerr', 'catalogue']].loc[uids]

		df = pd.DataFrame(columns=[self.ID, 'dt', 'dm', 'de', 'dm2_de2', 'cat'])
		for uid, group in sub_df.groupby(self.ID):
			#maybe groupby then iterrows? faster?
			mjd_mag = group[[time_key,'mag']].values
			magerr = group['magerr'].values
			cat	 = group['catalogue'].values
			n = len(mjd_mag)

			unique_pair_indicies = np.triu_indices(n,1)

			dcat = cat*cat[:,np.newaxis]
			dcat = dcat[unique_pair_indicies]

			dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
			dtdm = dtdm[unique_pair_indicies]
			dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

			dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
			dmagerr = dmagerr[unique_pair_indicies]
			
			dm2_de2 = dtdm[:,1]**2 - dmagerr**2
			
			duid = np.full(int(n*(n-1)/2),uid,dtype='uint32')
			
			# collate data to DataFrame and append
			df = df.append(pd.DataFrame(data={self.ID:duid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr, 'dm2_de2':dm2_de2, 'cat':dcat}))

			if (uid % 500 == 0):
				print(uid)

		return df

	def save_dtdm_rf_extras(self, df_ssa, uids, time_key):
		"""
		Similar to save_dtdm_rf except for plate data
		"""
		df = pd.DataFrame(columns=['uid', 'dt', 'dm', 'de', 'cat'])
		
		sub_df = df_ssa[[time_key, 'mag', 'magerr', 'catalogue']].loc[uids]
		
		for uid, group1 in sub_df.groupby(self.ID):
			group2 = self.df.loc[uid]
			mjd_mag1 = group1[[time_key, 'mag']].values # we need rest frame times
			mjd_mag2 = group2[[time_key,'mag']].values

			magerr1  = group1['magerr'].values
			magerr2  = group2['magerr'].values
			cat1 = group1['catalogue'].values
			cat2 = group2['catalogue'].values

			n = len(mjd_mag1)*len(mjd_mag2)
			dtdm = mjd_mag2[:, np.newaxis] - mjd_mag1
			dcat = cat2[:, np.newaxis]*cat1
			dmagerr = ( magerr2[:,np.newaxis]**2 + magerr1**2 )**0.5
			duid = np.full(n,uid,dtype='uint32')

			df = df.append(pd.DataFrame(data={self.ID:duid, 'dt':dtdm[:,:,0].ravel(), 'dm':dtdm[:,:,1].ravel(), 'de':dmagerr.ravel(), 'cat':dcat.ravel()}))
				
			if (uid % 500 == 0):
				print(uid)

		return df	
	
	def plot_sf_moments_pm(self, key, bounds, save = False, t_max=3011, ztf=False):
		"""
		Plots both positive and negative structure functions to investigate asymmetry in light curve.

		Parameters
		----------
		key : str
				Property from VAC
		bounds : array of z scores to use


		Returns
		-------
		fig, ax, fig2, axes2, fig3, axes3: axes handles
		"""
		fig, ax = plt.subplots(1,1,figsize = (16,8))
		fig2, axes2 = plt.subplots(2,1,figsize=(16,10))
		fig3, axes3 = plt.subplots(8,1,figsize = (16,50))
		label_range = {i:'{:.1f} < z < {:.1f}'.format(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)}
		label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}
		# label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
		label_moment = ['mean', 'Excess kurtosis']
		cmap = plt.cm.jet
		for i in range(8):
			dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = t_max, t_spacing='log', m_spacing='log', read_in=i+1, key=key, ztf=ztf)
			SF_n = (((m_bin_centres[:100]**2)*dms_binned[:,:100]).sum(axis=1)/dms_binned[:,:100].sum(axis=1))**0.5
			SF_p = (((m_bin_centres[100:]**2)*dms_binned[:,100:]).sum(axis=1)/dms_binned[:,100:].sum(axis=1))**0.5
			ax.plot(t_bin_chunk_centres, SF_p, label = label_range_val[i], lw = 0.5, marker = 'o', ls='-',  color = cmap(i/10))
			ax.plot(t_bin_chunk_centres, SF_n, label = label_range_val[i], lw = 0.5, marker = 'o', ls='--', color = cmap(i/10))
			ax.legend()
			ax.set(yscale='log', xscale='log')

			axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,:100].sum(axis=1), alpha = 0.5, label = '-ve',bins = 19)
			axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned[:,100:].sum(axis=1), alpha = 0.5, label = '+ve',bins = 19)
			axes3[i].set(yscale='log')
			dms_binned_norm = np.zeros((19,200))
			moments = np.zeros(19)
			for j in range(19):
				dms_binned_norm[j],_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[j], density=True);
				# print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
				# print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
			moments = calc_moments(m_bin_centres,dms_binned_norm)


			for idx, ax2 in enumerate(axes2.ravel()):
				ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
				# ax2.legend()
				ax2.set(xlabel='mjd', ylabel = label_moment[idx])
				ax2.axhline(y=0, lw=0.5, ls = '--')

		ax.set(xlabel='mjd', ylabel = 'structure function')
		if save:
			# fig.savefig('SF_{}.pdf'.format(key),bbox_inches='tight')
			fig2.savefig('moments_{}.pdf'.format(key),bbox_inches='tight')

		return fig, ax, fig2, axes2, fig3, axes3

	def plot_sf_moments(self, key, bounds, save = False, t_max=3011, ztf=False):
		fig, ax = plt.subplots(1,1,figsize = (16,8))
		fig2, axes2 = plt.subplots(2,1,figsize=(16,10))
		fig3, axes3 = plt.subplots(8,1,figsize = (16,50))
		label_range = {i:'{:.1f} < z < {:.1f}'.format(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)}
		label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}
		# label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
		label_moment = ['mean', 'Excess kurtosis']
		cmap = plt.cm.jet
		for i in range(8):
			dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = t_max, t_spacing='log', m_spacing='log', read_in=i+1, key=key, ztf=ztf)
			SF = (((m_bin_centres**2)*dms_binned).sum(axis=1)/dms_binned.sum(axis=1))**0.5
			ax.plot(t_bin_chunk_centres, SF, label = label_range_val[i], lw = 0.5, marker = 'o', ls='-',  color = cmap(i/10))
			ax.legend()
			ax.set(yscale='log', xscale='log')

			axes3[i].hist(t_bin_chunk[:-1], weights = dms_binned.sum(axis=1), alpha = 0.5,bins = 19)
			axes3[i].set(yscale='log')
			dms_binned_norm = np.zeros((19,200))
			moments = np.zeros(19)
			for j in range(19):
				dms_binned_norm[j],_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[j], density=True);
				# print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
				# print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
			moments = calc_moments(m_bin_centres,dms_binned_norm)


			for idx, ax2 in enumerate(axes2.ravel()):
				ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
				# ax2.legend()
				ax2.set(xlabel='mjd', ylabel = label_moment[idx])
				ax2.axhline(y=0, lw=0.5, ls = '--')

				# ax2.title.set_text(label_moment[idx])
		ax.set(xlabel='mjd', ylabel = 'structure function')
		if save == True:
			# fig.savefig('SF_{}.pdf'.format(key),bbox_inches='tight')
			fig2.savefig('moments_{}.pdf'.format(key),bbox_inches='tight')

		return fig, ax, fig2, axes2, fig3, axes3

	def plot_sf_ensemble(self, save = False):
		fig, ax = plt.subplots(1,1,figsize = (16,8))
		dms_binned_tot = np.zeros((8,19,200))
		for i in range(8):
			dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = self.bin_dtdm(t_max = 3011, t_spacing='log', m_spacing='log', read_in=i+1, key=key)
			dms_binned_tot[i] = dms_binned

		dms_binned_tot = dms_binned_tot.sum(axis=0)

		SF = (((m_bin_centres**2)*dms_binned_tot).sum(axis=1)/dms_binned_tot.sum(axis=1))**0.5
		ax.plot(t_bin_chunk_centres,SF, lw = 0.5, marker = 'o')
		ax.set(yscale='log',xscale='log')
		ax.set(xlabel='mjd',ylabel = 'structure function')
		if save == True:
			fig.savefig('SF_ensemble.pdf',bbox_inches='tight')

	def plot_series(self, uids, survey=None, filtercodes='r', axes=None, **kwargs):
		"""
		Plot lightcurve of given objects

		Parameters
		----------
		uids : array_like
				uids of objects to plot
		catalogue : int
				Only plot data from given survey
		survey : 1 = SSS_r1, 3 = SSS_r2, 5 = SDSS, 7 = PS1, 11 = ZTF

		"""
		if axes is None:
			fig, axes = plt.subplots(len(uids),1,figsize = (25,3*len(uids)), sharex=True)
		if len(uids)==1:
			axes=[axes]
			
		for uid, ax in zip(uids,axes):
			single_obj = self.df.loc[uid].sort_values('mjd')
			if len(filtercodes)>1:
				for band in filtercodes:
					single_band = single_obj[single_obj['filtercode']==band]
					if survey is not None:
						single_band = single_band[single_band['catalogue']==survey]
					for cat in single_band['catalogue'].unique():
						x = single_band[single_band['catalogue']==cat]
						ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 0.5, markersize = 3, marker = self.marker_dict[cat], label = self.survey_dict[cat]+' '+band, color = self.plt_color[band])
						mean = x['mag'].mean()
						ax2 = ax.twinx()
						ax.axhline(y=mean, color='k', ls='--', lw=0.4, dashes=(50, 20))
						ax2.set(ylim=np.array(ax.get_ylim())-mean, ylabel=r'$\mathrm{mag} - \overline{\mathrm{mag}}$')
						
			else:
				if survey is not None:
					single_obj = single_obj[single_obj['catalogue']==survey]
				for cat in single_obj['catalogue'].unique():
					x = single_obj[single_obj['catalogue']==cat]
					ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 0.5, marker = self.marker_dict[cat], label = self.survey_dict[cat]+' '+filtercodes, color = self.plt_color[filtercodes])
				mean = single_obj['mag'].mean()
				ax2 = ax.twinx()
				ax.axhline(y=mean, color='k', ls='--', lw=0.4, dashes=(50, 20))
				ax2.set(ylim=np.array(ax.get_ylim())-mean, ylabel=r'$\mathrm{mag} - \overline{\mathrm{mag}}$')


			ax.invert_yaxis()
			ax2.invert_yaxis()
			ax.set(xlabel='MJD', ylabel='mag', **kwargs)
	# 			ax.legend(loc=)
			ax.text(0.02, 0.9, 'uid: {}'.format(uid), transform=ax.transAxes, fontsize=10)
		
		plt.subplots_adjust(hspace=0)
		
		if axes is not None:
			return fig, axes

	def plot_series_bokeh(self, uids, survey=None, filtercodes=None):
		"""
		Plot lightcurve of given objects using bokeh

		Parameters
		----------
		uids : array_like
				uids of objects to plot
		catalogue : int
				Only plot data from given survey
		survey : 1 = SDSS, 2 = PS, 3 = ZTF

		"""

		plots = []
		for uid in uids:
			single_obj = self.df.loc[uid]
			if survey is not None:
				single_obj = single_obj[single_obj['catalogue']==survey]
			p = figure(title='uid: {}'.format(uid), x_axis_label='mjd', y_axis_label='r mag', plot_width=1000, plot_height=400)
			for cat in single_obj['catalogue'].unique():
				mjd, mag = single_obj[single_obj['catalogue']==cat][['mjd','mag']].sort_values('mjd').values.T
				p.scatter(x=mjd, y=mag, legend_label=self.survey_dict[cat], marker=self.marker_dict_bokeh[cat], color=self.plt_color_bokeh[filtercodes])
				p.line   (x=mjd, y=mag, line_width=0.5, color=self.plt_color_bokeh[filtercodes])
			p.y_range.flipped = True
			plots.append(p)

		show(column(plots))
	

	def plot_property_distributions(self, prop_range_dict, n_width, n_bins = 250, separate_catalogues = True):
		"""
		Parameters
		----------
		prop_range_dict : dict
				dictionary of keys and their ranges to be used in histogram
		n_width : int
				width of subplot
		n_bins : int
				number of bins to use in histogram
		separate_catalogues : boolean
				if True, plot data from each survey separately.
		"""
		m = -( -len(prop_range_dict) // n_width )
		fig, axes = plt.subplots(m, n_width,  figsize = (5*n_width,5*m))
		cat_label_dict = {1:'SDSS', 2:'PanSTARRS', 3:'ZTF'}
		for property_name, ax, in zip(prop_range_dict, axes.ravel()):
			if separate_catalogues == True:
				for cat, color in zip(self.cat_list,'krb'):
					self.properties[self.properties['catalogue']==cat][property_name].hist(bins = n_bins, ax=ax, alpha = 0.3, color = color, label = cat_label_dict[cat], range=prop_range_dict[property_name]);
				ax.legend()
			elif separate_catalogues == False:
				self.properties[property_name].hist(bins = 250, ax=ax, alpha = 0.3, range=prop_range_dict[property_name]);
			else:
				print('Error, seperate_catalogues must be boolean')
			ax.set(title = property_name)
