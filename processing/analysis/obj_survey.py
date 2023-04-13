import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns

# def stats_sdss(group):
# 	"""
# 	Calculates group statistics. Only ~half of sdss have multiple observations so probably better off splitting into repeat and non repeat.
# 	Note that for averaging magnitudes, we stack fluxes.
# 	"""
# 	mag	      = group[['mag_'   +b for b in 'griz']].values
# 	magerr    = group[['magerr_'+b for b in 'griz']].values
# 	magerr_max= np.nanmax(magerr,axis=0)
# 	mean      = -2.5*np.log10(np.nanmedian(10**(-(mag-8.9)/2.5), axis=0)) + 8.9
# 	meanerr	  = np.nansum(magerr**-2,axis=0)**-0.5 #REDO THIS with **-0.5
# 	count	  = int(len(mag[:,0]))

# 	mean_dict	 = {'mean_'     +b:mean[i]	     for i,b in enumerate('griz')}
# 	meanerr_dict = {'meanerr_'  +b:meanerr[i]    for i,b in enumerate('griz')}
# 	maxerr_dict  = {'magerr_max'+b:magerr_max[i] for i,b in enumerate('griz')}

# 	return {**mean_dict, **meanerr_dict, **maxerr_dict, **{'count':count}}#, mean_airmass

# def groupby_apply_ztf(df):

# 	def stats(group):
# 		mag       = group['mag'].dropna()
# 		mag_calib = group['mag_ps'].dropna()
# 		magerr    = group['magerr'].dropna()
# 		magerr_max= magerr.max()

# 		mean      = -2.5*np.log10(np.median(10**(-(mag      -8.9)/2.5))) + 8.9
# 		mean_calib= -2.5*np.log10(np.median(10**(-(mag_calib-8.9)/2.5))) + 8.9
# 		meanerr   = ((magerr**-2).sum())**-0.5
# 		count     = int(len(mag))
# 		return {'mean':mean, 'mean_ps':mean_calib, 'meanerr':meanerr, 'magerr_max':magerr_max, 'count':int(count)}
	
# 	return df.groupby(df.index.names).apply(stats).apply(pd.Series)

# def groupby_apply_ps(df):
	
# 	def stats_ps(group):
# 		mag       = group['mag'].dropna()
# 		magerr    = group['magerr'].dropna()
# 		magerr_max= magerr.max()

# 		mean      = -2.5*np.log10(np.median(10**(-(mag      -8.9)/2.5))) + 8.9
# 		meanerr   = ((magerr**-2).sum())**-0.5
# 		count     = int(len(mag))
# 		return {'mean':mean, 'meanerr':meanerr, 'magerr_max':magerr_max, 'count':int(count)}
	
# 	return df.groupby(df.index.names).apply(stats_ps).apply(pd.Series)

wdir = '/disk1/hrb/python/'

def ztf_reader(args):
	n_subarray, nrows, obj, ID = args
	return pd.read_csv(wdir+'/data/surveys/ztf/{}/lc_{}.csv'.format(obj,n_subarray), usecols = [ID, 'filtercode', 'mjd', 'mag', 'magerr', 'clrcoeff', 'oid'], index_col = [ID,'filtercode'], nrows=nrows, dtype = {'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})

class obj_survey():
	def __init__(self, survey, obj='qsos', ID='uid'):
		self.obj = obj
		self.ID = ID
		self.name = survey
#		 self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}[band]

	def read_in_raw(self, nrows=None, save=False):
		"""
		Read in raw survey data. All dataframes returned have index_col = [uid, filtercode]

		Parameters
		----------
		nrows  : number of rows to read in
		save   : boolean, true to save melted dataframe to file
		"""
		if self.name == 'sdss':
			self.df = pd.read_csv(wdir+'data/surveys/sdss/{}/sdss_secondary.csv'.format(self.obj), nrows=nrows, index_col=[self.ID, 'filtercode'])

		elif self.name == 'ps':
			# Check ps_secondary.csv to see if it has sources outside 1"
			cols = [self.ID, 'filter', 'obsTime',
					'psfFlux', 'psfFluxErr']
			dtype1 = {self.ID: np.uint32, 'objID': np.uint64, 'obsTime': np.float64,'psfFlux': np.float32, 'psfFluxErr': np.float32}

			self.df = pd.read_csv(wdir+'data/surveys/ps/{}/ps_secondary.csv'.format(self.obj), nrows=nrows, usecols=cols, dtype=dtype1)

			# Drop bad values
			self.df = self.df[self.df['psfFlux']!=0]

			self.df = self.df.rename(columns = {'obsTime': 'mjd', 'filter': 'filtercode'})
			self.df['mag'] = -2.5*np.log10(self.df['psfFlux']) + 8.90
			self.df['magerr'] = 1.086*self.df['psfFluxErr']/self.df['psfFlux']
			self.df = self.df.drop(['psfFlux','psfFluxErr'], axis = 1)
			self.df = self.df.set_index([self.ID,'filtercode'])
			
		elif self.name == 'ztf':
			if __name__ == 'funcs.analysis.'+self.__class__.__name__:
				pool = Pool(4)
				try:
					print('attempting parallel reading')
# 					df_list = pool.map(ztf_reader, [(0, nrows//4, self.obj), (1, nrows//4, self.obj), (2, nrows//4, self.obj), (3, nrows//4, self.obj)])
					df_list = pool.map(ztf_reader, [(i, nrows//4, self.obj, self.ID) for i in range(4)])
				except:
# 					df_list = pool.map(ztf_reader, [(0, nrows//4, self.obj), (1, nrows//4, self.obj), (2, nrows//4, self.obj), (3, nrows//4, self.obj)])
					df_list = pool.map(ztf_reader, [(i, nrows, self.obj, self.ID) for i in range(4)])
				self.df = pd.concat(df_list)
				print('completed parallel reading')
				
			# Remove a list of oids which are known to be outside 1". File loaded below is from merging_lcs.ipynb
			oid_to_remove = np.loadtxt(wdir+'data/surveys/ztf/ztf_oids_outside1arcsec.txt', dtype='uint')
			self.df = self.df[~self.df['oid'].isin(oid_to_remove)]
			assert (self.df['oid'].isin(oid_to_remove).sum() == 0), 'Not all bad oids have been removed'
			del self.df['oid']
			
		else:
			print('Error, unrecognised survey')
####################################################################################### Added code block from ../unused/qso_survey
	def pivot(self, read_in=True, magerr=None):
		if self.name == 'sdss':
			if read_in:
				self.df_pivot = pd.read_csv('/disk1/hrb/python/data/surveys/{}/{}/gb_sf_magerr_{:03d}.csv'.format(self.name, self.obj, int(magerr*100)), index_col=self.ID, dtype={'count':np.uint16}) 
			if not read_in:
				def stats(group):
					"""
					Calculates group statistics. Only ~half of sdss have multiple observations so probably better off splitting into repeat and non repeat.
					"""
					mag	= group[['mag_'   +b for b in 'griz']].values
					mag_ps = group[['mag_ps_'+b for b in 'griz']].values
					magerr = group[['magerr_'+b for b in 'griz']].values

				#	 mean_airmass = group['airmass'].mean()
					mean		 = (mag*magerr**-2).sum(axis=0)/(magerr**-2).sum(axis=0)
					mean_ps	  = (mag_ps*magerr**-2).sum(axis=0)/(magerr**-2).sum(axis=0)
					meanerr	  = 1/(magerr**-2).sum(axis=0)
					count		= len(mag[:,0])

					mean_dict	= {'mean_'   +b: mean[i]	for i,b in enumerate('griz')}
					meanps_dict  = {'mean_ps_'+b: mean_ps[i] for i,b in enumerate('griz')}
					meanerr_dict = {'meanerr_'+b: meanerr[i] for i,b in enumerate('griz')}

					mean_gr	  = group['g-r'].mean()
					mean_ri	  = group['r-i'].mean()
					mean_iz	  = group['i-z'].mean()

					meanerr_gr   = group['g-r'].std()
					meanerr_ri   = group['r-i'].std()
					meanerr_iz   = group['i-z'].std()

					return {**mean_dict, **meanps_dict, **meanerr_dict, **{'count':int(count), 'mean_gr':mean_gr, 'meanerr_gr':meanerr_gr, 'mean_ri':mean_ri, 'meanerr_ri':meanerr_ri, 'mean_iz':mean_iz, 'meanerr_iz':meanerr_ri}}#, mean_airmass

				self.df_pivot = self.df.groupby('uid').apply(stats).apply(pd.Series)
		else:
			if read_in:
				self.df_pivot = pd.read_csv('/disk1/hrb/python/data/surveys/{}/{}/gb_sf_magerr_{:03d}.csv'.format(self.name, self.obj, int(magerr*100)), index_col=self.ID, dtype={'count':np.uint16}) 

			elif not read_in:
				print('creating uid chunks to assign to each core')
				uids = np.array_split(self.df.index,4)
				uid_0 = uids[0].unique()
				uid_1 = np.setdiff1d(uids[1].unique(),uid_0,assume_unique=True)
				uid_2 = np.setdiff1d(uids[2].unique(),uid_1,assume_unique=True)
				uid_3 = np.setdiff1d(uids[3].unique(),uid_2,assume_unique=True)
				print('assigning chunk to each core')
				if __name__ == 'funcs.qso_survey':
					pool = Pool(4)
					df_list = pool.map(groupby_apply, [self.df.loc[uid_0],self.df.loc[uid_1],self.df.loc[uid_2],self.df.loc[uid_3]])
					grouped = pd.concat(df_list)
#			 self.grouped = grouped
				self.df_pivot = grouped.reset_index('filtercode').pivot(columns='filtercode', values=grouped.columns)
				self.df_pivot.columns = ["_".join(x) for x in self.df_pivot.columns.ravel()]


#######################################################################################
	def transform_to_ps(self, colors=None, color='g-r', system=''):
		"""
		Transform original data using colors. Adds column '_ps' with color corrected data
		"""
		if self.name == 'ztf':
			self.df = self.df.join(colors, how='left', on=self.ID) #merge colors onto ztf df
# 			self.df = self.df.reset_index(self.ID).set_index([self.ID,'filtercode'])	   #add filtercode to index
			self.df['mag_ps'] = 0
			slidx = pd.IndexSlice[:, ['r','g']]
			self.df.loc[slidx,'mag_ps'] = (self.df.loc[slidx,'mag'] + self.df.loc[slidx,'clrcoeff']*(self.df.loc[slidx,'mean_gr']))
			slidx = pd.IndexSlice[:, 'i']
			self.df.loc[slidx,'mag_ps'] = (self.df.loc[slidx,'mag'] + self.df.loc[slidx,'clrcoeff']*(self.df.loc[slidx,'mean_ri']))

		if self.name == 'sdss':
			color_transf = pd.read_csv(wdir+'analysis/transformations/transf_to_ps_{}.txt'.format(system), sep='\s+', index_col=0)
			
			self.df['mag_ps'] = 0
			for band in 'griz':
				a0, a1, a2, a3 = color_transf.loc[band].values
				# Convert to PS mags
				slidx = pd.IndexSlice[:, band]
				x = self.df.loc[slidx, color]
				self.df.loc[slidx, 'mag_ps'] = self.df.loc[slidx, 'mag'] + a0 + a1*x + a2*(x**2) + a3*(x**3)

	def transform_avg_to_ps(self, colors, color, bands='gri', system=''):
		"""
		Transforms average magnitudes to PanSTARRS photometric system

		Parameters
		----------
		colors : dataframe of colors
		color  : specific color used for transform (mean_gi for Morganson, mean_gr for Tonry)
		"""	
		# This works for sdss because transform(average) - average(transform) is negligible. Cannot do this for ZTF.
		color_transf = pd.read_csv(wdir+'analysis/transformations/transf_to_ps_{}.txt'.format(system), sep='\s+', index_col=0)
		x = colors[color]
		for band in bands:
			a0, a1, a2, a3 = color_transf.loc[band].values
			# Convert to PS mags
			self.df_pivot['mean_ps_'+band] = self.df_pivot['mean_'+band] + a0 + a1*x + a2*(x**2) + a3*(x**3)

	def residual(self, corrections):
		for band in corrections.keys():
			self.df_pivot['mean_ps_'+band] += corrections[band]
	
	def residual_raw(self, corrections):
		for band in corrections.keys():
			self.df.loc[pd.IndexSlice[:, band], 'mag_ps'] += corrections[band]

	def calculate_offset(self, other, system, bands='gri'):
		for b in bands:
			self.df_pivot[self.name+'-'+other.name+'_'+b]	  = self.df_pivot['mean_'+b]	- other.df_pivot['mean_'+b]
			self.df_pivot[self.name+'\'-'+other.name+'\'_'+b] = self.df_pivot['mean'+system+'_'+b] - other.df_pivot['mean'+system+'_'+b]

	def calculate_pop_mask(self, band, bounds=[0,17,18,19,20]):
		mag = self.df_pivot['mean_'+band]
# 		self.masks = pd.DataFrame(data = {'17>mag':(mag<17),'17<mag<18':(17<mag)&(mag<18), '18<mag<19':(18<mag)&(mag<19), '19<mag<20':(19<mag)&(mag<20), 'mag>20':(mag>20)})
		self.masks = pd.DataFrame(data={str(lb)+'<mag<'+str(ub):(lb<mag)&(mag<ub) for lb, ub in np.array([bounds[:-1],bounds[1:]]).T})

############### Everything below is plotting/correlating, should move to different script.
	def plot_offset_distrib(self, other, bands='gri', scale='log', save=False, range=(-0.5,0.5), **kwargs):	  
		fig, axes = plt.subplots(len(bands),2,figsize=(23,23))
		for name, mask in self.masks.iteritems():
			data	   = self.df_pivot.loc[mask, [self.name+'-'+other.name+'_'	  +b for b in bands]]
			data_calib = self.df_pivot.loc[mask, [self.name+'\'-'+other.name+'\'_'+b for b in bands]]

			print('Offset for {} population (uncalibrated):\n{}\n'.format(name, data.mean()))
			print('Offset for {} population (calibrated):\n{}\n'.format(name, data_calib.mean()))

			data	  .hist(range=range, alpha=0.5, label=name, ax=axes[:,0], **kwargs)
			data_calib.hist(range=range, alpha=0.5, label=name, ax=axes[:,1], **kwargs)

		#print mean onto graph
		for b, ax in zip(bands,axes[:,0]):
			ax.text(0.05,0.70,'mean = {:.4f}' .format(self.df_pivot[self.name+'-'+other.name+'_'+b].mean()), transform=ax.transAxes)
			ax.text(0.05,0.62,'skew  = {:.4f}'.format(self.df_pivot[self.name+'-'+other.name+'_'+b].skew()), transform=ax.transAxes)
		for b, ax in zip(bands,axes[:,1]):
			
			hist, bin_edges = np.histogram(data_calib[self.name+'\'-'+other.name+'\'_'+b], range=range, **kwargs)
			idx = np.argmax(hist)
			maximum = (bin_edges[idx] + bin_edges[idx+1])*0.5
			
			ax.text(0.05,0.52,'max   = {:.4f}'.format(maximum), transform=ax.transAxes)
			ax.axvline(x=maximum, lw=0.9, ls='--', color='k')
			ax.text(0.05,0.70,'mean = {:.4f}' .format(self.df_pivot[self.name+'\'-'+other.name+'\'_'+b].mean()), transform=ax.transAxes)
			ax.text(0.05,0.61,'skew  = {:.4f}'.format(self.df_pivot[self.name+'\'-'+other.name+'\'_'+b].skew()), transform=ax.transAxes)
		for ax in axes.ravel():
			ax.axvline(x=0)
			ax.set(xlabel = 'offset', yscale=scale)
			ax.legend()
		plt.subplots_adjust(hspace = 0.4)
		
		plt.suptitle('mag error cut: {:.2f}'.format(self.magerr),  y=0.92)
		
		if save:
			fig.savefig('{}/plots/1dhist/{}-{}_magerr_{}.pdf'.format(self.obj, self.name, other.name, self.magerr_str), bbox_inches='tight')
		
		plt.subplots_adjust(hspace = 0.4)
		
		return axes

	def correlate_offset_sns(self, other, band, x, pop, vmin, vmax, xrange=(-0.3,2), yrange=(-0.1,0.1), save=False, cmap='jet'):
		"""
		Plot 2d histogram of offset and colour distributions. 

		Parameters
		----------
		other : class object to be used to calculate difference against
		colors: dataframe of colors
		band  : band used to calculate difference
		x     : quantity to plot against
		vmin  : min value of colorbar
		vmax  : max value of colorbar

		Raises
		------
		Error : descrip
		"""
		
		if len(x) > 6:
			string = 'color'
		else:
			string = 'mag'
		
		from matplotlib.colors import LogNorm
	#     fig, ax = plt.subplots(2,3,figsize=(21,14))
		bounds1={self.name+  '-'+other.name+  '_'+band:yrange, x:xrange}
		bounds2={self.name+'\'-'+other.name+'\'_'+band:yrange, x:xrange}

		mask1 = np.array([(bound1[0] < self.df_pivot[key]) & (self.df_pivot[key] < bound1[1]) for key, bound1 in bounds1.items()]).all(axis=0)
		mask2 = np.array([(bound2[0] < self.df_pivot[key]) & (self.df_pivot[key] < bound2[1]) for key, bound2 in bounds2.items()]).all(axis=0)

		xname = x
		yname1= self.name+  '-'+other.name+  '_'+band
		yname2= self.name+'\'-'+other.name+'\'_'+band
		data1 = self.df_pivot.loc[mask1]
		data2 = self.df_pivot.loc[mask2]
	#     with sns.axes_style('white'):
		plot1 = sns.jointplot(x=xname, y=yname1, data=data1, kind='hex', xlim=bounds1[xname], ylim=bounds1[yname1], norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
		plot2 = sns.jointplot(x=xname, y=yname2, data=data2, kind='hex', xlim=bounds2[xname], ylim=bounds2[yname2], norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)

		plot1.ax_joint.axhline(y=0, lw=0.5, ls='--')
		plot2.ax_joint.axhline(y=0, lw=0.5, ls='--')
		
		if save:
			plot1.savefig('{}/plots/2dhist/{}/{}/{}-{}/{}-{}_{}_{}_{}.pdf'    .format(self.obj, string, x, self.name, other.name, self.name, other.name, band, x, pop))
			plot2.savefig('{}/plots/2dhist/{}/{}/{}-{}/{}\'-{}\'_{}_{}_{}.pdf'.format(self.obj, string, x, self.name, other.name, self.name, other.name, band, x, pop))
		
	def correlate_mag_magerr_hist_sns(self, band, vmin, vmax, save=False):
		from matplotlib.colors import LogNorm

		if self.name == 'sdss':
			data = self.df
			xname = 'mag_'+band
			yname = 'magerr_'+band
		else:
			data = self.df.loc[pd.IndexSlice[:,band],:]
			xname = 'mag'
			yname = 'magerr'
		bounds={xname:(13,26), yname:(-0.01,1)}
		g = sns.JointGrid(x=xname, y=yname, data=data, xlim=bounds[xname], ylim=bounds[yname], height=9)
		g = g.plot_joint(plt.hexbin, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='Blues')
		g.ax_marg_x.hist(data[xname], bins=200)
		g.ax_marg_y.hist(data[yname], bins=200, orientation='horizontal', density = True)
	#     g.ax_marg_y.hist(data[yname], bins=200, orientation='horizontal', cumulative=True, alpha=0.2, color='k')#, yticks=[1e-3, 1e-1, 1])
		# Could show 95, 99, 99.9% intervals on magerr histplot? Will need to do np.stats.quantile and ax.axvline
		q = [0.85,0.95,0.99]
		quantiles = zip(np.quantile(data[yname],q),q)
		for q_val,q in quantiles:
			g.ax_marg_y.axhline(y=q_val, lw=2, ls='--', color='k')
			g.ax_marg_y.text(y=q_val+0.003,x=0.8, s=f'{q*100:.0f}%: {q_val:.2f}', fontdict={'size':12}, horizontalalignment='center')
		g.ax_marg_y.set(xscale='log')

		plt.suptitle(self.name + ' ' + band, x=0.1, y=0.95)
		if save:
			g.savefig('{}/plots/mag_magerr_{}_{}.pdf'.format(self.obj, self.name, band))