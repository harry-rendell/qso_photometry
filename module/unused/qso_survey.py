import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def reader(args):
	n_subarray, nrows = args
	return pd.read_csv(cfg.D_DIR + 'surveys/ztf/dr2_clr/lc_{}.csv'.format(n_subarray), usecols = [0,1,2,3,4,5,6,7,8], index_col = ['uid','filtercode'], nrows=nrows, dtype = {'oid': np.uint64, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, 'uid': np.uint32})

class qso_survey():
	def __init__(self, survey):
		self.name = survey
#		 self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}[band]
		
	def read_in_raw(self, nrows=None):
		
		if self.name == 'sdss':
			self.df = pd.read_csv(cfg.D_DIR + 'surveys/sdss/raw_sdss_secondary.csv',index_col='uid', nrows=nrows, usecols=[0,9,10,11,12,13,14,15,16,17,18,19])
			# Add columns for colors
			for b1, b2 in zip('gri','riz'):
				self.df[b1+'-'+b2] = self.df[b1+'psf'] - self.df[b2+'psf']
			self.df.rename(columns={**{b+'psf':'mag_'+b for b in 'ugriz'},**{b+'psferr':'magerr_'+b for b in 'ugriz'}}, inplace=True)

		elif self.name == 'ps':
			cols = ['uid', 'filter', 'obsTime',# 'objID',
					'psfFlux', 'psfFluxErr']
			dtype1 = {'uid': np.uint32, 'objID': np.uint64, 'obsTime': np.float64,'psfFlux': np.float32, 'psfFluxErr': np.float32}

			self.df = pd.read_csv(cfg.D_DIR + 'surveys/ps/ps_secondary.csv', index_col=['uid','filtercode'], nrows=nrows, usecols=[0,3,4,7,8])
			
			# Drop bad values
			self.df = self.df[self.df['psfFlux']!=0]

			self.df = self.df.rename(columns = {'obsTime': 'mjd', 'filter': 'filtercode'})
			self.df['mag'] = -2.5*np.log10(self.df['psfFlux']) + 8.90
			self.df['magerr'] = 1.086*self.df['psfFluxErr']/self.df['psfFlux']
			self.df = self.df.drop(['psfFlux','psfFluxErr'], axis = 1)
			
		elif self.name == 'ztf':
			if __name__ == 'module.qso_survey':
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
	
	def pivot(self, read_in=True):
		if self.name == 'sdss':
			if read_in:
				self.df_pivot = pd.read_csv(cfg.D_DIR + 'surveys/{}/gb.csv'.format(self.name), index_col='uid', dtype={'count':np.uint16}) 
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
				grouped = pd.read_csv(cfg.D_DIR + 'surveys/{}/gb.csv'.format(self.name), index_col=['uid','filtercode'], dtype={'count':np.uint16}) 

			elif not read_in:
				print('creating uid chunks to assign to each core')
				uids = np.array_split(self.df.index,4)
				uid_0 = uids[0].unique()
				uid_1 = np.setdiff1d(uids[1].unique(),uid_0,assume_unique=True)
				uid_2 = np.setdiff1d(uids[2].unique(),uid_1,assume_unique=True)
				uid_3 = np.setdiff1d(uids[3].unique(),uid_2,assume_unique=True)
				print('assigning chunk to each core')
				if __name__ == 'module.qso_survey':
					pool = Pool(4)
					df_list = pool.map(groupby_apply, [self.df.loc[uid_0],self.df.loc[uid_1],self.df.loc[uid_2],self.df.loc[uid_3]])
					grouped = pd.concat(df_list)
#			 self.grouped = grouped
			self.df_pivot = grouped.reset_index('filtercode').pivot(columns='filtercode', values=grouped.columns)
			self.df_pivot.columns = ["_".join(x) for x in self.df_pivot.columns.ravel()]

	def transform_to_ps(self, colors=None):
		# Transform original data using colors. Adds column '_ps' with color corrected data
		if self.name == 'ztf':
			self.df = self.df.join(colors, how='left', on='uid') #merge colors onto ztf df
			self.df = self.df.reset_index('uid').set_index(['uid','filtercode'])	   #add filtercode to index
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

	def calculate_offset(self, other, bands='gri'):
		for b in bands:
			self.df_pivot[self.name+'-'+other.name+'_'+b]	= self.df_pivot['mean_'+b]	- other.df_pivot['mean_'+b]
			self.df_pivot[self.name+'\'-'+other.name+'\'_'+b] = self.df_pivot['mean_ps_'+b] - other.df_pivot['mean_ps_'+b]

	def calculate_pop_mask(self, band, bounds=[0,17,18,19,20]):
		mag = self.df_pivot['mean_'+band]
# 		self.masks = pd.DataFrame(data = {'17>mag':(mag<17),'17<mag<18':(17<mag)&(mag<18), '18<mag<19':(18<mag)&(mag<19), '19<mag<20':(19<mag)&(mag<20), 'mag>20':(mag>20)})
		self.masks = pd.DataFrame(data={str(lb)+'<mag<'+str(ub):(lb<mag)&(mag<ub) for lb, ub in np.array([bounds[:-1],bounds[1:]]).T})

	def plot_offset_distrib(self, other, scale='log', save=False, range=(-0.5,0.5), **kwargs):	  
		fig, axes = plt.subplots(3,2,figsize=(23,18))
		bands = 'gri' #add this to argument?
		bins = 250
		for name, mask in self.masks.iteritems():
			data	   = self.df_pivot.loc[mask, [self.name+'-'+other.name+'_'	  +b for b in bands]]
			data_calib = self.df_pivot.loc[mask, [self.name+'\'-'+other.name+'\'_'+b for b in bands]]

			print('Offset for {} population: \n{}\n'.format(name, data.mean()))
			print('Offset for {} population: \n{}\n'.format(name, data_calib.mean()))

			data	  .hist(bins=bins, range=range, alpha=0.5, label=name, ax=axes[:,0], **kwargs)
			data_calib.hist(bins=bins, range=range, alpha=0.5, label=name, ax=axes[:,1], **kwargs)
    
		#print mean onto graph
		for i, ax in enumerate(axes[:,0]):
			ax.text(0.7,0.4,f'mean = {data.mean()[i]:.4f}'          , transform=ax.transAxes)
			ax.set(xlabel = 'offset')
		for i, ax in enumerate(axes[:,1]):
			ax.text(0.7,0.4,f'mean = {data_calib.mean()[i]:.4f}', transform=ax.transAxes)
			ax.set(xlabel = 'offset')
		
		if save:
			fig.savefig(self.name+'-'+other.name+'.pdf', bbox_inches='tight')
		
		plt.subplots_adjust(hspace = 0.4)
		
		return axes
			
	def correlate_offset_color_hist_sns(self, other, band, color, vmin, vmax):
		from matplotlib.colors import LogNorm
	#     fig, ax = plt.subplots(2,3,figsize=(21,14))
		bounds1={self.name+  '-'+other.name+  '_'+band:(-0.1,0.1), color:(-0.3,2)}
		bounds2={self.name+'\'-'+other.name+'\'_'+band:(-0.1,0.1), color:(-0.3,2)}

		mask1 = np.array([(bound1[0] < self.df_pivot[key]) & (self.df_pivot[key] < bound1[1]) for key, bound1 in bounds1.items()]).all(axis=0)
		mask2 = np.array([(bound2[0] < self.df_pivot[key]) & (self.df_pivot[key] < bound2[1]) for key, bound2 in bounds2.items()]).all(axis=0)

		xname = color
		yname1= self.name+  '-'+other.name+  '_'+band
		yname2= self.name+'\'-'+other.name+'\'_'+band
		data1 = self.df_pivot.loc[mask1]
		data2 = self.df_pivot.loc[mask2]
	#     with sns.axes_style('white'):
		plot1 = sns.jointplot(x=xname, y=yname1, data=data1, kind='hex', xlim=bounds1[xname], ylim=bounds1[yname1], norm=LogNorm(vmin=vmin, vmax=vmax))
		plot2 = sns.jointplot(x=xname, y=yname2, data=data2, kind='hex', xlim=bounds2[xname], ylim=bounds2[yname2], norm=LogNorm(vmin=vmin, vmax=vmax))

		plot1.ax_joint.axhline(y=0, lw=0.5, ls='--')
		plot2.ax_joint.axhline(y=0, lw=0.5, ls='--')
		
	def correlate_mag_magerr_hist_sns(self, band, vmin, vmax, save=False):
		from matplotlib.colors import LogNorm

	#     fig, ax = plt.subplots(2,3,figsize=(21,14))
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
			g.savefig('mag_magerr_'+self.name+'_'+band+'.pdf')