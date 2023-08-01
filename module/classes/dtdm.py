import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..config import cfg
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

class dtdm_raw_analysis():
	"""
	Class for analysing dtdm files
	"""
	def __init__(self, obj, band, name):
		self.obj = obj
		self.ID = 'uid' if (obj == 'qsos') else 'uid_s'
		self.band = band
		self.data_path = cfg.D_DIR + 'merged/{}/clean/dtdm_{}/'.format(obj,band)
		self.name = name
		# sort based on filesize, then do ordered shuffle so that each core recieves the same number of large files
		if os.path.exists(self.data_path):
			fnames = [a for a in os.listdir(self.data_path) if re.match('dtdm_[0-9]{5,7}_[0-9]{5,7}.csv', a)]
			size=[]
			for file in fnames:
				size.append(os.path.getsize(self.data_path+file))
			self.fnames = [name for i in [0,1,2,3] for sizename, name in sorted(zip(size, fnames))[i::4]]
			self.fpaths = [self.data_path + fname for fname in self.fnames]

	def read(self, i=0, **kwargs):
		"""
		Function for reading dtdm data
		"""
		self.df = pd.read_csv(self.fpaths[i], index_col = self.ID, dtype = {self.ID: np.uint32, 'dt': np.float32, 'dm': np.float32, 'de': np.float32, 'sid': np.uint8}, **kwargs)

	def read_key(self, key):
		"""
		Read in the groups of uids for qsos binned into given key.
		"""
		self.key = key
		path = cfg.D_DIR + 'computed/archive/{}/binned/{}/uids/'.format(self.obj, self.key)
		fnames = sorted([fname for fname in os.listdir(path) if fname.startswith('group')])
		self.groups = [pd.read_csv(path + fname, index_col=self.ID) for fname in fnames]
		self.n_groups = len(self.groups)
		self.bounds_values = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/bounds_values.txt'.format(self.obj, self.key))
		self.label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],self.key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}

	def bin_de_2d(self, n_chunks, read=False):
		"""
		2D binning into (de,dm) to see correlation between ∆m and ∆error and store them in:
		self.mean_tot
		self.std_tot
		self.median_tot

		Parameters
		----------
		n_chunks : int
			number of files to read in
		read : bool
			If True, read in, if False, use current self.df

		"""
		xbins = 100
		ybins = 100
		xlim   = (0,0.15)
		ylim   = (-0.2,0.2)
		self.de_edges = np.linspace(*xlim, xbins+1)
		self.dm_edges = np.linspace(*ylim, ybins+1)
		self.de_centres = (self.de_edges[1:]+self.de_edges[:-1])/2
		self.dm_centres = (self.dm_edges[1:]+self.dm_edges[:-1])/2
		self.total_counts = np.full((xbins,ybins),0, dtype='uint64')
		n_slices = 10
		self.mean_tot = np.zeros((n_slices,n_chunks))
		self.std_tot  = np.zeros((n_slices,n_chunks))
		self.median_tot  = np.zeros((n_slices,n_chunks))

		self.de_edges_stat = np.linspace(*xlim,n_slices+1)
		self.de_centres_stat = (self.de_edges_stat[1:]+self.de_edges_stat[:-1])/2

		for n in range(n_chunks):
			if read:
				self.read(n)
			counts = np.histogram2d(self.df['de'],self.df['dm'],range=(xlim,ylim), bins=(xbins, ybins))[0].astype('uint64')
			self.total_counts += counts

			std = []
			mean = []
			median = []
			for de1, de2 in zip(self.de_edges_stat[:-1], self.de_edges_stat[1:]):
				slice_ = self.df['dm'][(de1 < self.df['de']) & (self.df['de'] < de2)]
				std.append(slice_.std())
				mean.append(slice_.mean())
				median.append(slice_.median())

			self.mean_tot[:,n] = np.array(mean)
			self.std_tot[:,n]  = np.array(std)
			self.median_tot[:,n] = np.array(median)

	def bin_dt_2d(self, n_chunks, log_or_lin, read=False):
		"""
		2D binning into (dt,dm2_de2), attempt at a 2D structure function

		Parameters
		----------
		n_chunks : int
			number of files to read in
		read : b
			If True, read in, if False, use current self.df
		"""
		xbins = 100
		ybins = 100
		xlim   = [0.9,23988.3/20]
		ylim   = [0.0001,0.6] # dm2_de2**0.5
		self.dt_edges = np.logspace(*np.log10(xlim), xbins+1)
		# self.dt_edges = np.linspace(*xlim, xbins+1) # for linear binning
		self.dm2_de2_edges = np.linspace(*ylim, ybins+1)
		self.dt_centres = (self.dt_edges[1:]+self.dt_edges[:-1])/2
		self.dm2_de2_centres = (self.dm2_de2_edges[1:]+self.dm2_de2_edges[:-1])/2
		self.total_counts = np.full((xbins,ybins),0, dtype='uint64')
		n_slices = 10
		self.mean_tot = np.zeros((n_slices,n_chunks))
		# self.std_tot  = np.zeros((n_slices,n_chunks))
		# self.median_tot  = np.zeros((n_slices,n_chunks))

		self.dt_edges_stat = np.linspace(*xlim,n_slices+1)
		self.dt_centres_stat = (self.dt_edges_stat[1:]+self.dt_edges_stat[:-1])/2

		for n in range(n_chunks):
			if read:
				self.read(n)
			boolean = self.df['dm2_de2']>0
			counts = np.histogram2d(self.df[boolean]['dt'],self.df[boolean]['dm2_de2']**0.5,range=(xlim,ylim), bins=(xbins, ybins))[0].astype('uint64')
			self.total_counts += counts

			# std = []
			mean = []
			# median = []
			for dt1, dt2 in zip(self.dt_edges_stat[:-1], self.dt_edges_stat[1:]):
				slice_ = self.df['dm2_de2']
				 # std.append(slice_.std())
				mean.append((slice_**0.5).mean())
				 # median.append(slice_.median())

			self.mean_tot[:,n] = np.array(mean)
			# self.std_tot[:,n]  = np.array(std)
			# self.median_tot[:,n] = np.array(median)

	def plot_dm_hist(self):
		n = len(self.de_edges_stat)-1
		fig, ax = plt.subplots(n, 1, figsize=(20,5*n))
		for i in range(n):
			de1, de2 = (self.de_edges_stat[i], self.de_edges_stat[i+1])
			slice_ = self.df['dm'][(de1 < self.df['de']) & (self.df['de'] < de2)]
			ax[i].hist(slice_, bins=101, range=(-0.5,0.5), alpha=0.4)
			ax[i].legend()

	def plot_dm2_de2_hist(self, figax, bins, **kwargs):
		n = 20
		if figax is None:
			fig, ax = plt.subplots(n,1, figsize=(18,5*n))
		else:
			fig, ax = figax
		mjds = np.linspace(0, 24000, n+1)
		for i, edges in enumerate(zip(mjds[:-1], mjds[1:])):
			mjd_lower, mjd_upper = edges
			boolean = (mjd_lower < self.df['dt']) & (self.df['dt']<mjd_upper)
			print(boolean.sum())
			ax[i].hist(self.df[boolean]['dm2_de2'], range=kwargs['xlim'], alpha=0.5, bins=bins, label='{:.2f} < ∆t < {:.2f}'.format(*edges))
			ax[i].set(xlabel='$(m_i-m_j)^2 - \sigma_i^2 - \sigma_j^2$', **kwargs) #title='Distribution of individual corrected SF values'
		ax.set(yscale='log')
		for i in range(n):
			de1, de2 = (self.de_edges_stat[i], self.de_edges_stat[i+1])
			slice_ = self.df['dm2_de2'][(de1 < self.df['de']) & (self.df['de'] < de2)]
			ax[i].hist(slice_, bins=101, range=(-0.5,0.5), alpha=0.4)
			ax[i].legend()

	def read_pooled_stats(self, log_or_lin, key=None):
		self.log_or_lin = log_or_lin
		self.key = key
		fpath = cfg.D_DIR + f'computed/{self.obj}/dtdm_stats/{key}/{self.log_or_lin}/{self.band}/'
		names = os.listdir(fpath)

		if key == 'all':
			self.pooled_stats = ({name[7:-4].replace('_',' '):np.loadtxt(fpath+name) 
								  for name in names if name.startswith('pooled')})
			self.mjd_edges = np.loadtxt(fpath + 'mjd_edges.csv')
			self.mjd_centres = (self.mjd_edges[:-1] + self.mjd_edges[1:])/2
		else:
			self.bounds_values = np.loadtxt(fpath + 'bounds_values.csv')
			self.n_groups = len(self.bounds_values)-1
			self.pooled_stats = ({name[7:-6].replace('_',' '):np.array([np.loadtxt('{}{}_{}.csv'.format(fpath,name[:-6],i)) 
							      for i in range(self.n_groups)])
								  for name in names if name.startswith('pooled')})
			self.label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],self.key,self.bounds_values[i+1]) for i in range(self.n_groups)}
			self.mjd_edges = np.array([np.loadtxt(fpath + f'mjd_edges_{i}.csv') for i in range(self.n_groups)])
			self.mjd_centres = (self.mjd_edges[:, :-1] + self.mjd_edges[:, 1:])/2

	def plot_stats(self, keys, figax, label=None, **kwargs):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(10,6))
		else:
			fig, ax = figax
		if keys=='all':
			keys = list(self.pooled_stats.keys())[1:]
		
		# Norm by log
		# normalised_bin_counts = np.log(self.pooled_stats['n']) + 50
		# Norm by total max
		# normalised_bin_counts = self.pooled_stats['n']/np.max(self.pooled_stats['n'])*1e4
		# Norm per time bin
		normalised_bin_counts = self.pooled_stats['n']/self.pooled_stats['n'].sum(axis=0)*1e3

		for key in keys:
			y = self.pooled_stats[key]
			if key.startswith('SF') & (self.log_or_lin=='log'):
				y[y<0] = np.nan
			
			if label is None:
				label = '{}, {}'.format(self.name,key)
			# ax.errorbar(self.mjd_centres, y[:,0], yerr=y[:,1]**0.5, label='{}, {}'.format(key,self.name), color=color, lw=2.5) # square root this
			ax.errorbar(self.mjd_centres, y[:,0], yerr=y[:,1],
						capsize=5,
						lw=0.6,
						elinewidth=0.5,
						markeredgecolor=0.2)
			ax.scatter(self.mjd_centres, y[:,0], s=normalised_bin_counts,
	      			   label=label)
	
			ax.set(xlabel='Rest frame time lag (days)')

		ax.set(**kwargs)
		ax.legend()
		for handle in plt.legend().legend_handles:
			try:
				handle.set_sizes([70])
			except:
				pass
		# ax.set(xlabel='$(m_i-m_j)^2 - \sigma_i^2 - \sigma_j^2$', title='Distribution of individual corrected SF values', **kwargs)
		return (fig,ax)
	
	def fit_stats(self, key, model_name, ax=None, least_sq_kwargs={}, **kwargs):
		if model_name == 'power_law':
			from module.modelling.fitting import fit_power_law
			y, yerr = self.pooled_stats[key].T
			# yerr = 10*np.ones(y.shape) # to fit without errors
			coefficient, exponent, pcov, model_values = fit_power_law(self.mjd_centres, y, yerr, **kwargs)
			label = r'$\Delta t^{\beta}, \beta='+'{:.2f}'.format(exponent)+'$'
			fitted_params = (coefficient, exponent)			
		elif model_name == 'broken_power_law':
			from module.modelling.fitting import fit_broken_power_law
			y, yerr = self.pooled_stats[key].T
			# yerr = 10*np.ones(y.shape) # to fit without errors
			amplitude, break_point, index_1, index_2, pcov, model_values = fit_broken_power_law(self.mjd_centres, y, yerr, least_sq_kwargs=least_sq_kwargs, **kwargs)
			print(f'fitted broken power law:\ny = {amplitude:.2f}*x^{index_1:.2f} for x < {break_point:.2f}\ny = {amplitude:.2f}*x^{index_2:.2f} for x > {break_point:.2f}')
			label = 'broken power law'
			fitted_params = (amplitude, break_point, index_1, index_2)
		elif model_name == 'broken_power_law_minimize':
			from module.modelling.fitting import fit_minimize, cost_function
			from module.modelling.models import bkn_pow_smooth
			y, yerr = self.pooled_stats[key].T
			fitted_params, _, model_values = fit_minimize(bkn_pow_smooth, cost_function, self.mjd_centres, y, yerr, **kwargs)
			print(f'fitted broken power law:\ny = {fitted_params[0]:.2f}*x^{fitted_params[2]:.2f} for x < {fitted_params[1]:.2f}\ny = {fitted_params[0]:.2f}*x^{fitted_params[3]:.2f} for x > {fitted_params[1]:.2f}')
			label = 'broken power law'

		if ax is not None:
			ax.plot(*model_values, lw=2, ls='-.', label=label)		
		
		return fitted_params

	def plot_comparison_data(self, ax, name='macleod'):
		# f = lambda x: 0.01*(x**0.443)
		# ax.plot(self.mjd_centres, f(self.mjd_centres), lw=0.5, ls='--', color='b', label='MacLeod 2012')
		if name=='macleod':
			x,y = np.loadtxt(cfg.D_DIR + 'archive/Macleod2012/SF/macleod2012.csv', delimiter=',').T
		# elif name=='morganson':

		ax.scatter(x, y, color='k')
		ax.plot(x, y, label = 'Macleod 2012', lw=2, ls='--', color='k')

	def plot_stats_property(self, keys, figax, macleod=False, fill_between=False, **kwargs):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(10,7))
		else:
			fig, ax = figax
		if keys=='all':
			keys = list(self.pooled_stats.keys())[1:]
		
		# Norm by log
		# normalised_bin_counts = np.log(self.pooled_stats['n']) + 50
		# Norm by total max
		# normalised_bin_counts = self.pooled_stats['n']/np.max(self.pooled_stats['n'])*1e4
		# Norm per group
		normalised_bin_counts = self.pooled_stats['n']/self.pooled_stats['n'].sum(axis=0)*1e3
		# Norm per time bin
		# normalised_bin_counts = self.pooled_stats['n']/(self.pooled_stats['n'].sum(axis=1).reshape(-1,1))*2e3
		
		if fill_between:
			for key in keys:
				max_ = self.pooled_stats[key].max(axis=0)
				min_ = self.pooled_stats[key].min(axis=0)
				ax.fill_between(self.mjd_centres.max(axis=0), min_[:,0]-min_[:,1], max_[:,0]+max_[:,1], color='#ff7f0e', alpha=0.2,
						edgecolor='#C3610C', lw=2)
			# Don't show errorbars if we are showing the fill_between
			elinewdith = 0
			markeredgewidth = 0
		else:
			elinewdith = 0.2
			markeredgewidth = 0.2

		for group_idx in range(self.n_groups):
			for key in keys:
				y = self.pooled_stats[key][group_idx]
				mask = abs(y[:,0]-y[:,0].mean()) > 10*y[:,0].std() # Note this is a hack to remove the large values from the plot
				y[mask, :] = np.nan,np.nan 
				color = cmap.gist_earth(group_idx/self.n_groups)
				ax.errorbar(self.mjd_centres[group_idx], y[:,0], yerr=y[:,1],
							capsize=5,
							lw=0.5,
							elinewidth=elinewdith,
							markeredgewidth=markeredgewidth)#, color=color) # square root this
				# make the size of the scatter points proportional to the number of points in the bin
				ax.scatter(self.mjd_centres[group_idx], y[:,0], s=normalised_bin_counts[group_idx],
	       				   alpha=0.6,
						   label=self.label_range_val[group_idx])#, color=color)

		if macleod:
			f = lambda x: 0.01*(x**0.443)
			ax.plot(self.mjd_centres.mean(), f(self.mjd_centres.mean()), lw=0.1, ls='--', color='b', label='MacLeod 2012')
			x,y = np.loadtxt(cfg.D_DIR + 'Macleod2012/SF/macleod2012.csv', delimiter=',').T
			ax.scatter(x, y, label = 'macleod 2012')

		ax.legend()

		for handle in plt.legend().legend_handles:
			try:
				handle.set_sizes([70])
			except:
				pass

		ax.set(xlabel='Rest frame time lag (days)', title='{}, {}'.format(keys[0], self.obj), **kwargs)

		return (fig,ax)

	def contour_de(self):
		fig, ax = plt.subplots(1,1, figsize=(20,10))

		ax.contourf(self.de_centres, self.dm_centres, self.total_counts.T, cmap='jet')

		plt.scatter(self.de_centres_stat, self.median_tot.mean(axis=1), color = 'b')
		for m in [-1,0,1]:
			plt.scatter(self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k')
			plt.plot   (self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k', lw=0.5)

	def contour_dt(self):
		fig, ax = plt.subplots(1,1, figsize=(20,10))

		ax.contourf(self.dt_centres, self.dm2_de2_centres, self.total_counts.T, levels=np.logspace(0,3.4,50), cmap='jet')
		# ax.set(xscale='log', yscale='log')
		# plt.scatter(self.dt_centres_stat, self.median_tot.mean(axis=1), color = 'b')
		# for m in [-1,0,1]:
		# 	plt.scatter(self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k')
		# 	plt.plot   (self.de_centres_stat,self.mean_tot.mean(axis=1)+self.std_tot.mean(axis=1)*m, color='k', lw=0.5)