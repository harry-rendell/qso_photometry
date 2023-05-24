import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module.preprocessing.binning import bin_data
from scipy.stats import binned_statistic, skew, skewtest, iqr, kurtosis, kurtosistest
from os import listdir, path
from multiprocessing import Pool
from time import time
import matplotlib.cm as cmap
from scipy.optimize import curve_fit
from scipy.stats import linregress, chisquare

wdir = cfg.W_DIR

class dtdm():
	"""
	Class for processing and analysing dtdm files

	Parameters
	----------
	obj : str
		qsos or calibStars
	name : str 
		name describing the survey pairs used to create the binned data. Eg sdss_ps or all
	label : str
		label to be used for plots
	n_bins_t : int
	n_bins_m : int
	n_bins_m2 : int
	t_max : int
	n_t_chunk : int
	steepness : float
	width : 
	leftmost_bin :
	verbose : bool
		If true, then print the number of dtdm counts in each ∆t bin
	subset :
	
	Notes from bin_data.calc_m_edges
	Produce a set of bins with max extent 'width'.
	Note that by keeping the steepness the same, the bin spacing is linear with width.
	if leftmost_bin is not None, then first bin will begin from leftmost_bin. This is used for dm2_de2 which has an asymmetrical distribution about zero. 
	qsos:  bins=248 for dm2_de2, leftmost_bin = -0.244
	stars: bins=235 for dm2_de2, leftmost_bin = -0.21

	"""	
	def __init__(self, obj, name, label, n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, steepness, width, leftmost_bin, subset='', verbose=False):
		self.obj = obj
		self.name = name
		self.label = label
		self.t_bin_edges, self.t_bin_chunk, self.t_bin_chunk_centres, self.m_bin_edges, self.m_bin_centres, self.m_bin_widths, self.e_bin_edges, self.t_dict, self.m2_bin_edges, self.m2_bin_widths, self.m2_bin_centres = bin_data(None, n_bins_t=n_bins_t, n_bins_m=n_bins_m, n_bins_m2=n_bins_m2, t_max=t_max, n_t_chunk=n_t_chunk, compute=False, steepness=steepness, width=width, leftmost_bin=leftmost_bin);		
		self.width = width
		self.subset = subset
		
		self.read()
		self.stats(verbose)

	def read(self):
		"""
		Read in binned data
		"""
		self.dts_binned = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dt/dts_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint64')
		self.dms_binned = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dm/dms_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint64')
		self.des_binned = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/de/des_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint64') # could get away with uint32, as the largest number we expect is ~2^29
		self.dcs_binned = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dc/dcs_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint16')
		self.dm2_de2_binned = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dm2_de2/dm2_de2_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint16')	

	def stats(self, verbose=False):
		if verbose:
			for i in range(19):
				print('dtdm counts in {}: {:,}'.format(self.t_dict[i],self.dts_binned.sum(axis=1)[i]))

		N_dm = self.dms_binned.sum(axis=1)
		x	= self.m_bin_centres*self.dms_binned
		mean = x.sum(axis=1)/N_dm

		self.means = mean
		self.modes = self.m_bin_centres[ (self.dms_binned/self.m_bin_widths).argmax(axis=1) ]
		self.modes = np.where(abs(self.modes)>1, np.nan, self.modes)

		self.skew_1  = skew(x, axis=1)
		self.skew_2  = N_dm**0.5 * ((x-mean[:,np.newaxis])**3).sum(axis=1) * ( ((x-mean[:,np.newaxis])**2).sum(axis=1) )**-1.5

	def plot_means(self, ax, ls='-'):
		ax.errorbar(self.t_bin_chunk_centres, self.means, yerr=self.means*self.dms_binned.sum(axis=1)**-0.5, lw=0.5, marker='o', label=self.label)
		 # ax.scatter(self.t_bin_chunk_centres, self.means, s=30, label=self.name, ls=ls)
		 # ax.plot   (self.t_bin_chunk_centres, self.means, lw=1.5, ls=ls)

	def plot_modes(self, ax, ls='-'):
		ax.errorbar(self.t_bin_chunk_centres, self.modes, yerr=self.means*self.dms_binned.sum(axis=1)**-0.5, lw=0.5, marker='o', label=self.label)
		 # ax.scatter(self.t_bin_chunk_centres, self.modes, s=30, label=self.name, ls=ls)
		 # ax.plot   (self.t_bin_chunk_centres, self.modes, lw=1.5, ls=ls)

	def plot_sf_ensemble(self, figax=None):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(15,8))
		else:
			fig, ax = figax
		SF = (((self.m_bin_centres**2)*self.dms_binned).sum(axis=1)/self.dms_binned.sum(axis=1))**0.5
		SF[self.dms_binned.sum(axis=1)**-0.5 > 0.1] = np.nan # remove unphysically large SF values
		# Check errors below a right
		ax.errorbar(self.t_bin_chunk_centres, SF, yerr=self.dms_binned.sum(axis=1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
		ax.set(yscale='log',xscale='log', xticks=[100,1000])
		ax.set(xlabel='∆t',ylabel = 'structure function')

		return figax, SF
	
	def plot_sf_ensemble_corrected(self, figax=None):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(15,8))
		else:
			fig, ax = figax
		SF = (((self.m_bin_centres**2)*self.dms_binned).sum(axis=1)/self.dms_binned.sum(axis=1))**0.5
		
		norm_cumsum_phot_err = self.des_binned.cumsum(axis=1)/self.des_binned.sum(axis=1)[:,np.newaxis]
		median_idx	 = np.abs(norm_cumsum_phot_err-0.5).argmin(axis=1)
		median_phot_err = self.e_bin_edges[median_idx]
		
		SF = np.sqrt(SF**2 - 2*median_phot_err**2)
		SF[self.dms_binned.sum(axis=1)**-0.5 > 0.1] = np.nan # remove unphysically large SF values
		# Check errors below a right
		ax.errorbar(self.t_bin_chunk_centres, SF, yerr=self.dms_binned.sum(axis=1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
		ax.set(yscale='log',xscale='log', xticks=[100,1000])
		ax.set(xlabel='∆t',ylabel = 'structure function')

		return figax, SF

	def plot_sf_ensemble_iqr(self, ax=None, **kwargs):
		if ax is None:
			fig, ax = plt.subplots(1,1, figsize=(15,8))
		norm_cumsum = self.dms_binned.cumsum(axis=1)/self.dms_binned.sum(axis=1)[:,np.newaxis]
		lq_idxs	 = np.abs(norm_cumsum-0.25).argmin(axis=1)
		uq_idxs	 = np.abs(norm_cumsum-0.75).argmin(axis=1)
		SF = 0.74 * ( self.m_bin_centres[uq_idxs]-self.m_bin_centres[lq_idxs] ) * (( self.dms_binned.sum(axis=1) - 1 ) ** -0.5)
		SF[SF == 0] = np.nan

		ax.errorbar(self.t_bin_chunk_centres, SF, yerr=self.dms_binned.sum(axis=1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
		ax.set(xticks=[100,1000], **kwargs)
		ax.set(xlabel='∆t',ylabel = 'structure function')
		return ax, SF
	
	def plot_sf_ensemble_asym(self, figax=None, color='b'):
		
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(15,8))
		else:
			fig, ax = figax

		SF_n = (((self.m_bin_centres[:100]**2)*self.dms_binned[:,:100]).sum(axis=1)/self.dms_binned[:,:100].sum(axis=1))**0.5
		SF_p = (((self.m_bin_centres[100:]**2)*self.dms_binned[:,100:]).sum(axis=1)/self.dms_binned[:,100:].sum(axis=1))**0.5
		SF_n[self.dms_binned[:,:100].sum(axis=1)**-0.5 > 0.1] = np.nan
		SF_p[self.dms_binned[:,100:].sum(axis=1)**-0.5 > 0.1] = np.nan
		
		# Check errors below are right
		ax.errorbar(self.t_bin_chunk_centres, SF_n, yerr=self.dms_binned[:, :100].sum(axis=1)**-0.5*self.means, ls='--', color=color, lw = 0.5, marker = 'o', label = self.label + ' negative')
		ax.errorbar(self.t_bin_chunk_centres, SF_p, yerr=self.dms_binned[:, 100:].sum(axis=1)**-0.5*self.means, ls='-', color=color, lw = 0.5, marker = 'o', label = self.label + ' positive') 

		ax.set(yscale='log',xscale='log', xticks=[100,1000])
		ax.set(xlabel='∆t',ylabel = 'structure function')

		return figax, SF_n, SF_p
	
	def hist_dm(self, window_width, overlay_gaussian=False, overlay_lorentzian=False, overlay_exponential=False, overlay_diff=False, cmap=plt.cm.cool, colors=['r','b','g'], alpha=1, save=False):

		def gaussian(x,peak,x0):
			sigma = (2*np.pi)**-0.5*1/peak
			return peak*np.exp( -( (x-x0)**2/(2*sigma**2) ) )
		def lorentzian(x,gam,x0):
			return gam / ( gam**2 + ( x - x0 )**2) * 1/np.pi
		def exponential(x,peak,x0,exponent):
			return peak*np.exp(-np.abs(x-x0)*exponent)
		
		fig, axes = plt.subplots(19,1,figsize = (15,3*19))
		n=1
		stds	 = np.zeros(19)
		r2s_tot = []
		for i, ax in enumerate(axes):
			r2s = []
			m,_,_= ax.hist(self.m_bin_edges[:-1], self.m_bin_edges[::n], weights = self.dms_binned[i], alpha = alpha, density=True, label = self.t_dict[i], color = colors[0]); # *0.25 is for jet color = cmap(0.6+i/20*0.5)
			ax.set(xlim=[-window_width,window_width], xlabel='∆m')
			# ax.axvline(x=modes[i], lw=0.5, ls='--', color='k')
			# ax.axvline(x=m_bin_centres[m.argmax()], lw=0.5, ls='--', color='r')
			text_height = 0.6
			if overlay_gaussian:
				#Also make sure that bins returned from .hist match m_bin_edges : it is
				x = np.linspace(-2,2,1000)

				popt, _ = curve_fit(gaussian, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
				# ax.plot(x,gaussian(x, m.max(), self.m_bin_edges[:-1:n][m.argmax()]), label = 'gaussian') what is this for
				ax.plot(x,gaussian(x, *popt), color=colors[1], label = 'gaussian', lw=1.5)

				# popt, _ = curve_fit(gaussian, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()]], sigma = 1/self.m_bin_widths)
				# ax.plot(x,gaussian(x, *popt), label = 'gaussian_weighted')
				
				slope, intercept, r, p, stderr = linregress(m, gaussian(self.m_bin_centres, *popt))
				chisq, p = chisquare(m, gaussian(self.m_bin_centres, *popt), ddof=0)
				diff	= m - gaussian(self.m_bin_centres, *popt)
				r2_gaussian = 1 -  (diff**2).sum() / ( (m - m.mean())**2).sum()
				r2s.append(r2_gaussian)
				if overlay_diff:
					ax.plot(self.m_bin_centres, diff, label = 'diff')
				ax.text(0.05, text_height, r'Gaussian     $r^2$ = {:.5f}'.format(r2_gaussian), transform=ax.transAxes)
				# ax.text(0.05, text_height-0.1, r'Gaussian linreg $r^2$ = {:.5f}'.format(r**2), transform=ax.transAxes)
				ax.text(0.05, text_height-0.2, r'Gaussian $\chi^2$ = {:.5f}'.format(chisq/m.sum()), transform=ax.transAxes)
######################### something wrong with chi squared
				text_height -= 0.1

			if overlay_lorentzian:
				x = np.linspace(-2,2,1000)

				popt, _ = curve_fit(lorentzian, self.m_bin_edges[:-1:n], m, p0 = [1/m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
				ax.plot(x,lorentzian(x,popt[0],popt[1]), color = colors[2], label = 'lorentzian', lw=2)
				
				# popt, _ = curve_fit(lorentzian, self.m_bin_edges[:-1:n], m, p0 = [1/m.max(),self.m_bin_edges[:-1:n][m.argmax()]], sigma = 1/self.m_bin_widths)
				# ax.plot(x,lorentzian(x,popt[0],popt[1]), label = 'lorentzian weighted')
				
				diff	= m - lorentzian(self.m_bin_centres, *popt)
				r2_lorentzian = 1 -  (diff**2).sum() / ( (m - m.mean())**2).sum() 
				r2s.append(r2_lorentzian)
				if overlay_diff:
					ax.plot(self.m_bin_centres, diff, label = 'diff')
				ax.text(0.05, text_height, r'Lorentzian  $r^2$ = {:.5f}'.format(r2_lorentzian), transform=ax.transAxes)
				text_height -= 0.1


			if overlay_exponential:
				# Also make sure that bins returned from .hist match m_bin_edges : it is
				x = np.linspace(-2,2,1000)

				popt, _ = curve_fit(exponential, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()], 1])
				# ax.plot(x,exponential(x, m.max(), self.m_bin_edges[:-1:n][m.argmax()]), label = 'exponential') what is this for
				ax.plot(x,exponential(x, *popt), color=colors[3], label = 'exponential', lw=2)

				# popt, _ = curve_fit(exponential, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()], 1], sigma = 1/self.m_bin_widths)
				# ax.plot(x,exponential(x, *popt), label = 'exponential_weighted')
				
				diff	= m - exponential(self.m_bin_centres, *popt)
				r2_exponential = 1 -  (diff**2).sum() / ( (m - m.mean())**2).sum() 
				r2s.append(r2_exponential)
				if overlay_diff:
					ax.plot(self.m_bin_centres, diff, label = 'diff')
				ax.text(0.05, text_height, r'Exponential $r^2$ = {:.5f}'.format(r2_exponential), transform=ax.transAxes)
				text_height -= 0.1

			r2s_tot.append(r2s)
			ax.text(0.05, 0.9, 'mean = {:.5f}'.format(self.means[i]), transform=ax.transAxes)
			ax.text(0.05, 0.8, 'mode = {:.5f}'.format(self.modes[i]), transform=ax.transAxes)
			ax.text(0.05, 0.7, 'skew  = {:.5f}'.format(self.skew_1[i]), transform=ax.transAxes)
			# ax.axvline(x=self.means[i], lw=0.4, ls='-', color='b')
			# ax.axvline(x=self.modes[i], lw=0.4, ls='-', color='r')
			# ax.axvline(x=self.m_bin_centres[m.argmax()], lw=0.6, ls='-', color='r')
			ax.axvline(x=0, lw=1, ls='--', color='k')
			ax.legend()
			plt.subplots_adjust(hspace=0.5)

		if save:
			fig.savefig(cfg.W_DIR + 'analysis/plots/{}_{}_dm_hist.pdf'.format(self.obj, self.name), bbox_inches='tight')
			plt.close()

		return fig, ax, np.array(r2s_tot).T

	def hist_de(self, window_width, overlay_gaussian=True, overlay_lorentzian=True, save=False):

		cmap = plt.cm.cool
		fig, axes = plt.subplots(19,1,figsize = (15,3*19))
		n=1
		stds	 = np.zeros(19)
		for i, ax in enumerate(axes):
			m,_,_= ax.hist(self.e_bin_edges[:-1], self.e_bin_edges[::n], weights = self.des_binned[i], alpha = 1, density=False, label = self.t_dict[i], color = cmap(i/20.0));
			ax.set(xlim=[0,window_width], xlabel='∆σ')
			ax.axvline(x=0, lw=0.5, ls='--')
			ax.legend()

		if save:
			fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_de_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
			plt.close()

		return fig, ax

	def hist_dt(self, overlay_gaussian=True, overlay_lorentzian=True, save=False):

		cmap = plt.cm.cool
		fig, axes = plt.subplots(19,1,figsize = (15,3*19))
		n=1
		stds	 = np.zeros(19)
		for i, ax in enumerate(axes):
			m,_,_= ax.hist(self.t_bin_edges[:-1], self.t_bin_edges[::n], weights = self.dts_binned[i], alpha = 1, density=True, label = self.t_dict[i], color = cmap(i/20.0));
			ax.set(xlim=[self.t_bin_chunk[i],self.t_bin_chunk[i+1]], xlabel='∆t')
			ax.axvline(x=0, lw=0.5, ls='--')
			ax.legend()

		if save:
			fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_dt_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
			plt.close()

		return fig, ax
	
	def hist_dt_stacked(self, overlay_gaussian=True, overlay_lorentzian=True, save=False):
		fig, ax = plt.subplots(1,1, figsize = (15,4))
		cmap = plt.cm.cool
		n=1
		stds	 = np.zeros(19)
		for i in range(19):
			m,_,_= ax.hist(self.t_bin_edges[:-1], self.t_bin_edges[::n], weights = self.dts_binned[i], alpha = 1, label = self.t_dict[i], color = cmap(i/20.0));
		ax.axvline(x=0, lw=0.5, ls='--')
		ax.set(yscale='log')
		# ax.legend()

		if save:
			fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_dt_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
			plt.close()

		return fig, ax


class dtdm_key():
	def __init__(self, obj, name, label, key, n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, steepness, width, leftmost_bin, verbose=False):
		self.obj = obj
		self.name = name
		self.label = label
		self.key = key

		self.n_t_chunk = n_t_chunk
		self.n_bins_t  = n_bins_t
		self.n_bins_m  = n_bins_m

		self.t_bin_edges, self.t_bin_chunk, self.t_bin_chunk_centres, self.m_bin_edges, self.m_bin_centres, self.m_bin_widths, self.e_bin_edges, self.t_dict, self.m2_bin_edges, self.m2_bin_widths, self.m2_bin_centres = bin_data(None, n_bins_t=n_bins_t, n_bins_m=n_bins_m, n_bins_m2=n_bins_m2, t_max=t_max, n_t_chunk=n_t_chunk, compute=False, steepness=steepness, width=width, leftmost_bin=leftmost_bin);

		self.width = width

		self.bounds_values = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/bounds_values.txt'.format(self.obj, self.key))
		self.label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],self.key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}

		self.read()
		# self.stats(verbose)

	def read(self):
		n = len([file for file in listdir(cfg.D_DIR + 'computed/{}/binned/{}/dm/'.format(self.obj,self.key)) if file.startswith('dms')])
		self.n = n
		self.dts_binned = np.zeros((n, self.n_t_chunk, self.n_bins_t), dtype = 'int64')
		self.dms_binned = np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
		self.dm2_de2_binned =  np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
		self.des_binned = np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
		self.dcs_binned = np.zeros((n, self.n_t_chunk, 9), dtype = 'int64')

		for i in range(n):

			self.dts_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dt/dts_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
			self.dms_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dm/dms_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
			self.dm2_de2_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dm2_de2/dm2_de2_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
			self.des_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/de/des_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint16')
			self.dcs_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/dc/dcs_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint16')

	def stats(self, verbose=False):
		if verbose:
			for i in range(19):
				print('dtdm counts in {}: {:,}'.format(self.t_dict[i],self.dts_binned.sum(axis=(0,-1))[i]))

		N_dm = self.dms_binned.sum(axis=-1)
		x	= self.m_bin_centres*self.dms_binned
		mean = x.sum(axis=-1)/N_dm

		self.means = mean
		self.modes = self.m_bin_centres[ (self.dms_binned/self.m_bin_widths).argmax(axis=-1) ]
		self.modes = np.where(abs(self.modes)>1, np.nan, self.modes)

		self.skew_1  = skew(x, axis=-1)
		self.skew_2  = N_dm**0.5 * ((x-mean[:,:,np.newaxis])**3).sum(axis=-1) * ( ((x-mean[:,:,np.newaxis])**2).sum(axis=-1) )**-1.5

	def plot_means(self, ax, ls='-'):
		for i in range(self.n):
			ax.errorbar(self.t_bin_chunk_centres, self.means[i], yerr=self.means[i]*(self.dms_binned[i].sum(axis=-1)**-0.5), lw=0.5, marker='o', label=self.label_range_val[i])
			# ax.scatter(self.t_bin_chunk_centres, self.means, s=30, label=self.name, ls=ls)
			# ax.plot   (self.t_bin_chunk_centres, self.means, lw=1.5, ls=ls)

	def plot_modes(self, ax, ls='-'):
		for i in range(self.n):
			ax.errorbar(self.t_bin_chunk_centres, self.modes[i], yerr=self.means[i]*(self.dms_binned[i].sum(axis=-1)**-0.5), lw=0.5, marker='o', label=self.label_range_val[i])
		 	# ax.scatter(self.t_bin_chunk_centres, self.modes, s=30, label=self.name, ls=ls)
		 	# ax.plot   (self.t_bin_chunk_centres, self.modes, lw=1.5, ls=ls)

	def plot_sf_ensemble(self, figax=None):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(15,8))
		else:
			fig, ax = figax

		for i in range(self.n):
			SF = (((self.m_bin_centres**2)*self.dms_binned[i]).sum(axis=-1)/self.dms_binned[i].sum(axis=-1))**0.5
			SF[self.dms_binned[i].sum(axis=-1)**-0.5 > 0.1] = np.nan
			# Check errors below a right
			ax.errorbar(self.t_bin_chunk_centres, SF, yerr=self.dms_binned[i].sum(axis=-1)**-0.5*self.means[i], lw = 0.5, marker = 'o', label=self.label_range_val[i])
		ax.set(yscale='log',xscale='log', xticks=[100,1000])
		ax.set(xlabel='∆t',ylabel = 'structure function')
		ax.legend()

		return figax, SF
	
	def plot_sf_dm2_de2(self, figax=None):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(15,8))
		else:
			fig, ax = figax
		
		for i in range(self.n):
			SF = (((self.m2_bin_centres**2)*self.dm2_de2_binned[i]).sum(axis=-1)/self.dm2_de2_binned[i].sum(axis=-1))
			# SF[self.dm2_de2_binned[i].sum(axis=-1)**-0.5 > 0.1] = np.nan
			# Check errors below a right
			ax.errorbar(self.t_bin_chunk_centres, SF, yerr=0, lw = 0.5, marker = 'o', label=self.label_range_val[i])
		ax.set(yscale='log',xscale='log', xticks=[100,1000])
		ax.set(xlabel='∆t',ylabel = 'structure function squared')
		ax.legend()
		
		return figax, SF
			
	def plot_sf_ensemble_iqr(self, ax=None):
		if ax is None:
			fig, ax = plt.subplots(1,1, figsize=(15,8))
		else:
			fig, ax = figax
		norm_cumsum = self.dms_binned.cumsum(axis=-1)/self.dms_binned.sum(axis=-1)[:,np.newaxis]
		lq_idxs	 = np.abs(norm_cumsum-0.25).argmin(axis=-1)
		uq_idxs	 = np.abs(norm_cumsum-0.75).argmin(axis=-1)

		SF = 0.74 * ( self.m_bin_centres[uq_idxs]-self.m_bin_centres[lq_idxs] ) * (( self.dms_binned.sum(axis=-1) - 1 ) ** -0.5)
		SF[SF == 0] = np.nan

		ax.errorbar(self.t_bin_chunk_centres, SF, yerr=self.dms_binned.sum(axis=-1)**-0.5*self.means, lw = 0.5, marker = 'o', label = self.label)
		ax.set(yscale='log',xscale='log', xticks=[100,1000])
		ax.set(xlabel='∆t',ylabel = 'structure function')

		return figax, SF
	
	def hist_dm(self, window_width, overlay_gaussian=True, overlay_lorentzian=True, save=False):

		def gaussian(x,peak,x0):
			sigma = (2*np.pi)**-0.5*1/peak
			return peak*np.exp( -( (x-x0)**2/(2*sigma**2) ) )
		def lorentzian(x,gam,x0):
			return gam / ( gam**2 + ( x - x0 )**2) * 1/np.pi

		cmap = plt.cm.cool
		fig, axes = plt.subplots(19,1,figsize = (15,3*19))
		n=1
		stds	 = np.zeros(19)
		for i, ax in enumerate(axes):
			m,_,_= ax.hist(self.m_bin_edges[:-1], self.m_bin_edges[::n], weights = self.dms_binned[i], alpha = 1, density=True, label = self.t_dict[i], color = cmap(i/20.0));
			ax.set(xlim=[-window_width,window_width])
			 # ax.axvline(x=modes[i], lw=0.5, ls='--', color='k')
			 # ax.axvline(x=m_bin_centres[m.argmax()], lw=0.5, ls='--', color='r')

			if overlay_gaussian:
				#Also make sure that bins returned from .hist match m_bin_edges : it is
				x = np.linspace(-2,2,1000)

				try:
					popt, _ = curve_fit(gaussian, self.m_bin_edges[:-1:n], m, p0 = [m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
					ax.plot(x,gaussian(x, m.max(), m_bin_edges[:-1:n][m.argmax()]), label = 'gaussian')

					diff	= m - gaussian(self.m_bin_centres,popt[0],popt[1]) # residuals
					ax.plot(self.m_bin_centres, diff, label = 'diff')

					print('x0: {:.5f}'.format(popt[1]))
				except:
					pass

			if overlay_lorentzian:
				x = np.linspace(-2,2,1000)

				try:
					popt, _ = curve_fit(lorentzian, self.m_bin_edges[:-1:n], m, p0 = [1/m.max(),self.m_bin_edges[:-1:n][m.argmax()]])
					ax.plot(x,lorentzian(x,popt[0],popt[1]), color = 'r', label = 'lorentzian')
					diff	= m - lorentzian(self.m_bin_centres,popt[0],popt[1])
					ax.plot(self.m_bin_centres, diff, label = 'diff')
					print('x0: {:.5f}'.format(popt[1]))
				except:
					pass

			ax.text(0.05, 0.9, 'mean: {:.5f}'.format(self.means[i]), transform=ax.transAxes)
			ax.text(0.05, 0.8, 'mode: {:.5f}'.format(self.modes[i]), transform=ax.transAxes)
			ax.text(0.05, 0.7, 'skew: {:.5f}'.format(self.skew_1[i]), transform=ax.transAxes)
			ax.axvline(x=self.means[i], lw=0.4, ls='-', color='b')
			ax.axvline(x=self.modes[i], lw=0.4, ls='-', color='r')
			ax.axvline(x=self.m_bin_centres[m.argmax()], lw=0.6, ls='-', color='r')

			ax.axvline(x=0, lw=0.5, ls='--')
			ax.legend()

		if save:
			fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_{}_dm_hist.pdf'.format(self.obj, self.obj, self.name), bbox_inches='tight')
			plt.close()

		return fig, ax
################################################################################################################################################################################################################################################################################################################################################################################################################################################
class dtdm_raw_analysis():
	"""
	Class for analysing dtdm files
	"""
	def __init__(self, obj, ID, band, name):
		self.obj = obj
		self.ID = ID
		self.band = band
		self.data_path = cfg.D_DIR + 'computed/{}/dtdm/raw/{}/'.format(obj,band)
		self.name = name
		# sort based on filesize, then do ordered shuffle so that each core recieves the same number of large files
		fnames = [a for a in listdir(self.data_path) if (a.startswith('dtdm_raw_{}_'.format(band)))]
		size=[]
		for file in fnames:
			size.append(path.getsize(self.data_path+file))
		self.fnames = [name for i in [0,1,2,3] for sizename, name in sorted(zip(size, fnames))[i::4]]
		self.fpaths = [self.data_path + fname for fname in self.fnames]

	def read_dtdm(self, fname):
		"""
		Function for reading dtdm data. Passed to self.read()
		"""
		df = pd.read_csv(self.data_path + fname, index_col = self.ID, dtype = {self.ID: np.uint32, 'dt': np.float32, 'dm': np.float32, 'de': np.float32, 'dm2_de2': np.float32, 'cat': np.uint8})
		return df.dropna()

	def read(self, n_chunk):
		"""
		Reading and concatenating dtdm data using multiprocessing

		Parameters
		----------
		n_chunk : int
			which chunk of data to read

		"""
		if hasattr(self, 'df'):
			del self.df
		if __name__ == 'module.preprocessing.dtdm':
			n_cores = 4
			p = Pool(n_cores)
			self.df = pd.concat(p.map(self.read_dtdm, self.fnames[n_cores*n_chunk:(n_chunk+1)*n_cores]))
		 # self.df = self.df[~self.df.isna().values.any(axis=1)] #drop na rows. This snippet is faster than self.df.dropna().
		 # self.df.loc[(self.df['cat']<4),'de'] += 5 # increase the error on plate pair observations by 0.05

	def read_key(self, key):
		"""
		Read in the groups of uids for qsos binned into given key.
		"""
		self.key = key
		path = cfg.D_DIR + 'computed/{}/binned/{}/uids/'.format(self.obj, self.key)
		fnames = sorted([fname for fname in listdir(path) if fname.startswith('group')])
		self.groups = [pd.read_csv(path + fname, index_col=self.ID) for fname in fnames]
		self.n_groups = len(self.groups)
		self.bounds_values = np.loadtxt(cfg.D_DIR + 'computed/{}/binned/{}/bounds_values.txt'.format(self.obj, self.key))
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


	def calculate_stats_looped(self, n_chunks, log_or_lin,  max_t=23576, save=False, inner=False):
		"""
		Loop over dtdm files and calculate stats of each file. Append to dictionary.
		Make sure to include name of desired quantites in names.
		
		Parameters
		----------
		n_chunks : int
			how many files to read in of files to read in.
			maximum value: stars = 200/4 = 50 (though 46 seems to be better, 50 runs of out memory), qsos = 52/4 = 13

		log_or_lin : str

		save : bool

		Returns
		-------
		results : dict of nd_arrays, shape (n_chunk, n_points)
		"""
		self.log_or_lin = log_or_lin
		n_points=20 # number of points to plot
		if log_or_lin.startswith('log'):
			self.mjd_edges = np.logspace(0, np.log10(max_t), n_points+1) # TODO add max t into argument
		elif log_or_lin.startswith('lin'):
			self.mjd_edges = np.linspace(0, max_t, n_points+1)

		self.mjd_centres = (self.mjd_edges[:-1] + self.mjd_edges[1:])/2

		# names = ['n','SF 1', 'SF 2', 'SF 3', 'SF 4', 'SF weighted', 'SF corrected', 'SF corrected weighted', 'SF corrected weighted fixed', 'SF corrected weighted fixed 2', 'mean', 'mean weighted']
		names = ['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'SF cwf p', 'SF cwf n', 'skewness', 'kurtosis']
		# names = ['SF cwf p', 'SF cwf n']
		results = {name:np.zeros(shape=(n_chunks, n_points, 2)) for name in names}
		results['n'] = np.zeros(shape=(n_chunks, n_points), dtype='uint64')

		pooled_results = {name:np.zeros(shape=(n_points, 2)) for name in names}
		pooled_results['n'] = np.zeros(shape=(n_points), dtype='uint64')
		start = time()
		for i in range(n_chunks):
			self.read(i)
			if inner:
				self.df = self.df[np.sqrt(self.df['cat'])%1==0]
			print('chunk: {}'.format(i))
			for j, edges in enumerate(zip(self.mjd_edges[:-1], self.mjd_edges[1:])):
				mjd_lower, mjd_upper = edges
				boolean = (mjd_lower < self.df['dt']) & (self.df['dt']<mjd_upper)# & (self.df['dm2_de2']>0) # include last condition to remove negative SF values
				print('number of points in {:.1f} < ∆t < {:.1f}: {}'.format(mjd_lower, mjd_upper, boolean.sum()))
				subset = self.df[boolean]
				# subset = subset[subset['dm2_de2']<1]
				# subset.loc[(subset['dm2_de2']<0).values,'dm2_de2'] = 0 # Include for setting negative SF values to zero. Need .values for mask to prevent pandas warning
				n = len(subset)
				results['n'][i,j] = n
				if n>0:
					# results['mean'][i,j, (0,1)] = subset['dm'].mean(), subset['dm'].std()
					# results['SF'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['de']**2).sum()/n
					# results['SF corrected'][i,j,(0,1)] = subset['dm2_de2'].mean(), subset['dm2_de2'].var()
					# results['SF a'][i,j,(0,1)] = (subset['dm']**2).mean(), 1/weights.sum()
					# results['SF b'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['dm']**2).var()


					weights = subset['de']**-2
					results['mean weighted a'][i,j,(0,1)] = np.average(subset['dm'], weights = weights), 1/weights.sum()
					results['mean weighted b'][i,j,(0,1)] = np.average(subset['dm'], weights = weights), subset['dm'].var() 
					if n>8:
						results['skewness'][i,j,(0,1)] = skew(subset['dm']), skewtest(subset['dm'])[1]
					if n>5:
						results['kurtosis'][i,j,(0,1)] = kurtosis(subset['dm']), kurtosistest(subset['dm'])[1]
					
					weights = 0.5*subset['de']**-4

					SF = np.average(subset['dm2_de2'], weights = weights)
					if SF < 0:
						SF = 0
					results['SF cwf a'][i,j, (0,1)] = SF, 1/weights.sum()
					results['SF cwf b'][i,j, (0,1)] = SF, subset['dm2_de2'].var() #we should square root this, but then it's too large
					
					mask_p = subset['dm']>0
					mask_n = subset['dm']<0
					
					try:
						SF_p = np.average(subset[mask_p]['dm2_de2'], weights = weights[mask_p])
						SF_n = np.average(subset[mask_n]['dm2_de2'], weights = weights[mask_n])
						if SF_p < 0:
							SF_p = 0
						if SF_n < 0:
							SF_n = 0
						results['SF cwf p'][i,j,(0,1)] = SF_p, 1/weights[mask_p].sum()
						results['SF cwf n'][i,j,(0,1)] = SF_n, 1/weights[mask_n].sum()
					except:
						print('weights cannot be normalized')
					
				else:
					print('number of points in bin:',n)
		self.results = results
		for key in results.keys():
			if key != 'n':
				pooled_mean = np.average(results[key][:,:,0], weights=results['n'], axis=0)
				pooled_var  = np.average(results[key][:,:,1], weights=results['n'], axis=0) + np.average((results[key][:,:,0]-pooled_mean)**2, weights=results['n'], axis=0)
				if key.startswith('SF'):
					pooled_results[key][:,0] = pooled_mean ** 0.5 # Square root to get SF instead of SF^2
					pooled_results[key][:,1] = pooled_var ** 0.5 # Square root to get std instead of var
				else:
					pooled_results[key][:,0] = pooled_mean
					pooled_results[key][:,1] = pooled_var
			else:
				pooled_results[key] = results[key].sum(axis=0)

		self.pooled_stats = pooled_results
		if save:
			for key in pooled_results.keys():
				np.savetxt(cfg.D_DIR + 'computed/{}/dtdm_stats/{}/pooled_{}.csv'.format(self.obj, self.log_or_lin, key.replace(' ','_')), pooled_results[key])
			np.savetxt(cfg.D_DIR + 'computed/{}/dtdm_stats/{}/mjd_edges.csv'.format(self.obj, self.log_or_lin), self.mjd_edges)
		print('time elapsed: {:.2f} minutes'.format((time()-start)/60.0))

	def calculate_stats_looped_key(self, n_chunks, log_or_lin, prop='Lbol', save=False):
		"""
		Loop over dtdm files and calculate stats of each file. Append to dictionary.

		Parameters
		----------
		n_chunks : int
			how many files to read in of files to read in.
			maximum value: stars = 200/4 = 50, qsos = 52/4 = 13

		log_or_lin : str

		save : bool

		Returns
		-------
		results : dict of nd_arrays, shape (n_chunk, n_points)
		"""
		self.log_or_lin = log_or_lin
		n_points=15 # number of points to plot
		if log_or_lin.startswith('log'):
			self.mjd_edges = np.logspace(0, 4.01, n_points+1) # TODO add max t into argument
		elif log_or_lin.startswith('lin'):
			self.mjd_edges = np.linspace(0, 10232.9, n_points+1)

		self.mjd_centres = (self.mjd_edges[:-1] + self.mjd_edges[1:])/2
		self.read_key(prop) # to give self.n_groups (and self.groups, but not needed yet)
		# names = ['n','SF 1', 'SF 2', 'SF 3', 'SF 4', 'SF weighted', 'SF corrected', 'SF corrected weighted', 'SF corrected weighted fixed', 'SF corrected weighted fixed 2', 'mean', 'mean weighted']
		names = ['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'skewness', 'kurtosis']
		results = {name:np.zeros(shape=(n_chunks, n_points, self.n_groups, 2)) for name in names}
		results['n'] = np.zeros(shape=(n_chunks, n_points, self.n_groups), dtype='uint64')

		pooled_results = {name:np.zeros(shape=(n_points, self.n_groups, 2)) for name in names} # this should have shape (self.n_groups, n_points) really, but then we have to go through and update the axes for results
		pooled_results['n'] = np.zeros(shape=(n_points, self.n_groups), dtype='uint64')
		start = time()
		for i in range(n_chunks):
			self.read(i)
			print('chunk: {}'.format(i))
			self.read_key(prop) #gives us self.groups
			for group_idx in range(self.n_groups):
				subgroup = self.df[self.df.index.isin(self.groups[group_idx].index)]
				print('subgroup: {}'.format(group_idx))
				print('\tmax ∆t: {:.2f}'.format(subgroup['dt'].max()))
				
				for j, edges in enumerate(zip(self.mjd_edges[:-1], self.mjd_edges[1:])):
					mjd_lower, mjd_upper = edges
					boolean = (mjd_lower < subgroup['dt']) & (subgroup['dt'] < mjd_upper)# & (subgroup['dm2_de2']>0) # include last condition to remove negative SF values
					subset = subgroup[boolean]
					# subset.loc[(subset['dm2_de2']<0).values,'dm2_de2'] = 0 # Include for setting negative SF values to zero. Need .values for mask to prevent pandas warning
					n = len(subset)
					results['n'][i,j, group_idx] = n
					print('\t\tnumber of points in {:.1f} < ∆t < {:.1f}: {}'.format(mjd_lower, mjd_upper, boolean.sum()))

					if n>0:
						# results['mean'][i,j, group_idx, (0,1)] = subset['dm'].mean(), subset['dm'].std()
						weights = subset['de']**-2
						results['mean weighted a'][i,j, group_idx, (0,1)] = np.average(subset['dm'], weights = weights), 1/weights.sum()
						results['mean weighted b'][i,j, group_idx, (0,1)] = np.average(subset['dm'], weights = weights), subset['dm'].var()
						results['skewness'][i,j, group_idx, (0,1)] = skew(subset['dm']), 0# skewtest(subset['dm'])[1]

						results['kurtosis'][i,j, group_idx, (0,1)] = kurtosis(subset['dm']), 0# kurtosistest(subset['dm'])[1]

						weights = 0.5*subset['de']**-4
						SF = np.average(subset['dm2_de2'], weights = weights)
						# mask = (SF < 0)
						# SF[mask] = 0
						results['SF cwf a'][i,j, group_idx, (0,1)] = SF, 1/weights.sum()
						results['SF cwf b'][i,j, group_idx, (0,1)] = SF, subset['dm2_de2'].var()

		#change below so it works with key
		self.results = results
		for key in results.keys():
			if key != 'n':
				pooled_mean = np.average(results[key][:,:,:,0], weights=results['n'], axis=0)
				pooled_var  = np.average(results[key][:,:,:,1], weights=results['n'], axis=0) + np.average((results[key][:,:,:,0]-pooled_mean)**2, weights=results['n'], axis=0) # this second term should be negligible since the SF of each chunk should be the same. # Note that the pooled variance is an estimate of the *common* variance of several populations with different means.
				if key.startswith('SF'):
					pooled_results[key][:,:,0] = pooled_mean ** 0.5 # Square root to get SF instead of SF^2
					pooled_results[key][:,:,1] = pooled_var ** 0.5 # Square root to get std instead of var
				else:
					pooled_results[key][:,:,0] = pooled_mean
					pooled_results[key][:,:,1] = pooled_var
			else:
				pooled_results[key] = results[key].sum(axis=0)
		
		self.pooled_stats = pooled_results
		if save:
			for key in pooled_results.keys():
				for group_idx in range(self.n_groups):
					np.savetxt(cfg.D_DIR + 'computed/{}/dtdm_stats/{}/{}/pooled_{}_{}.csv'.format(self.obj, prop, self.log_or_lin, key.replace(' ','_'), group_idx), pooled_results[key][:, group_idx])
			np.savetxt(cfg.D_DIR + 'computed/{}/dtdm_stats/{}/{}/mjd_edges.csv'.format(self.obj, prop, self.log_or_lin), self.mjd_edges)
		print('time elapsed: {:.2f} minutes'.format((time()-start)/60.0))

	def read_pooled_stats(self, log_or_lin, key=None):
		self.log_or_lin = log_or_lin
		if key is None:
			fpath = cfg.D_DIR + 'computed/{}/dtdm_stats/{}/'.format(self.obj, self.log_or_lin)
			names = listdir(fpath)
			self.pooled_stats = {name[7:-4].replace('_',' '):np.loadtxt(fpath+name) for name in names if name.startswith('pooled')}
		else:
			self.read_key(key)
			fpath = cfg.D_DIR + 'computed/{}/dtdm_stats/{}/{}/'.format(self.obj, key, self.log_or_lin)
			names = listdir(fpath)
			self.pooled_stats = {name[7:-6].replace('_',' '):np.array([np.loadtxt('{}{}_{}.csv'.format(fpath,name[:-6],i)) for i in range(self.n_groups)]) for name in names if name.startswith('pooled')}
		self.mjd_edges = np.loadtxt(fpath+'mjd_edges.csv')
		self.mjd_centres = (self.mjd_edges[:-1] + self.mjd_edges[1:])/2

	def plot_stats(self, keys, figax, macleod=False, fit=False, color='b', **kwargs):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(18,8))
		else:
			fig, ax = figax
		if keys=='all':
			keys = list(self.pooled_stats.keys())[1:]

		for key in keys:
			y = self.pooled_stats[key]
			# ax.errorbar(self.mjd_centres, y[:,0], yerr=y[:,1]**0.5, label='{}, {}'.format(key,self.name), color=color, lw=2.5) # square root this
			ax.errorbar(self.mjd_centres, y[:,0], yerr=y[:,1], label='{}, {}'.format(self.name,key), color=color, lw=2.5, capsize=10)
			ax.scatter(self.mjd_centres, y[:,0], color=color, s=80)
			ax.set(xlabel='Rest frame time lag (days)')

		if macleod:
			# f = lambda x: 0.01*(x**0.443)
			# ax.plot(self.mjd_centres, f(self.mjd_centres), lw=0.5, ls='--', color='b', label='MacLeod 2012')
			x,y = np.loadtxt(cfg.D_DIR + 'Macleod2012/SF/macleod2012.csv', delimiter=',').T
			ax.scatter(x, y, color='k')
			ax.plot(x, y, label = 'Macleod 2012', lw=2, ls='--', color='k')

		if fit:
			from scipy.optimize import curve_fit
			def power_law(x,a,b):
				return a*x**b
			for key in keys:
				y, err = self.pooled_stats[key].T
				popt, pcov = curve_fit(power_law, self.mjd_centres[10:], y[10:])
				x = np.logspace(-1,5,100)
				ax.plot(x, power_law(x, *popt), lw=2, ls='-.', label=r'$\Delta t^{\beta}, \beta='+'{:.2f}'.format(popt[1])+'$') #fix this
				print('Slope for {}: {:.2f}'.format(key, popt[1]))
		ax.set(**kwargs)
		ax.legend()
		# ax.set(xlabel='$(m_i-m_j)^2 - \sigma_i^2 - \sigma_j^2$', title='Distribution of individual corrected SF values', **kwargs)
		return (fig,ax)

	def plot_stats_property(self, keys, figax, macleod=False, fit=False, **kwargs):
		if figax is None:
			fig, ax = plt.subplots(1,1, figsize=(18,12))
		else:
			fig, ax = figax
		if keys=='all':
			keys = list(self.pooled_stats.keys())[1:]

		for group_idx in range(self.n_groups):
			for key in keys:
				y = self.pooled_stats[key][group_idx]
				color = cmap.jet(group_idx/self.n_groups)
				ax.errorbar(self.mjd_centres, y[:,0], yerr=y[:,1], label=self.label_range_val[group_idx], color=color, capsize=10) # square root this
				ax.scatter(self.mjd_centres, y[:,0], color=color)

		if macleod:
			f = lambda x: 0.01*(x**0.443)
			ax.plot(self.mjd_centres, f(self.mjd_centres), lw=0.5, ls='--', color='b', label='MacLeod 2012')
			x,y = np.loadtxt(cfg.D_DIR + 'Macleod2012/SF/macleod2012.csv', delimiter=',').T
			ax.scatter(x, y, label = 'macleod 2012')

		ax.legend()
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
