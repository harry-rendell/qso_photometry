###############################################################################################################

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
		self.dts_binned = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dt/dts_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint64')
		self.dms_binned = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm/dms_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint64')
		self.des_binned = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/de/des_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint64') # could get away with uint32, as the largest number we expect is ~2^29
		self.dcs_binned = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dc/dcs_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint16')
		self.dm2_de2_binned = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm2_de2/dm2_de2_binned_{}_{}.csv'.format(self.obj, self.subset, self.obj, self.name),  delimiter=',', dtype='uint16')	

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

		self.bounds_values = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/bounds_values.txt'.format(self.obj, self.key))
		self.label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(self.bounds_values[i],self.key,self.bounds_values[i+1]) for i in range(len(self.bounds_values)-1)}

		self.read()
		# self.stats(verbose)

	def read(self):
		n = len([file for file in os.listdir(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm/'.format(self.obj,self.key)) if file.startswith('dms')])
		self.n = n
		self.dts_binned = np.zeros((n, self.n_t_chunk, self.n_bins_t), dtype = 'int64')
		self.dms_binned = np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
		self.dm2_de2_binned =  np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
		self.des_binned = np.zeros((n, self.n_t_chunk, self.n_bins_m), dtype = 'int64')
		self.dcs_binned = np.zeros((n, self.n_t_chunk, 9), dtype = 'int64')

		for i in range(n):

			self.dts_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dt/dts_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
			self.dms_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm/dms_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
			self.dm2_de2_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dm2_de2/dm2_de2_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint64')
			self.des_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/de/des_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint16')
			self.dcs_binned[i] = np.loadtxt(cfg.D_DIR + 'computed/archive/{}/binned/{}/dc/dcs_binned_{}_{}_{}_{}.csv'.format(self.obj, self.key, self.obj, self.name, self.key, i),  delimiter=',', dtype='uint16')

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