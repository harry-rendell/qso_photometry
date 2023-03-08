import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_magerr_hist(surveys, bands, n_bins=201, quantiles=[0.05,0.1,0.15,0.2,0.25], show_lines=False, savename=None, magerr=None):
	"""
	Plots distribution and cumulative distribution of magnitude errors for each survey

	Parameters
	----------
	n_bins : number of bins in histogram
	quantiles : quantiles to use for finding how much of the population is below a certain magerr cut-off
	show_lines : plots horizontal and vertical lines indicating location of quantiles
	savename : if specified, the plot will be saved with name 'savename.pdf'
 	
	Returns
	-------
	ax : axes handle
	"""
	# find the error to cut at to keep fraction <value> of the population.
	def find_nearest(array, value):
		idx = (np.abs(array[1] - value)).argmin()
		return array[0,idx]

	
	sdss, ps, ztf = surveys
	fig, ax = plt.subplots(len(bands),1, figsize=(18,18))
	upper_bound = 0.4
	print('| band | max magerr |  SDSS |  PS   | ZTF   |\n|------|------------|-------|-------|-------|')
	for i,b in enumerate(bands):
		sdss_data = sdss.df['magerr_'+b]
		ps_data   = ps.df.loc[pd.IndexSlice[:,b],'magerr']
		try:
			ztf_data  = ztf.df.loc[pd.IndexSlice[:,b],'magerr']
		except:
			ztf_data = None
			
		ax_twin = ax[i].twinx()
		ax_twin.set(ylim=[0,1], ylabel='cumulative fraction')

		n, bins, _ = ax[i].hist(sdss_data, bins=n_bins, alpha=0.4, label='SDSS', range=(0,upper_bound), density=True, color='c')
# 		cum_frac_sdss = (bins[:-1]+(bins[1:]-bins[:-1]), np.cumsum(n)/n.sum())
		cum_frac_sdss = np.array([bins[:-1]+(bins[1:]-bins[:-1]), np.cumsum(n)/n.sum()])
		ax_twin.plot(cum_frac_sdss[0],cum_frac_sdss[1], color='c', ls='--')

		n, bins, _ = ax[i].hist(ps_data, bins=n_bins, alpha=0.4, label='PS'  , range=(0,upper_bound), density=True, color='r')
		cum_frac_ps = np.array([bins[:-1]+(bins[1:]-bins[:-1]), np.cumsum(n)/n.sum()])
		ax_twin.plot(cum_frac_ps[0],cum_frac_ps[1], color='r', ls='--')
		try:
			n, bins, _ = ax[i].hist(ztf_data,  bins=n_bins, alpha=0.4, label='ZTF' , range=(0,upper_bound), density=True, color='g')
			cum_frac_ztf = np.array([bins[:-1]+(bins[1:]-bins[:-1]), np.cumsum(n)/n.sum()])
			ax_twin.plot(cum_frac_ztf[0], cum_frac_ztf[1], color='g', ls='--')
		except:
			cum_frac_ztf=[]
			
		if show_lines:
			for x, ls in zip(quantiles,['--','-.','-',':','--']):
				idx = int(x*n_bins/upper_bound)
				if show_lines:
					ax_twin.axvline(x=x, color = 'k', lw=1, ls=ls)
					ax_twin.axhline(xmin=x*2.5, y=cum_frac_sdss[1][idx], color='c', lw=1.2, ls=ls, dashes=(5,10))
					ax_twin.axhline(xmin=x*2.5, y=cum_frac_ps  [1][idx], color='r', lw=1.2, ls=ls, dashes=(5,10))
					try:
						ax_twin.axhline(xmin=x*2.5, y=cum_frac_ztf [1][idx], color='g', lw=1.2, ls=ls, dashes=(5,10))
					except:
						pass
#			 print(f'observations with mag {b} < {x:.2f}:')
			if magerr is None:
				try:
					print(f'|  {b}   |	{x:.2f}	| {cum_frac_sdss[1][idx]*100:.1f}% | {cum_frac_ps  [1][idx]*100:.1f}% | {cum_frac_ztf [1][idx]*100:.1f}% |')
				except:
					print(f'|  {b}   |	{x:.2f}	| {cum_frac_sdss[1][idx]*100:.1f}% | {cum_frac_ps  [1][idx]*100:.1f}% | - |')

		if magerr is not None:
			try:
				print(f'|  {b}   |     80%    | {find_nearest(cum_frac_sdss, magerr):.4f}| {find_nearest(cum_frac_ps, magerr):.4f}| {find_nearest(cum_frac_ztf, magerr):.4f}|')
			except:
				print(f'|  {b}   |     80%    | {find_nearest(cum_frac_sdss, magerr):.4f}| {find_nearest(cum_frac_ps, magerr):.4f}|   -   |')

		ax[i].set(xlabel=f'{b} magnitude error', xlim=[0,upper_bound])
		ax[i].legend()
		plt.subplots_adjust(hspace=0.3)
		
		if savename is not None:
			fig.savefig('{}/plots/magerr_hist.pdf'.format(sdss.obj), bbox_inches='tight')
	
	return fig, ax
	
def plot_mag_dist(s1, s2, s3, inner=True, save=False):
	from funcs.parse import intersection
	"""
	Plots magnitude distribution for the intersection of all three surveys.

	Parameters
	----------
	s1 : survey1
	s2 : survey2
	s3 : survey3
	inner : bool
		if true, take the intersection of the surveys (so that we are only looking at the same quasars)
	save : bool
		if true, save plot.
	
	Returns
	-------
	ax
	"""
	fig, ax = plt.subplots(4,2, figsize = (25,20))
	scaling_dict = {'g':24000,'r':35500,'i':35500}
	hist_bounds = {'mean':(15.5,23)}
	ylims  = [(0,10000),(0,9500),(0,8000),(0,8000)]
    
	for j, col in enumerate(['','ps_']):
		for i,band in enumerate('griz'):
			print('band: {}\n'.format(band))
			sdss_data = s1.df_pivot['mean_'+col+band].dropna()
			ps_data   = s2.df_pivot['mean_'+band].dropna()

			if band != 'z':
				ztf_data  = s3.df_pivot['mean_'+col+band].dropna()
				if inner:
					sdss_data, ps_data, ztf_data = intersection(sdss_data, ps_data, ztf_data)
			else:
				if inner:
					sdss_data, ps_data = intersection(sdss_data, ps_data)
				del ztf_data

			sdss_data.hist(ax=ax[i,j], bins=200, range=hist_bounds['mean'], alpha = 0.5, label = 'sdss {}'.format(s1.obj))
			ps_data  .hist(ax=ax[i,j], bins=200, range=hist_bounds['mean'], alpha = 0.5, label = 'ps {}'.format(s1.obj))
			try:
				ztf_data .hist(ax=ax[i,j], bins=200, range=hist_bounds['mean'], alpha = 0.5, label = 'ztf {}'.format(s1.obj))
			except:
				pass

			ax[i,j].set(title='mean_'+col+band, xlim=hist_bounds['mean'], ylim=ylims[i])
			ax[i,j].legend()
	plt.suptitle('mag error cut: {:.2f}'.format(s1.magerr),  y=0.92)

	if save==True:
		fig.savefig('{}/plots/distn_comparison_intersection_magerr_{}.pdf'.format(s1.obj, s1.magerr_str), bbox_inches='tight')
    
	return ax

################################# Plotting dtdm results

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
#		 label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
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
#				 print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
#				 print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
		moments = calc_moments(m_bin_centres,dms_binned_norm)


		for idx, ax2 in enumerate(axes2.ravel()):
			ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
	#		 ax2.legend()
			ax2.set(xlabel='mjd', ylabel = label_moment[idx])
			ax2.axhline(y=0, lw=0.5, ls = '--')

#				 ax2.title.set_text(label_moment[idx])
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
#		 label_moment = ['mean', 'std', 'skew_stand', 'Excess kurtosis']
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
#				 print('number of -ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,:100].sum()))
#				 print('number of +ve dm in {} for {}: {:,}'.format(t_dict[j],label_range[i], dms_binned[j,100:].sum()))
		moments = calc_moments(m_bin_centres,dms_binned_norm)


		for idx, ax2 in enumerate(axes2.ravel()):
			ax2.plot(t_bin_chunk_centres, moments[idx], lw = 0.5, marker = 'o', label = label_range_val[i], color = cmap(i/10.0));
	#		 ax2.legend()
			ax2.set(xlabel='mjd', ylabel = label_moment[idx])
			ax2.axhline(y=0, lw=0.5, ls = '--')

#				 ax2.title.set_text(label_moment[idx])
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

def plot_series(self,uids,catalogue=None):
	"""
	Plot lightcurve of given objects

	Parameters
	----------
	uids : array_like
		uids of objects to plot
	catalogue : int
		Only plot data from given survey

	"""
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
