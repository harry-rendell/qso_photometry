import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..plotting.common import savefigs

def bin_data(dtdm, n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, leftmost_bin=None, t_spacing = 'log', m_spacing = 'log', compute=True, steepness=None, width=None):
	def calc_m_edges(n_bins_m_, steepness, width, leftmost_bin=None):
		"""
		Produce a set of bins with max extent 'width'.
		Note that by keeping the steepness the same, the bin spacing is linear with width.
		if leftmost_bin is not None, then first bin will begin from leftmost_bin. This is used for dm2_de2 which has an asymmetrical distribution about zero. 
			qsos:  bins=248 for dm2_de2, leftmost_bin = -0.244
			stars: bins=235 for dm2_de2, leftmost_bin = -0.21
		"""
		steepness = steepness*width
		start = np.log10(steepness)
		stop = np.log10(steepness+width)
		bin_edges = np.concatenate((-np.logspace(start,stop,int(n_bins_m_/2+1))[:0:-1]+steepness,np.logspace(start,stop,int(n_bins_m_/2+1))-steepness))
		if leftmost_bin is not None:
			bin_edges = bin_edges[abs(bin_edges-leftmost_bin).argmin():]
		bin_widths = bin_edges[1:] - bin_edges[:-1]
		bin_centres = (bin_edges[1:] + bin_edges[:-1])/2 # arthmetic mean
# 		bin_centres = np.sign(bin_edges[:-1])*(bin_edges[1:]*bin_edges[:-1]) ** 0.5 # geometric mean. Note, difference between geometric and arithmetic mean is only 0.0006 mag at most (at the edges) so it is not necessary.s
		return bin_edges, bin_widths, bin_centres

	def calc_t_bins(n_t_chunk, t_max, t_steepness):
		start = np.log10(t_steepness)
		stop = np.log10(t_steepness+t_max)
		return np.logspace(start, stop, n_t_chunk+1)-t_steepness

	#compute edges of bins
	m_bin_edges, m_bin_widths, m_bin_centres = calc_m_edges(n_bins_m, steepness, width)
	m2_bin_edges, m2_bin_widths, m2_bin_centres = calc_m_edges(n_bins_m2, steepness, width, leftmost_bin)
	t_bin_chunk = calc_t_bins(n_t_chunk, t_max, t_steepness = 1000)
	t_bin_edges = np.linspace(0,t_max,(n_bins_t+1))
	e_bin_edges = np.linspace(0,0.75,201) # This assumes that the max error pair is sqrt(0.5**2+0.5**2)

	# compute centre of bins
	t_bin_chunk_centres = (t_bin_chunk[1:] + t_bin_chunk[:-1])/2
	t_dict = dict(enumerate(['{0:1.0f}<t<{1:1.0f}'.format(t_bin_chunk[i],t_bin_chunk[i+1]) for i in range(len(t_bin_chunk)-1)]))

	if compute:
		dts_binned = np.zeros((n_t_chunk,n_bins_t), dtype = 'int64')
		dms_binned = np.zeros((n_t_chunk,n_bins_m), dtype = 'int64')
		des_binned = np.zeros((n_t_chunk,n_bins_m), dtype = 'int64')
		dm2_de2_binned = np.zeros((n_t_chunk,n_bins_m), dtype = 'int64') # This should stay as n_bins_m since we cut down the full array of length n_bins_m2
		dcat	   = np.zeros((n_t_chunk,122), dtype = 'int64')
		dtdms	   = [np.empty((0,5))]*n_t_chunk

		# First decide which larger bin chunk the data should be in
		idxs = np.digitize(dtdm[:,0], t_bin_chunk)-1
		for index in np.unique(idxs): #Can we vectorize this?
# 			try:
			dtdms[index] = np.append(dtdms[index],dtdm[(idxs == index),:],axis = 0)
# 			except:
# 				print(index)
# 				print(dtdm[(idxs == index),:])
# 				print('Could not bin into large time chunk')
# 				break
		# Bin up the data within each time chunk
		for i in range(n_t_chunk):
	#		 print('dtdm counts in {}: {:,}'.format(t_dict[i],len(dtdms[i])))
			dts_binned[i] = np.histogram(dtdms[i][:,0], t_bin_edges)[0] # here we should bin by a subset of t_bin_edges?
			dms_binned[i] = np.histogram(dtdms[i][:,1], m_bin_edges)[0]
			des_binned[i] = np.histogram(dtdms[i][:,2], e_bin_edges)[0]
			dm2_de2_binned[i] = np.histogram(dtdms[i][:,3], m2_bin_edges)[0]
			a			  = np.bincount( dtdms[i][:,4].astype('int64') )[4:]
			dcat[i]	   = np.pad(a, (0,122-len(a)), 'constant', constant_values=0)

		return dms_binned, dts_binned, des_binned, dm2_de2_binned, dcat

	else:
		return t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, m_bin_widths, e_bin_edges, t_dict, m2_bin_edges, m2_bin_widths, m2_bin_centres

def calculate_bins_and_z_scores(x, key, bounds = np.array([-6,-1.5,-1,-0.5,0,0.5,1,1.5,6]), plot=False, hist_kwargs={}, ax_kwargs={}):
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
	mean = x.mean()
	std  = x.std()
	z_score = (x-mean)/std
	bounds_values = bounds * std + mean
	for i in range(len(bounds)-1):
		# print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])&(self.properties['mag_count']>2)).sum()))
		print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])).sum()))
	bounds_tuple = list(zip(bounds[:-1],bounds[1:]))

	fig = None
	if plot:
		fig, ax = plt.subplots(1,1,figsize = (12,5))

		ax.hist(x, **hist_kwargs)
		for value, z in zip(bounds_values, bounds):
			ax.axvline(x=value, ymax=1, color = 'k', lw=0.5, ls='--')
			# ax.axvline(x=value, ymin=0.97, ymax=1, color = 'k', lw=0.5, ls='--') # If we prefer to have the numbers inside the plot, use two separate lines to make
			# a gap between text
			ax.text(x=value, y=1.01, s=r'${}\sigma$'.format(z), horizontalalignment='center', transform=ax.get_xaxis_transform(), fontsize='small')
		ax.set(xlim=[bounds_values[0],bounds_values[-1]], **ax_kwargs)


		# ax2 = ax.twiny()
		# hist_kwargs['alpha'] = 0
		# ax2.hist(z_score, **hist_kwargs)
		# for bound in bounds_values:
		# 	ax2.axvline(x=bound, color = 'k', lw=0.5, ls='--')
		
		# ax2.set_xticks(bounds)
		# ax2.set_xticklabels()

	return bounds_tuple, z_score, bounds_values, mean, std, fig
