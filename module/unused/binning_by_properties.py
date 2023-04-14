import numpy as np
import pandas as pd

def bin_data_by_properties(dtdm, n_bins_t, n_bins_m, t_max, n_t_chunk, t_spacing = 'log', m_spacing = 'log', compute=True, steepness=None, width=None):
	def calc_m_edges(n_bins_m, steepness, width):
		"""
		Produce a set of bins with max extent 'width'.
		Note that by keeping the steepness the same, the bin spacing is linear with width.
		"""
		steepness = steepness*width
		start = np.log10(steepness)
		stop = np.log10(steepness+width)
		bin_edges = np.concatenate((-np.logspace(start,stop,int(n_bins_m/2+1))[:0:-1]+steepness,np.logspace(start,stop,int(n_bins_m/2+1))-steepness))
		bin_widths = bin_edges[1:] - bin_edges[:-1]
		bin_centres = (bin_edges[1:] + bin_edges[:-1])/2 #this is linear rather than geometric
		return bin_edges, bin_widths, bin_centres

	def calc_t_bins(n_t_chunk, t_max, t_steepness):
		start = np.log10(t_steepness)
		stop = np.log10(t_steepness+t_max)
		return np.logspace(start, stop, n_t_chunk+1)-t_steepness


	#compute edges of bins
	m_bin_edges, m_bin_widths, m_bin_centres = calc_m_edges(n_bins_m, steepness, width)
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
		dcat	   = np.zeros((n_t_chunk,9), dtype = 'int64')
		dtdms	   = [np.empty((0,4))]*n_t_chunk

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
			dts_binned[i] = np.histogram(dtdms[i][:,0], t_bin_edges)[0] # removed +=
			dms_binned[i] = np.histogram(dtdms[i][:,1], m_bin_edges)[0]
			des_binned[i] = np.histogram(dtdms[i][:,2], e_bin_edges)[0]
			a			 = np.bincount( dtdms[i][:,3].astype('int64'))[4:]
			dcat[i]	   = np.pad(a, (0,9-len(a)), 'constant', constant_values=0)

		return dms_binned, dts_binned, des_binned, dcat

	else:
		return t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, m_bin_widths, e_bin_edges, t_dict
