	def struc_func(self, n_per_batch, n_batches, bin_edges, catalogue = None):
		"""
		Compute structure function

		Parameters
		----------
		n_per_batch : 
			desc

		Returns
		-------
		value : type
			desc
		"""
		"""
		input: dataframe of all observations. Filter for mag_count>2. Split by uid. Apply fn where we calculate dtdm
		and sample. Append this to larger set.
		output:
		"""
		n_sample = n_per_batch*n_batches

		if catalogue is not None:
			sub_uids = self.df_grouped[(self.df_grouped['mag_count'] > 2)].loc[pd.IndexSlice[:,1],:].index.get_level_values('uid').unique()
#			 self.df_grouped.loc[pd.IndexSlice[:,catalogue],:].index.get_level_values('uid')[(self.df_grouped['mag_count'] > 2)]
		else:
			sub_uids = self.df_grouped.index.get_level_values('uid')[(self.df_grouped['mag_count'] > 2)]

		sub_uids = np.random.choice(sub_uids,n_sample)
		sub_df = self.df[['mjd','mag']].loc[sub_uids]
		sub_uids_split = np.split(sub_uids,n_batches)

		total_tss = 0
		total_counts = 0

		#define uids_batch
		for idx,uid_batch in enumerate(sub_uids_split): #- first bin by black hole mass or luminosity??
			dtdms_unsorted = np.empty((0,2))
			print('Batch {}: {}'.format(idx,uid_batch))
			for uid in uid_batch:
				mjd_mag = sub_df.loc[uid].values
		#		 n = np.random.randint(len(mjd_mag))
		#		 n = int(10*truncexpon.rvs(len(mjd_mag)/10))
				dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
				dtdm = dtdm[np.triu_indices(len(mjd_mag),1)]
				dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

				dtdms_unsorted = np.append(dtdms_unsorted,dtdm, axis = 0)
			print('dtdm array length: {:,}'.format(len(dtdms_unsorted)))
			# Do in batches of 1000 objects, calculate tss and counts, add to previous, and move on.
			# Give batches to separate cores for multi-threading
			tss, _, _ = binned_statistic(dtdms_unsorted[:,0],dtdms_unsorted[:,1],lambda x: np.sum(x**2),bins=bin_edges) #try median too
			counts, _, _ = binned_statistic(dtdms_unsorted[:,0],dtdms_unsorted[:,1],'count', bins = bin_edges)

			total_tss += tss
			total_counts += counts
			print('total counts: {:,}'.format(total_counts.sum()))

		return tss, counts
