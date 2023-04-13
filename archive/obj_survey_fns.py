	def transform_to_sdss(self, bands = 'gri', colors=None):
		"""
		Adds column with transformed mags to DataFrame

		Parameters
		----------
		colors : DataFrame of mean colors

		"""
		color_transf = pd.read_csv('color_transf_coef_to_sdss.txt',sep='\s+',index_col=0)
		x = colors['mean_gi']
		for band in bands:
			a0, a1, a2, a3 = color_transf.loc[band].values
			# Convert to SDSS AB mags
			slidx = pd.IndexSlice[:,band]
			self.df.loc[slidx,'mag_sdss'] = self.df.loc[slidx,'mag'] + a0 + a1*x + a2*(x**2) + a3*(x**3)

	def transform_avg_to_sdss(self, colors, bands = 'gri'):
		if self.name == 'ps':
			color_transf = pd.read_csv('color_transf_coef_to_sdss.txt',sep='\s+',index_col=0)
			x = colors['mean_gi']
			for band in bands:
				a0, a1, a2, a3 = color_transf.loc[band].values
				# Convert to SDSS AB mags
				self.df_pivot['mean_sdss_'+band] = self.df_pivot['mean_'+band] - (a0 + a1*x + a2*(x**2) + a3*(x**3))

		if self.name == 'ztf':
			# first transform to ps
			color_transf = pd.read_csv('color_transf_coef_to_ps.txt',sep='\s+',index_col=0)
			x = colors['mean_gr']
			for band in bands:
				a0, a1, a2, a3 = color_transf.loc[band].values
				# Convert to PS mags
				self.df_pivot['mean_ps_'+band] = self.df_pivot['mean_'+band] + a0 + a1*x + a2*(x**2) + a3*(x**3)

			# second transform to sdss
			color_transf = pd.read_csv('color_transf_coef_to_sdss.txt',sep='\s+',index_col=0)
			x = colors['mean_gi']
			for band in bands:
				a0, a1, a2, a3 = color_transf.loc[band].values
				# Convert to SDSS AB mags
				self.df_pivot['mean_sdss_'+band] = self.df_pivot['mean_ps_'+band] - (a0 + a1*x + a2*(x**2) + a3*(x**3))

