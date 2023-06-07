from multiprocessing import Pool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
# from bokeh.plotting import figure, output_notebook, show
# from bokeh.layouts import column
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse, lightcurve_statistics

def calc_moments(bins,weights):
	"""
	Calculate mean and kurtosis
	"""
	x = bins*weights
	z = (x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis]
	return x.mean(axis=1), (z**4).mean(axis = 1) - 3

class analysis():
	def __init__(self, ID, obj, band):
		self.ID = ID
		self.obj = obj
		self.band = band
		self.plt_color = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}
		self.plt_color_bokeh = {'u':'magenta', 'g':'green', 'r':'red','i':'black','z':'blue'}
		self.marker_dict = {1:'^', 3:'v', 5:'D', 7:'s', 11:'o'}
		self.marker_dict_bokeh = {1:'triangle', 3:'inverted_triangle', 5:'diamond', 7:'square', 11:'circle'}
		self.survey_dict = {1:'SSS_r1', 3: 'SSS_r2', 5:'SDSS', 7:'PS1', 11:'ZTF'}
		self.coords = pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/{obj}_subsample_coords.csv', index_col=ID, comment='#')

	# Imported methods
	from .methods.plotting import plot_sf_moments_pm
	from .methods.io import read_merged_photometry, read_grouped, read_vac, read_redshifts

	def summary(self):
		"""
		Run to create the following attributes:
		self.idx_uid : array_like
				unique set of uids
		self.uids_missing : array_like
				uids of objects which are present in DR14Q but not in observations
		self.n_qsos : int
				number of objects for which we have observations
		self.idx_cat : array_like
				list of surveys which contribute to observations (sdss=1, ps=2, ztf=3)
		"""
		# Check which qsos we are missing and which we have, given a list
		uids_complete	 = self.coords.index
		
		self.idx_uid	  = self.df.index.unique()
		self.uids_missing = uids_complete[~np.isin(uids_complete,self.idx_uid)]
		self.n_qsos		   = len(self.idx_uid)
		# self.idx_cat	  = self.df['catalogue'].unique()

		print('Number of qsos with lightcurve: {:,}'.format(self.n_qsos))
		print('Number of datapoints in:\nSDSS: {:,}\nPS: {:,}\nZTF: {:,}'.format((self.df['catalogue']==5).sum(),(self.df['catalogue']==7).sum(),(self.df['catalogue']==11).sum()))

	def sdss_quick_look(self, uids):
		if np.issubdtype(type(uids),np.integer): uids = [uids]
		coords = self.coords.loc[uids].values
		for ra, dec in coords:
			print("https://skyserver.sdss.org/dr18/VisualTools/quickobj?ra={}&dec={}".format(ra, dec))

	def merge_with_catalogue(self, catalogue='dr12_vac', remove_outliers=True, prop_range_any = {'MBH_MgII':(6,12), 'MBH_CIV':(6,12)}):
		"""
		Reduce self.df to intersection of self.df and catalogue.
		Compute summary() to reupdate idx_uid, uids_missing, n_qsos and idx_cat
		Create new DataFrame, self.properties, which is inner join of [df_grouped, vac] along uid.
		Add columns to self.properties:
				mag_abs_mean : mean absolute magnitude
				mjd_ptp_rf   : max ∆t in rest frame for given object

		Parameters
		----------
		catalogue : DataFrame
				value added catalogue to be used for analysis
		remove_outliers : boolean
				remove objects which have values outside range specified in prop_range_any
		prop_range_any : dict
				dictionary of {property_name : (lower_bound, upper_bound)}

		"""
		if catalogue == 'dr12_vac':
			# cols = z, Mi, L5100, L5100_err, L3000, L3000_err, L1350, L1350_err, MBH_MgII, MBH_CIV, Lbol, Lbol_err, nEdd, sdss_name, ra, dec, uid
			prop_range_all = {'Mi':(-30,-20),'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48),'nEdd':(-3,0.5)}
			self.prop_range = {**prop_range_all, **prop_range_any}
			vac = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr12q/SDSS_DR12Q_BH_matched.csv', index_col=self.ID)
			vac = vac.rename(columns={'z':'redshift_vac'});

		if catalogue == 'dr14_vac':
			# cols = ra, dec, uid, sdssID, plate, mjd, fiberID, z, pl_slope, pl_slope_err, EW_MgII_NA, EW_MgII_NA_ERR, FWHM_MgII_NA, FWHM_MgII_NA_ERR, FWHM_MgII_BR, FWHM_MgII_BR_ERR, EW_MgII_BR, EW_MgII_BR_ERR, MBH_CIV, MBH_CIV_ERR, MBH, MBH_ERR, Lbol

			prop_range_all = {'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48)}
			self.prop_range = {**prop_range_all, **prop_range_any}
			vac = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_spec_prop_matched.csv', index_col=self.ID)
			vac = vac.rename(columns={'z':'redshift_vac'});

		print(vac.index)
		self.df = self.df[self.df.index.isin(vac.index)]

		# Recalculate and print which qsos we are missing and which we have
		self.summary()
		self.properties = self.df_grouped.join(vac, how = 'inner', on=self.ID)
		self.vac = vac
		
		#calculate absolute magnitude
		self.properties['mag_abs_mean'] = self.properties['mag_mean'] - 5*np.log10(3.0/7.0*self.redshifts*(10**9))
		self.properties['mjd_ptp_rf']   = self.properties['mjd_ptp']/(1+self.redshifts)
	
	def merge_slopes():
		names = ['restframe_all','restframe_ztf']
		slopes = pd.concat([pd.read_csv(cfg.D_DIR + 'catalogues{}/slopes_{}.csv'.format(self.obj, name), index_col=self.ID, usecols=[self.ID,'m_optimal']) for name in names], axis=1)
		slopes.columns = []
		self.properties = self.properties.join(slopes, how='left', on=self.ID)

	def save_dtdm_rf_extras(self, df_ssa, uids, time_key):
		"""
		Similar to save_dtdm_rf except for plate data
		"""
		df = pd.DataFrame(columns=['uid', 'dt', 'dm', 'de', 'cat'])
		
		sub_df = df_ssa[[time_key, 'mag', 'magerr', 'catalogue']].loc[uids]
		
		for uid, group1 in sub_df.groupby(self.ID):
			group2 = self.df.loc[uid]
			mjd_mag1 = group1[[time_key, 'mag']].values # we need rest frame times
			mjd_mag2 = group2[[time_key,'mag']].values

			magerr1  = group1['magerr'].values
			magerr2  = group2['magerr'].values
			cat1 = group1['catalogue'].values
			cat2 = group2['catalogue'].values

			n = len(mjd_mag1)*len(mjd_mag2)
			dtdm = mjd_mag2[:, np.newaxis] - mjd_mag1
			dcat = cat2[:, np.newaxis]*cat1
			dmagerr = ( magerr2[:,np.newaxis]**2 + magerr1**2 )**0.5
			duid = np.full(n,uid,dtype='uint32')

			df = df.append(pd.DataFrame(data={self.ID:duid, 'dt':dtdm[:,:,0].ravel(), 'dm':dtdm[:,:,1].ravel(), 'de':dmagerr.ravel(), 'cat':dcat.ravel()}))
				
			if (uid % 500 == 0):
				print(uid)

		return df	
