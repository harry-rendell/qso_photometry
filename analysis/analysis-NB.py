# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
# from profilestats import profile
from scipy.stats import binned_statistic
from funcs.analysis.analysis import *
# %matplotlib inline

def reader(n_subarray):
    return pd.read_csv('../data/merged/{}/r_band/with_ssa/lc_{}.csv'.format(obj,n_subarray), comment='#', index_col = ID, dtype = {'catalogue': np.uint8, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})


# +
obj = 'qsos'
ID  = 'uid'
band = 'r'
redshift_bool = True

# obj = 'calibStars'
# ID  = 'uid_s'
# band = 'r'
# redshift_bool = False

# Here we load the analysis class. This has various attibutes and methods outlined in /funcs/analysis.py
# Examples:
# Photometry is in dr.df

# Grouped statistics is in dr.grouped
# DR12 VAC properties are in dr.properties
dr = analysis(ID, obj)
# -

dr.read_in(reader, redshift=redshift_bool)
dr.band = band
dr.group(keys = ['uid'],read_in=True, redshift=redshift_bool, survey = 'all')

dr.df_grouped

dr.df_grouped['n_prod'] = np.prod(dr.df_grouped.iloc[:,1:6].values, axis=1)

uids = dr.df_grouped.sort_values('n_prod', ascending=False).head(20).index

dr.plot_series(uids=uids, filtercodes='r')#, xlim=[58000, 59000])

dr.df['mjd'].min()

dr.df['redshift'].hist(bins=100)

dr.df[dr.df['catalogue']==1]

dr.df['mag'].max()

dr.group(keys = ['uid'],read_in=True, redshift=redshift_bool, survey='SSS')

plt.hist(dr.df_grouped['mag_mean'], bins=200)

dr.df_grouped['mag_mean'].max()

dr.df['catalogue'].unique()

dr.df.loc[5]

mask = dr.df.duplicated()

dr.df[dr.df['catalogue']==1]

dr.df[dr.df['catalogue']==5]

fig, ax = dr.plot_series([1], survey=3, filtercodes='r')
ax[0].set(xlim=[58300, 58400])
plt.xlabel('mjd',fontsize=20)
plt.ylabel('mag',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks([22,21.5,21,20.5],fontsize=18)

dr.df[['mjd','mag']].values.shape

dr.df_grouped

len(dr.df)

threshold_mag = 19

(dr.df['mag']<threshold_mag).sum()

dr.df['mag'].max()

np.savetxt('../data/computed/{}/binned/bright_{}/mag_lt_{}_uid.txt'.format(obj, threshold_mag, threshold_mag), dr.df_grouped[dr.df_grouped['mag_mean']<threshold_mag].index, fmt='%i')



subdf = dr.df[dr.df['catalogue']==1]

subdf['mjd'].describe()

# for mjd1,mjd2 in [(58100,58200),(58200,58300),(58300,58400),(58400,58500),(58500,58600),(58600,58700)]:
fig, ax = plt.subplots(1,1,figsize=(10,5))
# for mjd1,mjd2 in [(51000,51500),(51500,52000),(52000,52500),(52500,53000),(53000,53500),(53500,54000),(54000,54500),(54500,55000)]:
for mjd1,mjd2 in [(51000,52000),(52000,53000),(53000,54000),(54000,55000)]:
    subdf[((mjd1<subdf['mjd']) & (subdf['mjd']<mjd2))].hist('magerr',bins=100, label='{} < ∆t < {}'.format(mjd1,mjd2),ax=ax,alpha=0.5,range=(0,0.25))
ax.legend()


dr.df['magerr'].describe()

dr.df['magerr'].hist(bins=100)


def save_dtdm_rf(self, uids, time_key):

    sub_df = self.df[[time_key, 'mag', 'magerr', 'catalogue']].loc[uids]

    df = pd.DataFrame(columns=[self.ID, 'dt', 'dm', 'de', 'cat'])
    for uid, group in sub_df.groupby('uid'):
        #maybe groupby then iterrows? faster?
        mjd_mag = group[[time_key,'mag']].values
        magerr = group['magerr'].values
        cat	 = group['catalogue'].values
        n = len(mjd_mag)
        # dtdm defined as: ∆m = (m2 - m1), ∆t = (t2 - t1) where (t1, m1) is the first obs and (t2, m2) is the second obs.
        # Thus a negative ∆m corresponds to a brightening of the object
        unique_pair_indicies = np.triu_indices(n,1)

        dcat = 3*cat + cat[:,np.newaxis]
        dcat = dcat[unique_pair_indicies]

        dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
        dtdm = dtdm[unique_pair_indicies]
        dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

        dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
        dmagerr = dmagerr[unique_pair_indicies]

        duid = np.full(int(n*(n-1)/2),uid,dtype='uint32')
        # collate data to DataFrame and append
        df = df.append(pd.DataFrame(data={self.ID:duid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr,'cat':dcat}))

    if (uid % 500 == 0):
        print(uid)

    return df


uids = dr.df.index.unique()

from time import time
# times1 = []
uppers = [10]
for upper in uppers:
    start = time()
    df = save_dtdm_rf_2(dr, uids[:upper], 'mjd_rf')
    end = time()
    duration = end-start
    print('time taken: {:.2f}s'.format(end-start))
#     times.append(duration)

print(times1)

#well I suppose it takes the same amount of time to do dr.df.loc and groupby. Groupby looks nicer so lets go with that
uppers = [10,50,100,500,1000,2000,5000]
times = [4.35,4.64,4.99,12.4,38.4,146.,914.] #.iterrows
times1 = [4.55, 4.71, 4.857, 12.6, 38.8, 150.8, 927.8] # .loc
plt.plot(uppers,times) #2
plt.plot(uppers,times1) #1

bg_qsos = pd.read_csv('/disk1/hrb/quirky_quasars/data/bg92.csv', index_col='bg_id')

len(bg_qsos)

bg_uids = [ 76294, 171762, 201673, 204295, 238866, 259047, 260145, 265119,
            337997, 343954, 345712, 383141, 439973, 444240, 452269, 467721,
            492186]
uid = bg_uids[0]


def plot_series(self, uids, survey=None):
    """
    Plot lightcurve of given objects

    Parameters
    ----------
    uids : array_like
            uids of objects to plot
    catalogue : int
            Only plot data from given survey
    survey : 1 = SDSS, 2 = PS, 3 = ZTF

    """
    fig, axes = plt.subplots(len(uids),1,figsize = (20,3*len(uids)), sharex=True)
    if len(uids)==1:
        axes=[axes]

    for uid, ax in zip(uids,axes):
        single_obj = self.df.loc[uid].sort_values('mjd')
        for band in 'ugriz':
            single_band = single_obj[single_obj['filtercode']==band]
            if survey is not None:
                single_band = single_band[single_band['catalogue']==survey]
            for cat in single_band['catalogue'].unique():
                x = single_band[single_band['catalogue']==cat]
                ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 0.5, markersize = 2, marker = self.marker_dict[cat], label = self.survey_dict[cat]+' '+band, color = self.plt_color[band])
        ax.invert_yaxis()
        ax.set(xlabel='mjd', ylabel='mag')
#         ax.legend(loc=1)
        ax.text(0.02, 0.9, 'uid: {}'.format(uid), transform=ax.transAxes, fontsize=10)

    plt.subplots_adjust(hspace=0)

    return fig, ax


len(bg_uids)

fig, ax = plot_series(dr, bg_uids)

fig.savefig('bg92.pdf', bbox_inches='tight')

# ### Example: finding qsos given ra, dec

# > Define a list of ra and decs in the format below

# ra_dec = [[0.00531, -2.0332],[359.999615, 3.268586], [359.99851, -0.65588]]
ra_dec = bg_qsos[['ra','dec']]

found = dr.search(ra_dec, 1)
print(len(found))
found

bg_uids = bg_uids[bg_uids != 260652]

bg_uids = found.index

# bg_uids = found.index### Grouped statistics of the photometry

dr.df_grouped

# ### Load DR12 VAC catalogue

dr.merge_with_catalogue(catalogue='dr12_vac', remove_outliers=True, prop_range_any = {'MBH_MgII':(6,12), 'MBH_CIV':(6,12)})

# > Columns included in the value added catalogue:

list(dr.properties.columns)

dr.properties

output_notebook()

# > Example: plot photometry of quasars with uid: [6, 526352, 526344] (currently only r band)

dr.plot_series_bokeh(bg_uids)


