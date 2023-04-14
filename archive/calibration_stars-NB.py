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

# # Assessing the magnitude distributions of the stars

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
font = {'size'   : 18}
matplotlib.rc('font', **font)

# from module.star_survey import star_survey#, reader
import sys
sys.path.append('../')
wdir = '/disk1/hrb/python/'
from module.analysis.obj_survey import obj_survey#, reader
from module.preprocessing.parse import intersection, filter_data

# # PS colors

ps_colors = ps.df.groupby(['uid_s','filtercode']).last().loc[pd.IndexSlice[:,['g','i']],'mag'].reset_index('filtercode').pivot(columns='filtercode',values='mag')
ps_colors.columns = ["_".join(x) for x in ps_colors.columns.ravel()]
ps_colors['gi'] = ps_colors['g']-ps_colors['i']

# # SDSS colours

colors = pd.read_csv(wdir+'data/computed/calibStars/colors_sdss.csv', index_col=0)

# +
colors = pd.DataFrame()
colors['gr'] = sdss.df['mag_g']-sdss.df['mag_r']
colors['ri'] = sdss.df['mag_r']-sdss.df['mag_i']
colors['gi'] = sdss.df['mag_g']-sdss.df['mag_i']
colors['iz'] = sdss.df['mag_i']-sdss.df['mag_z']
colors['ug'] = sdss.df['mag_u']-sdss.df['mag_g']

# bounds_clr = {'gr':(-0.6,1.6), 'ri':(-0.6,2), 'gi':(-0.6,2), 'iz':(-0.6,1.2), 'ug':(-0.6,3.5)}
bounds_clr = {color:(-0.6,3.5) for color in colors.columns}
colors = filter_data(colors, bounds_clr).dropna()
# -

# # Using raw (unaveraged) data

nrows = None
bounds_generic={'mag':(15,25),'magerr':(0,2)}

sdss = obj_survey('sdss', 'calibStars', 'uid_s')
sdss.read_in_raw(nrows)
sdss.df = filter_data(sdss.df, bounds_generic, bands = 'griz')

ps = obj_survey('ps', 'calibStars', 'uid_s')
ps.read_in_raw(nrows) 
ps.df = filter_data(ps.df, bounds_generic)

ztf = obj_survey('ztf', 'calibStars', 'uid_s')
ztf.read_in_raw(nrows)
ztf.df = filter_data(ztf.df, bounds_generic)

# # Saving groupby frames

# + [markdown] jupyter={"outputs_hidden": true}
# sdss.pivot(read_in=False, magerr=2)

# + [markdown] jupyter={"outputs_hidden": true}
# ztf.transform_to_ps(colors=colors)
# ztf.pivot(read_in=False, magerr=2)

# + [markdown] jupyter={"outputs_hidden": true}
# ps.pivot(read_in=False, magerr=2)

# + active=""
# ###########################################################
# -

# takes ~30s, could reduce but need to parallelise
sdss.df, ps.df, ztf.df = intersection(sdss.df,ps.df,ztf.df)
# sdss.df, ps.df = intersection(sdss.df,ps.df)

ztf.df = ztf.df.reset_index('uid_s').set_index(['uid_s','filtercode'])
# ztf.df

ztf.df.index.get_level_values('filtercode').unique()

ztf.transform_to_ps(colors=colors)

ps.df = ps.df.reset_index('filtercode')
ps.df

from module.plotting import plot_magerr_hist
ax = plot_magerr_hist([sdss, ps, ztf], bands='griz', quantiles=[0.09,0.10,0.11,0.12], show_lines=0, savename = 'plots/calibStars/stars_magerr_hist', magerr=0.6)

sdss.df

sdss.df['catalogue'] = 1
ps  .df['catalogue'] = 2
ztf .df['catalogue'] = 3
cols = ['mjd','mag_ps','magerr','catalogue']
master = pd.concat([sdss.df[cols].reset_index(), ps.df[cols].reset_index(), ztf.df[cols].reset_index()], axis = 0, ignore_index = True).astype({sdss.ID:'int'}).set_index([sdss.ID, 'catalogue', 'filtercode'])

for band in 'griz':
    chunks = np.array_split(master.loc[pd.IndexSlice[:, :, band], :].reset_index('filtercode', drop = True),4)
    for i in range(4):
        chunks[i].to_csv('/disk1/hrb/python/data/merged/qsos/lc_{}_{}.csv'.format(band, i))


def convert(mag_magerr):
    mag, magerr = mag_magerr.values.T
    flux     = 10**(-(mag-8.9)/2.5+6)
    flux_err = 0.92103403712*flux*magerr 
    return flux,flux_err


cols = [x for y in zip(['flux_'+b for b in 'ugriz'], ['fluxerr_'+b for b in 'ugriz']) for x in y]
a = pd.DataFrame(columns=cols)
for b in 'ugriz':
    flux, flux_err = convert(sdss.df[['mag_'+b,'magerr_'+b]])
    a['flux_'+b] = flux
    a['fluxerr_'+b] = flux_err

# +
################
# sdss.pivot(read_in=False, magerr='_magerr_007')

# + jupyter={"outputs_hidden": true}
frac = 0.05
# sns.jointplot(x='gr',y='ri',data=colors.sample(frac=frac), kind='scatter', s=0.01, xlim=[-0.6,1.8], ylim=[-0.55,2])
# sns.jointplot(x='ug',y='gr',data=colors.sample(frac=frac), kind='scatter', s=0.01, xlim=[-0.6,3.5], ylim=[-0.6,1.6])
# sns.jointplot(x='ri',y='iz',data=colors.sample(frac=frac), kind='scatter', s=0.01, xlim=[-0.6,2], ylim=[-0.51,1.1])
# sns.set(style="ticks")
sns.jointplot(x='gr',y='ri',data=colors.sample(frac=frac), kind='hex', xlim=[-0.6,1.8], ylim=[-0.55,2])
sns.jointplot(x='ug',y='gr',data=colors.sample(frac=frac), kind='hex', xlim=[-0.6,3.5], ylim=[-0.6,1.6])
sns.jointplot(x='ri',y='iz',data=colors.sample(frac=frac), kind='hex', xlim=[-0.6,2.0], ylim=[-0.51,1.1])

# +
# uid_oid = pd.read_csv('/disk1/hrb/python/data/surveys/ztf/calibStars/calibStars_ztf.txt', skiprows = 59, usecols = [3,6], names=['uid_s','oid'], sep='\s+', index_col = 'oid')
# mask = uid_oid['oid'].duplicated()
# uid_oid = uid_oid[mask]
# test = ztf.df.join(uid_oid,on='oid')
# test
# -

# Checking match between control star sample and quasars

# + jupyter={"outputs_hidden": true}
import seaborn as sns
sdss.correlate_mag_magerr_hist_sns('g', 2e0,1e6, save=0)
sdss.correlate_mag_magerr_hist_sns('r', 2e0,1e6, save=0)
sdss.correlate_mag_magerr_hist_sns('i', 2e0,1e6, save=0)
# -

# # Using averaged data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib
font = {'size' : 18}
matplotlib.rc('font', **font)

from module.obj_survey import obj_survey#, reader
from module.parse import intersection, filter_data


def compute_colors(survey):
    colors = pd.DataFrame()
    colors['mean_gi'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_i']
    colors['mean_gr'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_r']
    colors['mean_ri'] = survey.df_pivot['mean_r'] - survey.df_pivot['mean_i']
    colors['mean_iz'] = survey.df_pivot['mean_i'] - survey.df_pivot['mean_z']
    return colors


# colors = pd.read_csv('computed/colors_ps_last.csv', index_col=0)
colors = pd.read_csv('calibStars/computed/colors_sdss.csv', index_col=0)

# | band | max magerr |  SDSS |  PS   | ZTF   |
# |------|------------|-------|-------|-------|
# |  g   |     60%    | 0.0557| 0.0876| 0.1274|
# |  r   |     60%    | 0.0358| 0.0597| 0.1154|
# |  i   |     60%    | 0.0299| 0.0418| 0.0935|
# |  z   |     60%    | 0.0637| 0.0557|   -   |

# | band | max magerr |  SDSS |  PS   | ZTF   |
# |------|------------|-------|-------|-------|
# |  g   |     70%    | 0.0736| 0.1075| 0.1473|
# |  r   |     70%    | 0.0418| 0.0736| 0.1353|
# |  i   |     70%    | 0.0358| 0.0517| 0.1095|
# |  z   |     70%    | 0.0816| 0.0697|   -   |

# | band | max magerr |  SDSS |  PS   | ZTF   |
# |------|------------|-------|-------|-------|
# |  g   |     80%    | 0.0995| 0.1333| 0.1672|
# |  r   |     80%    | 0.0517| 0.0915| 0.1552|
# |  i   |     80%    | 0.0438| 0.0657| 0.1254|
# |  z   |     80%    | 0.1075| 0.0915|   -   |

# | band | max magerr |  SDSS |  PS   | ZTF   |
# |------|------------|-------|-------|-------|
# |  g   |     90%    | 0.1413| 0.1672| 0.1910|
# |  r   |     90%    | 0.0657| 0.1174| 0.1791|
# |  i   |     90%    | 0.0557| 0.0915| 0.1433|
# |  z   |     90%    | 0.1532| 0.1254|   -   |

# +
#if any observation has a value outside these bounds, that observation is NaN'd
bounds_generic  = {'mean':(15,25), 'meanerr':(0,2)} 
bounds_specific_fn = lambda x: {'magerr_max_'+band:value for band,value in zip('griz',zip([0,0,0,0],x))}
bands = 'griz'
magerr = 2
pop = 80

print('SDSS')
sdss = obj_survey('sdss', 'calibStars', 'uid_s')
sdss.pivot(magerr=magerr)
# for b in bands:
#     sdss.df_pivot['mean_sdss_'+b] = sdss.df_pivot['mean_'+b]
# sdss.df_pivot = filter_data(sdss.df_pivot, bounds_generic, bounds_specific_fn([0.056, 0.036, 0.03, 0.064]), bands) #60
# sdss.df_pivot = filter_data(sdss.df_pivot, bounds_generic, bounds_specific_fn([0.074, 0.042, 0.036, 0.082]), bands) #70
sdss.df_pivot = filter_data(sdss.df_pivot, bounds_generic, bounds_specific_fn([0.100, 0.052, 0.044, 0.110]), bands) #80
# sdss.df_pivot = filter_data(sdss.df_pivot, bounds_generic, bounds_specific_fn([0.141, 0.066, 0.056, 0.153]), bands) #90
sdss.transform_avg_to_ps(colors, 'mean_gr', 'griz', system='tonry')
sdss.residual({'g':0.0148, 'r':0.0049, 'i':0.0198, 'z':0.042})

# colors = compute_colors(sdss)
# sdss.df_pivot = filter_data(sdss.df_pivot, {'mean_gi':(0.4,2.7)}) 

print('\nPS')
ps = obj_survey('ps', 'calibStars', 'uid_s')
ps.pivot(magerr=magerr)
# ps.df_pivot = filter_data(ps.df_pivot, bounds_generic, bounds_specific_fn([0.088, 0.060, 0.042, 0.056]), bands) #60
# ps.df_pivot = filter_data(ps.df_pivot, bounds_generic, bounds_specific_fn([0.108, 0.074, 0.052, 0.070]), bands) #70
ps.df_pivot = filter_data(ps.df_pivot, bounds_generic, bounds_specific_fn([0.133, 0.092, 0.062, 0.092]), bands) #80
# ps.df_pivot = filter_data(ps.df_pivot, bounds_generic, bounds_specific_fn([0.167, 0.112, 0.092, 0.125]), bands) #90
# duplicating rows to keep the naing convention consistent
for b in bands:
    ps.df_pivot['mean_ps_'+b] = ps.df_pivot['mean_'+b]

bands='gri'
print('\nZTF')
ztf = obj_survey('ztf', 'calibStars', 'uid_s')
ztf.pivot(magerr=magerr)
# ztf.df_pivot = filter_data(ztf.df_pivot, bounds_generic, bounds_specific_fn([0.127, 0.115, 0.094]), bands) #60
# ztf.df_pivot = filter_data(ztf.df_pivot, bounds_generic, bounds_specific_fn([0.147, 0.135, 0.110]), bands) #70
ztf.df_pivot = filter_data(ztf.df_pivot, bounds_generic, bounds_specific_fn([0.167, 0.155, 0.125]), bands) #80
# ztf.df_pivot = filter_data(ztf.df_pivot, bounds_generic, bounds_specific_fn([0.191, 0.172, 0.143]), bands) #90
ztf.residual({'g':0.0074, 'r':-0.0099, 'i':0})

# -

sdss.df_pivot, ps.df_pivot, ztf.df_pivot = intersection(sdss.df_pivot, ps.df_pivot, ztf.df_pivot)
# sdss.df_pivot, ps.df_pivot  = intersection(sdss.df_pivot, ps.df_pivot)
# sdss.df_pivot, ztf.df_pivot = intersection(sdss.df_pivot, ztf.df_pivot)
# ztf.df_pivot , ps.df_pivot  = intersection(ztf.df_pivot , ps.df_pivot)

ps.df_pivot = ps.df_pivot.join(colors, on=ps.ID)
ztf.df_pivot = ztf.df_pivot.join(colors, on=ztf.ID)
# ps.transform_avg_to_sdss(colors, bands = 'griz')
# ztf.transform_avg_to_sdss(colors, bands = 'gri')

# +
# Cell below plots mag distribution for SDSS, PS, ZTF for each band, taking the intersection of all three surveys in each band in turn.

# +
def plot_mag_dist(inner=True, save=False):
	fig, ax = plt.subplots(4,2, figsize = (25,20))
	scaling_dict = {'g':24000,'r':35500,'i':35500}
	hist_bounds = {'mean':(15.5,23)}
	ylims  = [(0,12800),(0,8800),(0,4000),(0,7000)]
    
	for j, col in enumerate(['','ps_']):
		for i,band in enumerate('griz'):

			sdss_data = sdss.df_pivot['mean_'+col+band].dropna()
			ps_data   = ps  .df_pivot['mean_'+band].dropna()

			if band != 'z':
				ztf_data  = ztf .df_pivot['mean_'+col+band].dropna()
				if inner:
					sdss_data, ps_data, ztf_data = intersection(sdss_data, ps_data, ztf_data)
			else:
				if inner:
					sdss_data, ps_data = intersection(sdss_data, ps_data)
				del ztf_data

			sdss_data.hist(ax=ax[i,j], bins=200, range=hist_bounds['mean'], alpha = 0.5, label = 'sdss stars')
			ps_data  .hist(ax=ax[i,j], bins=200, range=hist_bounds['mean'], alpha = 0.5, label = 'ps stars')
			try:
				ztf_data .hist(ax=ax[i,j], bins=200, range=hist_bounds['mean'], alpha = 0.5, label = 'ztf stars')
			except:
				pass

			ax[i,j].set(title='mean_'+col+band, xlim=hist_bounds['mean'], ylim=ylims[i])
			ax[i,j].legend()
	plt.suptitle('mean mag error cut: {:.2f}'.format(bounds_generic['meanerr'][1]), y=0.95)

	if save==True:
		fig.savefig('calibStars/plots/star_distn_comparison_intersection.pdf', bbox_inches='tight')
        
plot_mag_dist(inner=True, save=1)
# -

# # Color corrections

mag_bounds = [15,21]
system='_ps'
bins=81

sdss_mag_g = sdss.df_pivot['mean_g']
sdss.calculate_pop_mask('g', bounds=mag_bounds)

ps.calculate_pop_mask('g', bounds=mag_bounds)
ps.calculate_offset(sdss, system=system, bands='griz')


ztf.calculate_pop_mask('g', bounds=mag_bounds)
ztf.calculate_offset(sdss, system=system, bands='gri')
ztf.calculate_offset(ps, system=system, bands='gri')


# #### Cells below plots the distribution of the offset (difference in average magnitude per quasar for each survey) for 5 populations of apparent brightness

axes = ztf.plot_offset_distrib(sdss, bands='gri', scale='linear', save=0, range=(-0.1,0.1), density=1, bins=bins)

axes = ps.plot_offset_distrib(sdss, bands='griz', scale='linear', save=0, range=(-0.1,0.1), density=1, bins=bins)

axes = ztf.plot_offset_distrib(ps, bands='gri', scale='linear', save=0, range=(-0.1,0.1), density=1, bins=bins)

# #### Cells below plots a 2D histogram of the offset vs colour (try combine this into a grid of subplots)

# +
cmap = 'jet'

# ps.correlate_offset_color_hist_sns(sdss,'g', 'mean_gr', 1e0,1e5)
# ps.correlate_ofset_color_hist_sns(sdss,'g', 'mean_ri', 1e0,1e5)

# ps.correlate_offset_color_hist_sns(sdss,'r', 'mean_gr', 1e0,1e5)
# ps.correlate_offset_color_hist_sns(sdss,'r', 'mean_ri', 1e0,1e5)

# ps.correlate_offset_sns(sdss, 'i', 'mean_gr', 1e0,1e5, yrange=(-0.1,0.1))
# ps.correlate_offset_sns(sdss,'i', 'mean_ri', 1e0,1e5)

# ps.correlate_offset_sns(sdss,'z', 'mean_gr', 1e0,1e5)
# ps.correlate_offset_sns(sdss,'z', 'mean_ri', 1e0,1e5)
xrange = (-0.5,2)
# xrange = (16,21)
ps.correlate_offset_sns(sdss, 'g', 'mean_gr', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)
ps.correlate_offset_sns(sdss, 'r', 'mean_ri', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)
ps.correlate_offset_sns(sdss, 'i', 'mean_iz', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)
# -

cmap = 'jet'
# xrange=(15,22.5)
ztf.correlate_offset_sns(ps, 'g', 'mean_gr', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)
ztf.correlate_offset_sns(ps, 'r', 'mean_ri', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)
ztf.correlate_offset_sns(ps, 'i', 'mean_iz', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)

cmap = 'jet'
# xrange=(15,22.5)
ztf.correlate_offset_sns(sdss, 'g', 'mean_gr', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)
ztf.correlate_offset_sns(sdss, 'r', 'mean_ri', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)
ztf.correlate_offset_sns(sdss, 'i', 'mean_iz', pop, 1e1,3e3, xrange=xrange, yrange=(-0.3,0.2), save=1, cmap=cmap)

# #### Cells below plots a distribution of errors on the mean magnitude for each population. We should repeat this for mag rather than meanmag.

# def plot_magerr(self, range, bins):
#     fig, ax = plt.subplots(3,1, figsize=(23,15))
#     for i, b in enumerate('gri'):
#         for name,mask in self.masks.iteritems():
#             self.df_pivot.loc[mask, 'meanerr_'+b].hist(bins=bins, range=range, ax=ax[i], label=name, density=True, alpha = 0.5)#, cumulative=True, density=True)
#      
#     #     sdss.df_pivot[['meanerr_' + b for b in 'gri']].hist(bins=100,range=(0,0.15), ax=ax[0,:], label='sdss', cumulative=True, density=True, alpha=0.5)
#     #     ps.  df_pivot[['meanerr_' + b for b in 'gri']].hist(bins=100,range=(0,0.06), ax=ax[1,:], label='ps', cumulative=True, density=True, alpha=0.5)
#     #     ztf. df_pivot[['meanerr_' + b for b in 'gri']].hist(bins=100,range=(0,0.1), ax=ax[2,:], label='ztf', cumulative=True, density=True, alpha=0.5)
#
#     for axis in ax:
#         axis.set(yscale='log')
#         axis.legend()

plot_magerr(ztf, range=(0,0.01), bins=150)

plot_magerr(ps, range=(0,0.001), bins=150)
