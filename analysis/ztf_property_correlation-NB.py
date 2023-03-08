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

# # Notebook for correlating ZTF lightcurve statistics with black hole properties

from funcs.lc_analysis import multisurvey_prop
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('font', **{'size': 14})
import seaborn as sns


# + active=""
# DR12_VAC description
#
# RAdeg      Right Ascension in decimal degrees (J2000)
# DEdeg      Declination in decimal degrees (J2000)
# z          Redshift (visual from Paris et al. (2016))
# Mi         Absolute Magnitude i-band from Paris et al. 
# L5100      log10(L at 5100A in erg s^-1)  (1)
# eL5100     error in log10(L at 5100A in erg s^-1)  (1)
# L3000      log10(L at 3000A in erg s^-1)  (1)
# eL3000     error in log10(L at 3000A in erg s^-1)  (1)
# L1350      log10(L at 1350A in erg s^-1)  (1)
# eL1350     error in log10(L at 1350A in erg s^-1)  (1)
# MBH_MgII   log10(M_BH in units on M_SUN from MgII)  (1)
# MBH_CIV    log10(M_BH in units on M_SUN from CIV)  (1)  
# Lbol       log10(bolometric L in erg s^-1)  (1)
# eLbol      error in log10(bolometric L in erg s^-1)  (1)
# nEdd       log10(Eddington ratio)  (1)
#
# Note (1): 
#     -99.999 = no measurement.
#     -9.999  = no measurement or no uncertainty.
# This table is line-by-line match to the Quasar Data Release 12 
# from Paris et al. (2016)
# -

def reader(n_subarray):
    return pd.read_csv('/disk1/hrb/python/data/surveys/ztf/dr2/lc_{}.csv'.format(n_subarray), usecols = [0,1,2,3,4,5], index_col = 0, dtype = {'oid': np.uint64, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, 'uid': np.uint32})



# dr2_g = singlesurvey_prop('g')
# dr2_g.read_in(reader=reader);
dr2_r = multisurvey_prop('r')
dr2_r.read_in(reader=reader);

dr2_r.group(read_in=True)

dr.merge_with_catalogue(catalogue = 'dr12', remove_outliers=False, prop_range_any={'MBH_CIV':(5,13)})

# + active=""
# 80% of ZTF lightcurves have mjd_ptp > 300 days, and a max of 466 days.
# -

counts, bins = np.histogram(dr2_r.df_grouped['mjd_ptp'],bins = 50)

lower1, lower2 = bins[:-1][(counts == np.sort(counts)[-1]) | (counts == np.sort(counts)[-2])]
uppper1, upper2 = bins[1:][(counts == np.sort(counts)[-1]) | (counts == np.sort(counts)[-2])]

upper2


# +
# dr2_g.df.loc[75581].to_csv('g.csv')
# dr2_r.df.loc[75581].to_csv('r.csv')
# -

class vac():
    def __init__(self):
        self.prop_range = {'Mi':(-30,-20),'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48),'nEdd':(-3,0.5),'MBH_MgII':(7,11), 'MBH_CIV':(7,11)}
        self.df = pd.read_csv('/disk1/hrb/python/data/catalogues/SDSS_DR12Q_BH_matched.csv', index_col=16)
        self.df = self.df.rename(columns={'z':'redshift'});
    def filter_(self):
        for key in self.prop_range:
            lower, upper = self.prop_range[key]
            self.df = self.df[(lower < self.df[key]) & (self.df[key] < upper)]


dr12vac = vac()
dr2_r.merge_with_catalogue(dr12vac.df)
dr2_r.info()
# dr12vac.filter_()
prop_range = {'Mi':(-30,-20),'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48),'nEdd':(-3,0.5),'MBH_MgII':(7,11), 'MBH_CIV':(7,11)}
for key in prop_range:
    lower, upper = prop_range[key]
    filtered_properties = dr2_r.properties[(lower < dr2_r.properties[key]) & (dr2_r.properties[key] < upper)]

# +
dr2_r.plot_property_distributions(dr12vac.prop_range, n_width = 3, n_bins = 150, separate_catalogues=False)


# -

print('Total number of objects: {:,}'.format(len(dr2_r.properties)))
print('Number of objects within specified range: {:,}'.format(len(filtered_properties)))

axes = pd.plotting.scatter_matrix(filtered_properties[keys].sample(frac = 0.1),alpha=0.05, figsize = (14,14),diagonal = 'hist',hist_kwds={'bins':100});
# plt.savefig('correlation_pd.pdf',bbox_inches = 'tight')

features = ['MBH_CIV','Lbol','mag_std']
cbar_feature = ['MBH_CIV']
x = filtered_properties[features].sample(frac=0.1)


def scatter_3d(data, keys):
    """
    Plot keys in data

    Parameters
    ----------
    data: dataframe with keys in column
    
    keys : names of columns to be plotted. 
        In format [x,y,z,colorbar]
    """
    from mpl_toolkits.mplot3d import Axes3D
    xlabel, ylabel, zlabel, cbarlabel = keys
    x, y, z, c = data[keys].values.T
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    cbar_ax = ax.scatter(x, y, z, c = c, marker='o', s = 1)
    cbar = plt.colorbar(cbar_ax, shrink = 0.5)
    cbar.ax.set_ylabel(cbarlabel)
    ax.set(xlabel = xlabel, ylabel = ylabel, zlabel = zlabel);


# %matplotlib notebook
scatter_3d(filtered_properties,features+cbar_feature)


# ## Correlation plots

def correlation_plot(data, save = False):
    """
    Plot correlation matrix of data

    Parameters
    ----------
    data : dataframe
    
    save : set to True to save fig

    """
    g = sns.PairGrid(data, layout_pad = -0.2, height = 3)
    g.map_lower(plt.scatter, s = 10, marker = '.', alpha = 0.05)
    g.map_upper(sns.kdeplot, shade = True, n_levels = 10)
    g.map_diag(plt.hist, bins = 100);
    if save == True:
        plt.savefig('correlation_sns.pdf',bbox_inches = 'tight')


correlation_plot(x)

# ## PCA analysis
# ------------

# %matplotlib inline
correlation_plot(x)

# Standardize the data

qs = x['mag_std'].quantile(q=[0.15866,0.84134,0.97725]).values #-1sig, 1sig, 2sig


def assign_pop(x):
    if x < 0.10232226:
        return 1
    elif (x >= 0.10232226) & (x < 0.25250696):
        return 2
    elif (x >= 0.25250696) & (x < 0.39380597):
        return 3
    else:
        return 4


def pca_(x,n_components):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
#     X = pd.DataFrame(StandardScaler().fit_transform(x), columns = x.columns, index=x.index)
    pca = PCA(n_components)
    pc = pca.fit_transform(x)
    df_pc = pd.DataFrame(data = pc, columns = ['pc' + str(n) for n in range(1,n_components+1)], index=x.index)
    print('Variance in:')
    for n in range(0,n_components):
        print('pc{}: {:.3f}'.format(n+1,pca.explained_variance_ratio_[n]))
    print('sum: {:.3f}'.format(pca.explained_variance_ratio_.sum()))
    return pca, pd.concat([df_pc,x], axis = 1)
pca, df_pc = pca_(x,2)

df_pc

pca.components_

df_pc['variability'] = df_pc['mag_std'].apply(assign_pop)

#red are < -1sig, blue -1sig to 1sig, green 1sig to 2sig, black 2sig plus
# %matplotlib inline
fig,ax = plt.subplots(1,1,figsize = (10,10))
for target, color in enumerate('rbgk',1):
    subdf = df_pc[df_pc['variability']==target]
    ax.scatter(subdf['pc1'],subdf['pc2'], c = color,s=2, label = target)
ax.set(xlabel='principle component 1', ylabel='principle component 2')
ax.legend()

df_pc

# Plot MBH vs Lbol, color by vbility

fig,ax = plt.subplots(1,1,figsize = (10,10))
for target, color in enumerate('rbgk',1):
    subdf = df_pc[df_pc['variability']==target]
    ax.scatter(subdf['MBH_CIV'],subdf['Lbol'], c = color,s=2, label = target)
ax.set(xlabel='MBH', ylabel='Lbol')
ax.legend()


