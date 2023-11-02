# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: astro
#     language: python
#     name: python3
# ---

# + language="bash"
# jupytext --to py cosmo_time_dilation.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb cosmo_time_dilation.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import data_io, parse, binning
from module.preprocessing.color_transform import calculate_wavelength
from module.assets import load_vac
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

obj = 'qsos'
ID = 'uid' if obj == 'qsos' else 'uid_s'

properties = load_vac(obj, usecols=['z','Lbol'])

# +
sig_str = 'sig50'
tau_str = 'tau50'
bounds = {'g':{sig_str:(-2, 0.15), tau_str:(1,5)}, 'r':{sig_str:(-2,0.15), tau_str:(1,5)}, 'i':{sig_str:(-2,0.15), tau_str:(1,5)}}
sigs = []
for band in 'gri':
    sig = pd.read_csv(cfg.D_DIR + f'computed/{obj}/mcmc_fits/mcmc_drw_fits_{band}_sdss_ps_ztf_100_best_phot_obs_frame.csv', index_col=ID)
    sig['band'] = band
    sig = sig.join(properties, on=ID)
    sig['wavelength'] = calculate_wavelength(band, sig['z'])
    sig = parse.filter_data(sig, bounds=bounds[band], dropna=False, verbose=True)
    sigs.append(sig)
    
sigs = pd.concat(sigs)

# +
# sns.pairplot(sigs[['logtau_obs','logsig','wavelength','logz','Lbol']].dropna(), kind='hist')

# +

vac = load_vac('qsos', usecols=['z','Lbol'])
skewfits = []
for band in 'gri':
    s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/skewfit_{band}_sdss_ps_ztf_100.csv", index_col=ID)
    s['band'] = band
    vac['wavelength'] = calculate_wavelength(band, vac['z'])
    s = s.join(vac, on=ID)
    skewfits.append(s)
skewfits = pd.concat(skewfits).dropna().sort_index()
skewfits = parse.filter_data(skewfits, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5)}, verbose=True)

mask_dict = parse.create_mask_lambda_lbol(skewfits, n = 15, l_low = 1000, l_high = 5000, L_low = 45.2, L_high = 47.2)
fits = [skewfits[['a','loc','scale','z']][mask].sample(100).values for mask in mask_dict.values()]
print(f'mcmc dimensionality: {len(fits)+1}')

# +
# Create matplotlib polygon with vertices at the bin edges of Lbol_edges and lambda_edges
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import itertools

n = 15
l_low = 1000
l_high = 5000
L_low = 45.2
L_high = 47.2

Lbol_edges   = np.linspace(L_low, L_high, n)
lambda_edges = np.linspace(l_low, l_high, n)

# # create a series of 2d bins from the edges
# Lbol_bins = pd.cut(sigs['Lbol'], Lbol_edges, labels=False)
# lambda_bins = pd.cut(sigs['wavelength'], lambda_edges, labels=False)

# # masks = [(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n-1), range(n-1))]
# masks_full = {(l,L):(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n-1), range(n-1))}
# masks = {key:value for key,value in masks_full.items() if (value.sum() > 250) and (key[0] % 3 == 0) and (key[1] % 3 == 0)}

masks = parse.create_mask_lambda_lbol(sigs, n, l_low, l_high, L_low, L_high)

fig, ax = plt.subplots(1,1, figsize=(10,10))
binrange = np.array([[l_low-100, l_high+100], [L_low-0.5, L_high+0.5]])
sns.histplot(data=sigs.reset_index(), x='wavelength',y='Lbol', bins=100, cmap='Spectral_r', binrange=binrange, ax=ax)

vertices = [[lambda_edges[i], Lbol_edges[j]] for i,j in masks.keys()]
squares = [Rectangle(vertex, width=(l_high-l_low)/(n-1), height=2/(n-1)) for vertex in vertices]
# add text to each square, showing the i, j indices

p = PatchCollection(squares, alpha=1, lw=2, ec='k', fc='none')
ax.add_collection(p)
ax.set(xlim=binrange[0], ylim=binrange[1], xlabel='wavelength', ylabel='Lbol')


for i, j in masks.keys():
    ax.text(lambda_edges[i]+2000/(n-1), Lbol_edges[j]+1/(n-1), s=f'({i},{j})', ha='center', va='center', fontsize=8)


# +
# Create matplotlib polygon with vertices at the bin edges of Lbol_edges and lambda_edges
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

groups = {1: (0,3), 2: (0,2), 3: (1,3), 4: (1,2), 5: (1,1), 6: (2,3), 7: (2,2), 8: (2,1), 9: (2,0), 10: (3,2), 11: (3,1), 12: (3,0)}
fig, ax = plt.subplots(1,1, figsize=(10,10))
binrange = np.array([[900, 6000], [45, 48]])
sns.histplot(data=sigs.reset_index(), x='wavelength',y='Lbol', bins=100, cmap='Spectral_r', binrange=binrange, ax=ax)

vertices = [[lambda_edges[i], Lbol_edges[j]] for i,j in groups.values()]
squares = [Rectangle(vertex, width=1000, height=0.5) for vertex in vertices]
p = PatchCollection(squares, alpha=1, lw=2, ec='k', fc='none')

ax.add_collection(p)
ax.set(xlim=(900, 4900), ylim=(45, 47.5), xlabel='wavelength', ylabel='Lbol')


# -

# Plotting data from Cosmo-time dilation Nature paper

tau_low = 2
bounds = {'tau16':(tau_low,4.5), 'tau50':(tau_low+0.4,5), 'tau84':(tau_low+0.4,6), 'sig16':(-1.1,0.5), 'sig50':(-1,0.5), 'sig84':(-0.9,0.5)}
sigs = parse.filter_data(sigs, bounds=bounds)

# +
cosmot = pd.read_csv(cfg.W_DIR + 'temp/TotalDat.csv', index_col=0)
cosmot['logz'] = np.log10(cosmot['Z']+1)

# load pickle file in cfg.W_DIR/temp
import pickle
with open(cfg.W_DIR + 'temp/SkewFit.pkl', 'rb') as f:
    skewfit_data = pickle.load(f)
# -

for band in 'gri':
    cosmot[f'tau16_{band}'] = cosmot[f'log_TAU_OBS_{band}'] - cosmot[f'log_TAU_OBS_{band}_ERR_L']
    cosmot[f'tau50_{band}'] = cosmot[f'log_TAU_OBS_{band}']
    cosmot[f'tau84_{band}'] = cosmot[f'log_TAU_OBS_{band}'] + cosmot[f'log_TAU_OBS_{band}_ERR_U']

    cosmot[f'sig16_{band}'] = cosmot[f'log_SIGMA_{band}'] - cosmot[f'log_SIGMA_{band}_ERR_L']
    cosmot[f'sig50_{band}'] = cosmot[f'log_SIGMA_{band}']
    cosmot[f'sig84_{band}'] = cosmot[f'log_SIGMA_{band}'] + cosmot[f'log_SIGMA_{band}_ERR_U']

bands = 'r'
cosmo = cosmot[[f'{key}_{x}' for key in ['sig16','sig50','sig84','tau16','tau50','tau84'] for x in bands]]

cosmo.describe()

sigs.describe()

pd.DataFrame(skewfit_data).describe()

tau_z = sigs[['tau16','tau50','tau84','logz']].dropna()

tau_z = parse.filter_data(tau_z, bounds={'tau16':(0,5), 'tau50':(1,5), 'tau84':(1,6)}, dropna=True, verbose=True)

vac = load_vac('qsos', usecols=['z','Lbol'])
vac['logz'] = np.log10(vac['z']+1)
vac['wavelength'] = calculate_wavelength('g', vac['z'])

from module.preprocessing import color_transform
vac = load_vac('qsos', usecols=['z','Lbol'])
skewfits = []
for band in 'gri':
    s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/skewfit_{band}_sdss_ps_ztf_100.csv", index_col='uid')
    s['band'] = band
    vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
    s = s.join(vac, on='uid')
    skewfits.append(s)
skewfits = pd.concat(skewfits).dropna().sort_index()
skewfits = parse.filter_data(skewfits, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5)}, verbose=True)

skewfits.loc[444249]

sigs_ = [sigs['tau50'].values[mask] for mask in masks]

# +
vac = load_vac('qsos', usecols=['z','Lbol'])
skewfits = []
for band in 'ri':
    s = pd.read_csv(cfg.D_DIR + f'computed/qsos/mcmc_fits/skewfit_{band}.csv', index_col=ID)
    s['band'] = band
    vac['logz'] = np.log10(vac['z']+1)
    vac['wavelength'] = calculate_wavelength(band, vac['z'])
    s = s.join(vac, on=ID)
    skewfits.append(s)
skewfits = pd.concat(skewfits).dropna().sort_index()

Lbol_edges   = np.linspace(45.2, 47.2, 5)
lambda_edges = np.linspace(900, 4900, 5)

# create a series of 2d bins from the edges
Lbol_bins = pd.cut(skewfits['Lbol'], Lbol_edges, labels=False)
lambda_bins = pd.cut(skewfits['wavelength'], lambda_edges, labels=False)

# change groups so that the y axis is flipped
groups = {1: (0,3), 2: (0,2), 3: (1,3), 4: (1,2), 5: (1,1), 6: (2,3), 7: (2,2), 8: (2,1), 9: (2,0), 10: (3,2), 11: (3,1), 12: (3,0)}

masks = [(Lbol_bins == L).values & (lambda_bins == l).values for l,L in groups.values()]

fits = [skewfits[['a','loc','scale','z']].values[mask] for mask in masks]
# -

vac = load_vac('qsos')

vac

from scipy.stats import skewnorm
def logprob(x):
    models = [x[i+1] + x[0]*np.log10(1+fits[i][:,3]) for i in range(len(x)-1)]
    t_pdfs = [np.sum(skewnorm.logpdf(models[i], a=fits[i][:,0], loc=fits[i][:,1], scale=fits[i][:,2])) for i in range(len(models))]
    return sum(t_pdfs)


x = '9.999994165699942483e-01 2.862710572949632049e+00 2.769789894058614976e+00 3.037188195671315594e+00 3.148670397964941525e+00 3.122755495069454490e+00 2.712941034058935053e+00 2.885263947785130068e+00 2.982085534589460352e+00 3.337460146170504682e+00 2.752014812763630580e+00 2.622277248512973280e+00 2.671287604723956655e+00'.split(' ')
# x = '9.999990514355071580e-01 2.649495148614880335e+00 2.734138629633564221e+00 1.177457862193407756e+00 1.246828610947124272e+00 1.595753459400893970e+00 3.962651167128824703e+00 1.841946357505710452e-01 -2.412573152599222226e+00 3.172637873425212973e+00 1.239209703860985279e+00 2.641853156259578839e+00 2.848195526366154695e+00'.split(' ')
x = [float(a) for a in x]

logprob(x)

import pickle as pkl
file = open(cfg.W_DIR + 'temp/zfits.pkl','rb')
zfits = pkl.load(file)
file.close()
file = open(cfg.W_DIR + 'temp/tfits.pkl','rb')
tfits = pkl.load(file)
file.close()

pd.DataFrame(tfits[3]).hist()

pd.DataFrame(fits[3][:,:3]).hist()
