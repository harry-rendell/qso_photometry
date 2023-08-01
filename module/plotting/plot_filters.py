import matplotlib.pyplot as plt
import pandas as pd
from ..config import cfg

def plot_filters(**kwargs):
	"""
    Plots SDSS, PanSTARRS, ZTF filters
	
    Returns
    -------
	fig, ax, sdss, ps, ztf
	
    """
	wdir = cfg.D_DIR + 'surveys/filters/'
	sdss  = pd.read_csv(wdir + 'sdss.csv')
	ztf_g = pd.read_csv(wdir + 'raw/ztf/ztf_g.csv')
	ztf_r = pd.read_csv(wdir + 'raw/ztf/ztf_r.csv')
	ztf_i = pd.read_csv(wdir + 'raw/ztf/ztf_i.csv')
	ps    = pd.read_csv(wdir + 'ps.csv')


	# Colors
	colors = [cfg.FIG.COLOURS.BANDS[c] for c in 'gri']

	# Getting everything into the same units
	ztf_g.loc[:, 'g'] /= 100
	ztf_r.loc[:, 'r'] /= 100
	ztf_i.loc[:, 'i'] /= 100
	ztf_g.loc[:,'lambda'] *= 10
	ztf_r.loc[:,'lambda'] *= 10
	ztf_i.loc[:,'lambda'] *= 10
	ps.loc[:,'lambda'] *= 10

	fig, ax = plt.subplots(1,1, figsize=(10,5))
	ztf_g.plot(x='lambda', y='g', ax=ax, ls='-', ms=10, label='ztf_g', color = cfg.FIG.COLOURS.BANDS['g'])
	ztf_r.plot(x='lambda', y='r', ax=ax, ls='-', ms=10, label='ztf_r', color = cfg.FIG.COLOURS.BANDS['r'])
	ztf_i.plot(x='lambda', y='i', ax=ax, ls='-', ms=10, label='ztf_i', color = cfg.FIG.COLOURS.BANDS['i'])

	sdss.plot(x='lambda', y=['g','r','i'], ax=ax, ls='--' , ms=10, label=['sdss_'+b for b in 'gri'], color = colors)
	ps  .plot(x='lambda', y=['g','r','i'], ax=ax, ls='-.', ms=10, label=['ps_' + b for b in 'gri'], color = colors)
# 	sdss.plot(x='lambda', y=['g','r','i','z'], ax=ax, ls='-' , ms=10, label=['sdss_'+b for b in 'griz'], color = list('grbk'))
# 	ps  .plot(x='lambda', y=['g','r','i','z'], ax=ax, ls='-.', ms=10, label=['ps_' + b for b in 'griz'], color = list('grbk'))

	ax.set(xlabel='wavelength (Å)', ylabel='transmission', **kwargs);
	ax.get_legend().remove()
	ztf = [ztf_g, ztf_r, ztf_i]
	
	return fig, ax, sdss, ps, ztf