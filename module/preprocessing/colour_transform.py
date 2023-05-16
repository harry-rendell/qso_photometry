import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from .import data_io
from ..config import cfg

def transform_ztf_to_ps(df, obj, band):
    """
    Add a column onto df with magnitudes transformed to the PS system
    """
    ID = df.index.name
    colors = pd.read_csv(cfg.USER.D_DIR + 'computed/{}/colors_sdss.csv'.format(obj), usecols=[ID,'g-r','r-i']).set_index(ID)
    df = df.join(colors, how='inner', on=ID).rename({'mag':'mag_orig'}, axis=1) #merge colors onto ztf df
    if (band == 'r') | (band == 'g'):
        df['mag'] = (df['mag_orig'] + df['clrcoeff']*df['g-r']).astype(cfg.COLLECTION.ZTF.dtypes.mag)
    elif band == 'i':
        df['mag'] = (df['mag_orig'] + df['clrcoeff']*df['r-i']).astype(cfg.COLLECTION.ZTF.dtypes.mag)
    else:
        raise Exception('Unrecognised band: '+band)
    return df[['mjd', 'mag', 'mag_orig', 'magerr']].dropna(subset=['mag']) # There are some NaN entries in colors_sdss.csv

def transform_sdss_to_ps(df, color='g-r', system='tonry'):
    """
    Add a column onto df with magnitudes transformed to the PS system.
    There are few options of published transformations available. Here we use ones from Tonry 2012.
    TODO: Move transformations to data/assets (unversioned).
    """
    color_transf = pd.read_csv(cfg.USER.W_DIR+'pipeline/transformations/transf_to_ps_{}.txt'.format(system), sep='\s+', index_col=0)
    df = df.rename({'mag':'mag_orig'}, axis=1)
    df['mag'] = 0
    for band in 'griz':
        a0, a1, a2, a3 = color_transf.loc[band].values
        # Convert to PS mags
        slidx = pd.IndexSlice[:, band]
        x = df.loc[slidx, color]
        df.loc[slidx, 'mag'] = df.loc[slidx, 'mag_orig'] + a0 + a1*x + a2*(x**2) + a3*(x**3)
    return df[['mjd', 'mag', 'mag_orig', 'magerr']].dropna(subset=['mag'])


#------------------------------------------------------------------------------
# Class for calculating and plotting transformations for SuperCOSMOS
#------------------------------------------------------------------------------

class ssa_transform():
    def __init__(self, obj, band, band_ssa, ref_survey_name):
        self.obj = obj
        self.ID = 'uid' if obj == 'qsos' else 'uid_s'
        self.band = band
        self.band_ssa = band_ssa
        self.qso_color_range = {'g-r':(-0.09786, 0.63699), 'r-i':(-0.11934, 0.44450), 'i-z':(-0.19706, 0.57335)}
        self.ref_survey_name = ref_survey_name
        
    def read(self):
        df = pd.read_csv(cfg.USER.D_DIR + 'surveys/supercosmos/{}/ssa_secondary.csv'.format(self.obj)).set_index(self.ID)
        self.ssa_secondary = df
        surveyID_dict = {'r1':2, 'r2':9}
        df = df[df['surveyID']==surveyID_dict[self.band_ssa]].rename(columns={'smag':'mag_orig'})
        kwargs = {'dtypes': cfg.PREPROC.stats_dtypes,
                  'ID':self.ID,
                  'basepath': cfg.USER.D_DIR + 'surveys/{}/{}/clean/{}_band/'.format(self.ref_survey_name, self.obj, self.band)}
        ref_survey_grouped = data_io.reader('grouped.csv', kwargs)
        colors = pd.read_csv(cfg.USER.D_DIR + 'computed/{}/colors_sdss.csv'.format(self.obj)).set_index(self.ID)
        ref_survey_grouped = ref_survey_grouped.join(colors, how='inner')

        key = 'mag_med' # can change this to mag_mean etc
        self.df = df.join(ref_survey_grouped[[key]+['g-r','r-i','i-z']], how='inner')
        
    def color_transform(self, color_name, poly_deg=None, p=None):
        """
        If p is None, poly_deg (degree of polynomial fit) should be specified
        Otherwise, polydeg will be taken from the length of p
        """
        color, mag_ssa, mag_ref = self.df[[color_name, 'mag_orig', 'mag_med']].values.T
        self.offset = mag_ssa - mag_ref
        self.mask = ((self.qso_color_range[color_name][0] < color) & (color < self.qso_color_range[color_name][1]))
        
        if p is None:
            p, res, _, _, _ = np.polyfit(color[mask].flatten(), offset[mask].flatten(), deg=poly_deg, full=True)
        else:
            poly_deg = len(p)-1
        self.mag_ssa = mag_ssa
        self.mag_ssa_transf = mag_ssa - np.array([p[poly_deg-i]*color**i for i in range(poly_deg+1)]).sum(axis=0)
        self.offset_transf = self.mag_ssa_transf - mag_ref
        self.p = p
        self.poly_deg = poly_deg
    #     # Snippet below uses scipy so we can quantify the error of the linear fit.
    #     m, c, r, p, std_err = linregress(color, offset)
    #     print('slope = {:.6f}, intercept = {:.6f}, r^2 = {:.4f}, err = {:.6f}'.format(m,c,r**2,std_err))
    #     mag_ssa_transf = mag_ssa - m*color - c
        
    def hist_1d(self):
        fig, ax = plt.subplots(1,1, figsize=(16,5))
        n, bins, _ = ax.hist(self.offset, bins=201, range=(-3,3), alpha=0.4, label='untransformed');
        n, bins, _ = ax.hist(self.offset_transf, bins=201, range=(-3,3), alpha=0.4, label='own');
        mode = (bins[n.argmax()]+bins[n.argmax()+1])/2

        ax.text(0.02,0.8, 'Post transformation:\nMean = {:.4f}\nStd = {:.4f}\nPeak = {:.4f}'.format(self.offset_transf[self.mask].mean(), self.offset_transf[self.mask].std(), mode), transform=ax.transAxes)
        ax.set(xlabel='{} - {}_{}'.format(self.band_ssa, self.band, self.ref_survey_name))
        ax.axvline(x=0, color='k', ls='--', lw=0.8)
        ax.legend()
        return ax
    
    def hist_2d(self, color_name):
        """
        Plot 2d histogram using color_name as x axis, comparing against previously calculated transformations
        """
        fig, axes = plt.subplots(1,2, figsize=(16,7))
        axes[0].hist2d(self.df[color_name], self.offset,        range=[self.qso_color_range[color_name],[-2,2]], bins=100, cmap=cmap.get_cmap('jet'));
        axes[1].hist2d(self.df[color_name], self.offset_transf, range=[self.qso_color_range[color_name],[-2,2]], bins=100, cmap=cmap.get_cmap('jet'));
        x = np.linspace(0,2,30)
        axes[0].plot(x, sum([self.p[self.poly_deg - i]*x**i for i in range(self.poly_deg+1)]), lw=3, ls='--', color='r', label='Linear fit')
        axes[0].axhline(y=0, lw=2, ls='--', color='k')
        axes[1].axhline(y=0, lw=2, ls='--', color='k')
        axes[0].set(ylabel=r'$r_\mathrm{PS}-r_\mathrm{SSS}$', xlabel=r'${}$'.format(color_name))
        axes[1].set(ylabel=r'$r_\mathrm{PS}-r_\mathrm{SSS}^\prime$', xlabel=r'${}$'.format(color_name))
        fig.subplots_adjust(wspace=0.3)
        return axes
    
    def mag_correlation(self):
        fig, ax = plt.subplots(1,2, figsize=(17,8))
        ax[0].hist2d(self.mag_ssa, self.df['mag_med'], bins=[100,100], range=[[16,22],[16,22]], cmap=cmap.get_cmap('jet'));
        ax[1].hist2d(self.mag_ssa_transf, self.df['mag_med'], bins=[100,100], range=[[16,22],[16,22]], cmap=cmap.get_cmap('jet'));
        x = np.linspace(15,22,10)
        ax[0].plot(x,x)
        ax[0].set(xlabel=self.band_ssa, ylabel='r mag (ps)')
        ax[1].plot(x,x)
        ax[1].set(xlabel=self.band_ssa + ' transformed', ylabel='r mag (ps)')
