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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
path = '/disk1/hrb/python/'


class calib():
    def __init__(self, obj, ID, band):
        self.obj = obj
        self.band = band
        self.ID = ID
        
    def read(self):
        self.sdss = pd.read_csv(path+'data/merged/{}/{}_band/grouped_stats_{}_sdss.csv'.format(self.obj, self.band, self.band), index_col=self.ID)
        self.ps   = pd.read_csv(path+'data/merged/{}/{}_band/grouped_stats_{}_ps.csv'  .format(self.obj, self.band, self.band), index_col=self.ID)
        self.ztf  = pd.read_csv(path+'data/merged/{}/{}_band/grouped_stats_{}_ztf.csv' .format(self.obj, self.band, self.band), index_col=self.ID)
        
    def join_colors(self):
        self.colors = pd.read_csv(path+'data/computed/{}/colors_sdss.csv'.format(self.obj), index_col=self.ID)
        self.sdss = self.sdss.join(self.colors, on=self.ID, how='left')
        self.ps   = self.ps  .join(self.colors, on=self.ID, how='left')
        self.ztf  = self.ztf .join(self.colors, on=self.ID, how='left')
        
    def calculate_offsets(self):
        offsets = self.colors
        offsets['ps-sdss_nat'] = self.ps ['mag_med_native'] - self.sdss['mag_med_native']
#         offsets['sdss-ztf_nat']= self.sdss['mag_med_native']-self.ztf['mag_med_native']
        offsets['ps-ztf_nat']  = self.ps  ['mag_med_native']-self.ztf['mag_med_native']
        
        offsets['ps-sdss'] = self.ps ['mag_med'] - self.sdss['mag_med']
#         offsets['sdss-ztf']= self.sdss['mag_med']-self.ztf['mag_med']
        offsets['ps-ztf']  = self.ps  ['mag_med']-self.ztf['mag_med']
        
        offsets['mean_mag_sdss']  = self.sdss['mag_med']
        
        self.offsets = offsets
        
    def sns_correlate(self, xname, yname, vmin, vmax, color='blue', limits=None, colorscale=None, g=None, save=False, **kwargs):
        """
        Seaborn jointplot with marginal distributions

        Parameters
        ----------
        xname : str
        yname : str
        vmin  : int
        vmax  : int
        color : str
        limits: list of tuples
        colorscale : str
        g : axis handle
        save : bool
        kwargs : dict, passed to ax.set()

        Returns
        -------
        value : type
            desc
        """
        data = self.offsets[[xname,yname]]

        # Setting up limits
        if limits is None:
            lims = lambda x, n: (x.mean()-n*x.std(), x.mean()+n*x.std())
            xbounds = lims(data[xname], 3) # set the window to +/- 3 sigma from the mean
            ybounds = lims(data[yname], 2.5)
        else:
            xbounds, ybounds = limits[0], limits[1]
        bounds={xname:xbounds, yname:ybounds}

        # Remove data outside limits
        data = data[((xbounds[0] < data[xname])&(data[xname] < xbounds[1])) & ((ybounds[0] < data[yname])&(data[yname] < ybounds[1]))]

        # Setting up colors
        marg_color = {'blue':'royalblue', 'red':'salmon'}[color]
        joint_color = {'blue':'Blues', 'red':'Reds'}[color]

        if g is None:
            g = sns.JointGrid(x=xname, y=yname, data=data, xlim=bounds[xname], ylim=bounds[yname], height=8)

        # If colorscale is 'log', use log scale and vmin, vmax as limits. Otherwise, default
        if colorscale == 'log':
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=vmin, vmax=vmax)
            vmin=None
            vmax=None
        else:
            norm = None
        # Plot
        g = g.plot_joint(plt.hexbin, norm=norm, cmap='jet', vmin=vmin, vmax=vmax)#, gridsize=(200,200))
    #     g = sns.jointplot(x=xname, y=yname, data=data, xlim=bounds[xname], ylim=bounds[yname], height=6)
        g.ax_marg_x.hist(data[xname], bins=200, color=marg_color, alpha=0.5)
        g.ax_marg_y.hist(data[yname], bins=200, orientation='horizontal', density = True, color=marg_color, alpha=0.5)

        # Uncomment below to invert axis - needed when plotting absolute mag on y axis
    #     g.ax_joint.invert_yaxis()

        # Set axis labels and fontsize
        g.ax_joint.set_xlabel(kwargs['xlabel'], fontsize=25)
        g.ax_joint.set_ylabel(kwargs['ylabel'], fontsize=25)
        # Increase fontsize of ticks
        g.ax_joint.tick_params(axis='both', which='major', labelsize=20)
    #     g.ax_joint.set(**kwargs)

        # Uncomment below to add y=0 and x=0 dashed lines
        plt.axhline(y=0, lw=0.7, color='k', ls='--', dashes=(20,10))
    #     plt.axvline(x=0, lw=0.2, color='k', ls='--', dashes=(20,10))

        plt.grid(lw = 0.2, which='major')

        # Save
        if save:
            g.savefig(path+'analysis/plots/{}_{}_vs_{}.pdf'.format(self.obj,xname,yname), bbox_inches='tight')

        # Return axis handle
        return g


qsos_r = calib('qsos','uid','r')
qsos_r.read()
qsos_r.join_colors()
qsos_r.calculate_offsets()

# + active=""
# fig, ax = plt.subplots(1,1, figsize=(15,6))
# ax.hist((qsos_r.ps['mag_med']-qsos_r.ztf['mag_med'])  , bins=200, alpha=0.5, range=(-0.5,0.5), label='transformed');
# ax.hist((qsos_r.ps['mag_med_native']-qsos_r.ztf['mag_med_native'])  , bins=200, alpha=0.5, range=(-0.5,0.5), label='untransformed');
# # ax.hist((qsos_r.sdss['mag_mean']-qsos_r.ps['mag_mean']), bins=200, alpha=0.5, range=(-2,2), label='mean');
# ax.legend()
# ax.set(xlabel=r'$r_\mathrm{SDSS} - r_\mathrm{PS}$ (mag)', ylabel='Number')
# ax.axvline(x=0, lw=0.4, ls='--')
# -

# # STARS
# ---

star_r = calib('calibStars','uid_s','r')
star_r.read()
star_r.join_colors()
star_r.calculate_offsets()

star_r.offsets['ps-sdss_nat'].mean()

star_r.offsets['ps-sdss'].mean()

star_r.offsets['ps-ztf_nat'].mean()

star_r.offsets['ps-ztf'].mean()

# +
save = True
for color in ['mean_ri','mean_gr']:
    for survey1, survey2 in [('ps','sdss'),('ps','ztf')]:
#     color = 'mean_ri'
#     survey1 = 'ps'
#     survey2 = 'ztf'
        g = star_r.sns_correlate(color, survey1+'-'+survey2+'_nat', 1.5e1, 1e3, limits=[(0,2),(-0.2,0.4)], colorscale='log', xlabel=r'$'+color[-2]+'-'+color[-1]+'$', ylabel=r'$r_\mathrm{'+survey1.upper()+'}-r_\mathrm{'+survey2.upper()+'}$', save=save)
        g = star_r.sns_correlate(color, survey1+'-'+survey2       , 1.5e1, 1e3, limits=[(0,2),(-0.2,0.4)], colorscale='log', xlabel=r'$'+color[-2]+'-'+color[-1]+'$', ylabel=r'$r_\mathrm{'+survey1.upper()+'}-r_\mathrm{'+survey2.upper()+'}^\prime$', save=save)


# -

# # QSOS
# ---

qsos_r = calib('qsos','uid','r')
qsos_r.read()
qsos_r.join_colors()
qsos_r.calculate_offsets()

qsos_r.offsets['ps-sdss_nat'].mean()

qsos_r.offsets['ps-sdss'].mean()

qsos_r.offsets['ps-ztf_nat'].mean()

qsos_r.offsets['ps-ztf'].mean()

# +
save = True
for color in ['mean_ri','mean_gr']:
    for survey1, survey2 in [('ps','sdss'),('ps','ztf')]:
#     color = 'mean_ri'
#     survey1 = 'ps'
#     survey2 = 'ztf'
        g = qsos_r.sns_correlate(color, survey1+'-'+survey2+'_nat', 1.5e1, 1e3, limits=[(-0.5,1),(-1.25,1.25)], colorscale='log', xlabel=r'$'+color[-2]+'-'+color[-1]+'$', ylabel=r'$r_\mathrm{'+survey1.upper()+'}-r_\mathrm{'+survey2.upper()+'}$', save=save)
        g = qsos_r.sns_correlate(color, survey1+'-'+survey2       , 1.5e1, 1e3, limits=[(-0.5,1),(-1.25,1.25)], colorscale='log', xlabel=r'$'+color[-2]+'-'+color[-1]+'$', ylabel=r'$r_\mathrm{'+survey1.upper()+'}-r_\mathrm{'+survey2.upper()+'}^\prime$', save=save)


# -

color = 'mean_gr'
survey1 = 'sdss'
survey2 = 'ztf'
g = qsos_r.sns_correlate(color, survey1+'-'+survey2+'_nat', 1.5e1, 1e3, limits=[(-0.5,1),(-1.25,1.25)], colorscale='log', xlabel=r'$'+color[-2]+'-'+color[-1]+'$', ylabel=r'$r_\mathrm{'+survey1.upper()+'}-r_\mathrm{'+survey2.upper()+'}$', save=True)
g = qsos_r.sns_correlate(color, survey1+'-'+survey2       , 1.5e1, 1e3, limits=[(-0.5,1),(-1.25,1.25)], colorscale='log', xlabel=r'$'+color[-2]+'-'+color[-1]+'$', ylabel=r'$r_\mathrm{'+survey1.upper()+'}-r_\mathrm{'+survey2.upper()+'}^\prime$', save=True)


