import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from module.config import cfg
import matplotlib.pyplot as plt
from module.preprocessing.binning import calculate_groups

def plot_groups(x, bounds, plot=False, hist_kwargs={}, ax_kwargs={}):
    """
    Plot distribution of quasar property from VAC, and show groups.
    """
    groups, bounds_values, label_range_val = calculate_groups(x, bounds)
    for i in range(len(bounds)-1):
        print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],len(groups[i])))
        # print('{:+.2f} < z < {:+.2f}: {:,}'.format(bounds[i],bounds[i+1],((bounds[i]<z_score)&(z_score<bounds[i+1])&(self.properties['mag_count']>2)).sum()))

    fig, ax = plt.subplots(1,1,figsize = (12,5))
    ax.hist(x, **hist_kwargs)
    for value, z in zip(bounds_values, bounds):
        ax.axvline(x=value, ymax=1, color = 'k', lw=0.5, ls='--')
        # ax.axvline(x=value, ymin=0.97, ymax=1, color = 'k', lw=0.5, ls='--') # If we prefer to have the numbers inside the plot, use two separate lines to make
        # a gap between text
        ax.text(x=value, y=1.01, s=r'${}\sigma$'.format(z), horizontalalignment='center', transform=ax.get_xaxis_transform(), fontsize='small')
    ax.set(xlim=[bounds_values[0],bounds_values[-1]], **ax_kwargs)

    return fig