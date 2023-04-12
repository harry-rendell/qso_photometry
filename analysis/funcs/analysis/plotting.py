import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_series(df, uids, axes=None, grouped=None, **kwargs):
	"""
	Simple plotting function for lightcurves
	"""
	if np.issubdtype(type(uids),np.integer): uids = [uids]
	if axes is None:
		fig, axes = plt.subplots(len(uids),1,figsize = (25,3*len(uids)), sharex=True)
	if len(uids)==1:
		axes=[axes]
	for uid, ax in zip(uids,axes):
		x = df.loc[uid]
		ax.errorbar(x=x['mjd'], y=x['mag'], yerr=x['magerr'], lw = 0.5, markersize = 3)
		ax.scatter(x['mjd'], x['mag'], s=10)
		ax.invert_yaxis()
		ax.set(xlabel='MJD', ylabel='mag', **kwargs)
		ax.text(0.02, 0.9, 'uid: {}'.format(uid), transform=ax.transAxes, fontsize=10)

		if grouped is not None:
			y = grouped.loc[uid]
			mask = ( abs(x['mag']-y['mag_med']) > 3*y['mag_std'] ).values
			ax.scatter(x[mask]['mjd'], x[mask]['mag'], color='r', s=30)

	plt.subplots_adjust(hspace=0)
	return axes