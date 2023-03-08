def plot_magerr_hist(n_bins=201, quantiles=[0.05,0.1,0.15,0.2,0.25], show_lines=False, savename=None):
	"""
	Plots distribution and cumulative distribution of magnitude errors for each survey

	Parameters
	----------
	n_bins : number of bins in histogram
	quantiles : quantiles to use for finding how much of the population is below a certain magerr cut-off
	show_lines : plots horizontal and vertical lines indicating location of quantiles
	savename : if specified, the plot will be saved with name 'savename.pdf'
 	
	Returns
	-------
	ax : axes handle
	"""
    fig, ax = plt.subplots(3,1, figsize=(18,18))
    upper_bound = 0.4
    print('| band | max magerr |  SDSS |  PS   | ZTF   |\n|------|------------|-------|-------|-------|')
    for i,b in enumerate('gri'):
        sdss_data = sdss.df['magerr_'+b]
        ps_data   = ps.df.loc[pd.IndexSlice[:,b],'magerr']
        ztf_data  = ztf.df.loc[pd.IndexSlice[:,b],'magerr']
        ax_twin = ax[i].twinx()
        ax_twin.set(ylim=[0,1], ylabel='cumulative fraction')

        n, bins, _ = ax[i].hist(sdss_data, bins=n_bins, alpha=0.4, label='sdss', range=(0,upper_bound), density=True, color='c')
        cum_frac_sdss = (bins[:-1]+(bins[1:]-bins[:-1]), np.cumsum(n)/n.sum())
        ax_twin.plot(cum_frac_sdss[0],cum_frac_sdss[1], color='c', ls='--')

        n, bins, _ = ax[i].hist(ps_data,   bins=n_bins, alpha=0.4, label='ps'  , range=(0,upper_bound), density=True, color='r')
        cum_frac_ps = (bins[:-1]+(bins[1:]-bins[:-1]), np.cumsum(n)/n.sum())
        ax_twin.plot(cum_frac_ps[0],cum_frac_ps[1], color='r', ls='--')

        n, bins, _ = ax[i].hist(ztf_data,  bins=n_bins, alpha=0.4, label='ztf' , range=(0,upper_bound), density=True, color='g')
        cum_frac_ztf = (bins[:-1]+(bins[1:]-bins[:-1]), np.cumsum(n)/n.sum())
        ax_twin.plot(cum_frac_ztf[0], cum_frac_ztf[1], color='g', ls='--')
            
        for x, ls in zip(quantiles,['--','-.','-',':','--']):
            idx = int(x*n_bins/upper_bound)
            if show_lines:
                ax_twin.axvline(x=x, color = 'k', lw=1, ls=ls)
                ax_twin.axhline(xmin=x*2.5, y=cum_frac_sdss[1][idx], color='c', lw=1.2, ls=ls, dashes=(5,10))
                ax_twin.axhline(xmin=x*2.5, y=cum_frac_ps  [1][idx], color='r', lw=1.2, ls=ls, dashes=(5,10))
                ax_twin.axhline(xmin=x*2.5, y=cum_frac_ztf [1][idx], color='g', lw=1.2, ls=ls, dashes=(5,10))
#             print(f'observations with mag {b} < {x:.2f}:')
            print(f'|  {b}   |    {x:.2f}    | {cum_frac_sdss[1][idx]*100:.1f}% | {cum_frac_ps  [1][idx]*100:.1f}% | {cum_frac_ztf [1][idx]*100:.1f}% |')

        ax[i].set(xlabel=f'{b} magnitude error', xlim=[0,upper_bound])
        ax[i].legend()
		
		if savename is not None:
			fig.savefig(savename + '.pdf', bbox_inches='tight')
    
	return ax