import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''Establish path and read in data'''
path = '/disk1/hrb/Python/dr7q.txt'
df = pd.read_csv(path, sep = '\s+', comment='#')

'''Some PSF entries are zero - need to remove these?'''
                 
bands = ['u','g','r','i','z']
thresholds = [20,20,20,20,19]
#n_bins = np.array([[11,12,16,16,16],[16,16,15,13,15]])

def d_m(band,df):
    delta_m = df[band + '2_PSF'].values - df[band + '1_PSF'].values
    max_m = np.array([max(df[band + '2_PSF'].values[i],df[band + '1_PSF'].values[i]) for i in range(len(df))])
    err     = ( df[band + '2_err'].values**2 + df[band + '1_err'].values**2 ) ** 0.5
    delta_t      = df['MJD2'].values - df['MJD1'].values
    
    delta_m = delta_m * np.sign(delta_t)
    delta_t = delta_t * np.sign(delta_t)
    
    '''sort by delta_m'''
    unsorted_arr = np.array([delta_t,delta_m,err,max_m])
    sorted_arr   = unsorted_arr[:,np.argsort(unsorted_arr[1,:])]
    
    return sorted_arr

'''Choose band'''
#q = d_m('u',df) # delta_t, delta_mag. error, max mag

'''Limit to 300 < dt < 400 and mag < 20'''
#q = q[:,(q[0] < 400) & (q[0] > 300) & (q[-1] < 20)]

test1 = d_m(bands[0],df)

def bin_arr(bin_edges,q,counts):
    err_arr = q[2]
    mag = q[1]
    n = len(bin_edges)
    N = len(err_arr)
    err_sumsq = np.zeros(n-1)
    j = 0
    for i in range(N):
        if (mag[i] > bin_edges[j+1]):
            j += 1
        err_sumsq[j] += err_arr[i]**2
    
    err_sumsq = np.divide(err_sumsq,counts,out = np.zeros_like(err_sumsq),where = counts != 0)
    
    return (err_sumsq ** 0.5)

def plot_hist(ax,array,n_bins,label,ylabel=None):
        
        dist = array[1]    
    
        counts,bin_edges = np.histogram(dist,n_bins,range = (-1.25,1.25))
        
        err = bin_arr(bin_edges,array,counts)
        
        width = 2.5/n_bins
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        ntot = counts.sum()
        ax.errorbar(bin_centres, counts/(width*ntot),yerr = err, fmt='+', ms = 10, label = label)  
        ax.set(xlim = [-1.25,1.25], yscale = 'log', ylim = [0.5e-3,1e1], ylabel = ylabel, xlabel = r'$\Delta m$')
        ax.set_yticks([0.001, 0.01, 0.1, 1, 10], minor = True)
        ax.legend()
        ax.axvline(x=0,lw = 0.7, color = 'k')

        
        return err

    
fig, axes = plt.subplots(5,2,sharex = True, sharey = True, figsize = (9,15))
plt.subplots_adjust(wspace = 0, hspace = 0)

for i in range(5):
    
    q = d_m(bands[i],df) # delta_t, delta_mag. error, max mag
    q1 = q[:,(q[0] < 400) & (q[0] > 300) & (q[-1] < thresholds[i]) & (abs(q[1]) < 1.25)]
    q2 = q[:,(q[0] < 3e3) & (q[0] > 1.4e3) & (q[-1] < thresholds[i]) & (abs(q[1]) < 1.25)]
    
    err1 = plot_hist(axes[i,0],q1,16,label = bands[i] + ' < ' + str(thresholds[i]),ylabel = r'$n/N_{tot}$')
    err2 = plot_hist(axes[i,1],q2,16,label = bands[i] + ' < ' + str(thresholds[i]))
    
