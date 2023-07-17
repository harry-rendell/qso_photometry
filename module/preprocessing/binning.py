import numpy as np
from time import time
def create_bins(bin_dict):
    n_bins_t = bin_dict['n_bins_t']
    n_bins_T = bin_dict['n_bins_T']
    n_bins_m = bin_dict['n_bins_m']
    n_bins_m2 = bin_dict['n_bins_m2']
    t_max = bin_dict['t_max']
    steepness = bin_dict['steepness']
    width = bin_dict['width']
    leftmost_bin = bin_dict['leftmost_bin']

    def calc_m_edges(n_bins_m_, steepness, width, leftmost_bin=None):
        """
        Produce a set of bins with max extent 'width'.
        Note that by keeping the steepness the same, the bin spacing is linear with width.
        if leftmost_bin is not None, then first bin will begin from leftmost_bin. This is used for dm2_de2 which has an asymmetrical distribution about zero. 
            qsos:  bins=248 for dm2_de2, leftmost_bin = -0.244
            stars: bins=235 for dm2_de2, leftmost_bin = -0.21
        """
        steepness = steepness*width
        start = np.log10(steepness)
        stop = np.log10(steepness+width)
        bin_edges = np.concatenate((-np.logspace(start,stop,int(n_bins_m_/2+1))[:0:-1]+steepness,np.logspace(start,stop,int(n_bins_m_/2+1))-steepness))
        if leftmost_bin is not None:
            bin_edges = np.logspace(start,stop,int(n_bins_m_+1))-steepness
        # bin_centres = np.sign(bin_edges[:-1])*(bin_edges[1:]*bin_edges[:-1]) ** 0.5 # geometric mean. Note, difference between geometric and arithmetic mean is only 0.0006 mag at most (at the edges) so it is not necessary.s
        return bin_edges

    def calc_t_bins(n_bins_T, t_max, t_steepness):
        start = np.log10(t_steepness)
        stop = np.log10(t_steepness+t_max)
        return np.logspace(start, stop, n_bins_T+1)-t_steepness

    #compute edges of bins
    m_bin_edges = calc_m_edges(n_bins_m, steepness, width)
    m2_bin_edges = calc_m_edges(n_bins_m2, steepness, width, leftmost_bin)
    T_bin_edges = calc_t_bins(n_bins_T, t_max, t_steepness = 1000)
    t_bin_edges = np.linspace(0,t_max,(n_bins_t+1))
    e_bin_edges = np.linspace(0,0.75,201) # This assumes that the max error pair is sqrt(0.5**2+0.5**2)

    # t denotes smaller time bins, T denotes larger time bins
    T_dict = dict(enumerate(['{0:1.0f}<t<{1:1.0f}'.format(T_bin_edges[i],T_bin_edges[i+1]) for i in range(len(T_bin_edges)-1)]))

    # only bin_edges need to be returned since:
    # n_bins = len(edges)-1
    # width = edges[1:] - edges[:-1]
    # centres = (edges[1:] + edges[:-1])/2

    bin_dict = {'t_bin_edges':t_bin_edges,
                'T_bin_edges':T_bin_edges,
                'm_bin_edges':m_bin_edges,
                'm2_bin_edges':m2_bin_edges,
                'e_bin_edges':e_bin_edges,
                'T_dict':T_dict}
    
    return bin_dict

def bin_data(dtdm, kwargs):
    args = kwargs['bin_dict']
    dts_binned = np.zeros((args['n_bins_T'],args['n_bins_t']), dtype = 'int64')
    dms_binned = np.zeros((args['n_bins_T'],args['n_bins_m']), dtype = 'int64')
    des_binned = np.zeros((args['n_bins_T'],args['n_bins_m']), dtype = 'int64')
    dm2_de2_binned = np.zeros((args['n_bins_T'],args['n_bins_m2']), dtype = 'int64') # This should stay as n_bins_m since we cut down the full array of length n_bins_m2
    dsid           = np.zeros((args['n_bins_T'],122), dtype = 'int64')

    # First decide which larger bin chunk the data should be in
    # start = time()
    idxs = np.digitize(dtdm[:,0], args['T_bin_edges'])-1
    for i in np.unique(idxs): #Can we vectorize this?
        mask = (idxs == i)
        dts_binned[i]     = np.histogram(dtdm[mask,0], args['t_bin_edges'])[0]
        dms_binned[i]     = np.histogram(dtdm[mask,1], args['m_bin_edges'])[0]
        des_binned[i]     = np.histogram(dtdm[mask,2], args['e_bin_edges'])[0]
        dm2_de2_binned[i] = np.histogram(dtdm[mask,3], args['m2_bin_edges'])[0]
        a                 = np.bincount( dtdm[mask,4].astype('int64'))[4:]
        dsid[i]           = np.pad(a, (0,122-len(a)), 'constant', constant_values=0)

    del dtdm
    
    return dts_binned, dms_binned, des_binned, dm2_de2_binned, dsid

def create_binned_data_from_dtdm(df, kwargs):
    """
    TODO: This could be put in the function above
    """
    if not (('basepath' in kwargs) and ('fname' in kwargs)):
        raise Exception('Both basepath and fname must be provided')
    print(f"processing file: {kwargs['fname']}", flush=True)
    # If a subset of objects has been provided, restrict our DataFrame to that subset.
    if 'subset' in kwargs:
        df = df[df.index.isin(kwargs['subset'])]
    if 'inner' in kwargs:
        if kwargs['inner']:
            df = df[(np.sqrt(df['dsid'])%1==0).values]

    df['dm2_de2'] = df['dm']**2 - df['de']**2

    return bin_data(df[['dt','dm','de','dm2_de2','dsid']].values, kwargs)

def calculate_groups(x, bounds):
    """
    Compute z score of key for each object

    Parameters
    ----------
    key : string
            property from VAC

    Returns
    -------
    bounds : array_like
            array of bounds to be used
    z_score : pandas.DataFrame
            z value of property column (value-mean / std) and original value
    self.bounds_values : values of property for each value in bounds
    """
    bounds_tuple = list(zip(bounds[:-1],bounds[1:]))
    mean = x.mean()
    std  = x.std()
    z_score = (x-mean)/std
    bounds_values = bounds * std + mean
    groups = [z_score[(lower <= z_score).values & (z_score < upper).values].index.values for lower, upper in bounds_tuple]
    # label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(bounds_values[i],key,bounds_values[i+1]) for i in range(len(bounds_values)-1)}
    return groups, bounds_values
