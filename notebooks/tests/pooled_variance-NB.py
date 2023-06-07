# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np

a = np.random.normal(2,0.1, size=10000)

true_var = a.var()
true_var

b = a.reshape(20,500)
b_mean = b.mean(axis=-1)
b_var  = b.var(axis=-1)
n = np.full((20),500)
pooled_mean = np.average(b_mean, weights = n, axis=0)
pooled_var  = np.average(b_var,  weights = n, axis=0) + np.average((b_mean-pooled_mean)**2, weights = n, axis=0)

pooled_mean

pooled_var

c = a.reshape(200,50)
c_mean = c.mean(axis=-1)
c_var  = c.var(axis=-1)
n = np.full((200),50)
pooled_mean = np.average(c_mean, weights = n, axis=0)
pooled_var  = np.average(c_var,  weights = n, axis=0) + np.average((c_mean-pooled_mean)**2, weights = n, axis=0)

pooled_mean

pooled_var

# We see that the pooled variances are identical. Thus it does not matter if we group a into 20 groups of 500 or 200 groups of 50.

# to test the above in practice, the code below splits the 

# +
    def func(self, log_or_lin, save=False):
        
        n_points=20 # number of points to plot
        self.log_or_lin = log_or_lin
        if log_or_lin.startswith('log'):
            self.mjd_edges = np.logspace(0, 4.37247, n_points+1) # TODO add max t into argument
        elif log_or_lin.startswith('lin'):
            self.mjd_edges = np.linspace(0, 23576, n_points+1) # CHANGE TO cfg.MAX_DT_REST_FRAME
            
        self.mjd_centres = (self.mjd_edges[:-1] + self.mjd_edges[1:])/2
        
        if __name__ == '__main__':
            n_cores = 4
            p = Pool(n_cores)
            names = ['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'SF cwf c']
            pooled_results = {name:np.zeros(shape=(n_points, 2)) for name in names}
            pooled_results['n'] = np.zeros(shape=(n_points), dtype='uint64')
            results = {name:np.zeros(shape=(n_cores, 2)) for name in names}
            
            multi_proc_list = p.map(self.calculate_stats_looped_single_core, np.array_split(np.arange(52),4));
            
#             return multi_proc_list
            
        for key in names:
            results[key] = np.concatenate([a[key] for a in multi_proc_list])
            
        for key in names:
            if key != 'n':
                pooled_mean = np.average(results[key][:,:,0], weights=results['n'], axis=0)
                pooled_var  = np.average(results[key][:,:,1], weights=results['n'], axis=0) + np.average((results[key][:,:,0]-pooled_mean)**2, weights=results['n'], axis=0)

                pooled_results[key][:,0] = pooled_mean
                pooled_results[key][:,1] = pooled_var
            else:
                pooled_results[key] = results[key].sum(axis=0)
        
        self.pooled_stats = pooled_results
        
    def calculate_stats_looped_single_core(self, n_chunks_arr):
        """
        Loop over dtdm files and calculate stats of each file. Append to dictionary.

        Parameters
        ----------
        n_chunks : int
            how many files to read in of files to read in.
            maximum value: stars = 200/4 = 50, qsos = 52/4 = 13
            
        log_or_lin : str
        
        save : bool
        
        Returns
        -------
        results : dict of nd_arrays, shape (n_chunk, n_points)
        """
        log_or_lin = 'log'
        n_points=20 # number of points to plot
        if log_or_lin.startswith('log'):
            self.mjd_edges = np.logspace(0, 4.37247, n_points+1) # TODO add max t into argument
        elif log_or_lin.startswith('lin'):
            self.mjd_edges = np.linspace(0, 23576, n_points+1)
            
        self.mjd_centres = (self.mjd_edges[:-1] + self.mjd_edges[1:])/2

        #hardcoding
        n_chunks = 13

#         names = ['n','SF 1', 'SF 2', 'SF 3', 'SF 4', 'SF weighted', 'SF corrected', 'SF corrected weighted', 'SF corrected weighted fixed', 'SF corrected weighted fixed 2', 'mean', 'mean weighted']
        names = ['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'SF cwf c']
        results = {name:np.zeros(shape=(n_chunks, n_points, 2)) for name in names} # 12/4 = 3, max is 52/4 = 13
        results['n'] = np.zeros(shape=(n_chunks, n_points), dtype='uint64')
           
        for j in n_chunks_arr:
            i = j % n_chunks
#             self.read(i)
            self.df = self.read_dtdm(self.fnames[j])
            print('chunk: {}'.format(j))
            for j, edges in enumerate(zip(self.mjd_edges[:-1], self.mjd_edges[1:])):
                mjd_lower, mjd_upper = edges
                boolean = (mjd_lower < self.df['dt']) & (self.df['dt']<mjd_upper)# & (self.df['dm2_de2']>0) # include last condition to remove negative SF values
#                 print('number of points in {:.1f} < ∆t < {:.1f}: {}'.format(mjd_lower, mjd_upper, boolean.sum()))
                subset = self.df[boolean]
                subset.loc[(subset['dm2_de2']<0).values,'dm2_de2'] = 0 # Include for setting negative SF values to zero. Need .values for mask to prevent pandas warning
                n = len(subset)
                results['n'][i,j] = n
                if n>0:
#                     results['mean'][i,j, (0,1)] = subset['dm'].mean(), subset['dm'].std()

#                     results['SF 1'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['de']**2).sum()/n
#                     results['SF 2'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['dm']**2).var()
#                     results['SF 3'][i,j,(0,1)] = (subset['dm']**2).mean(), 1/(subset['de']**-2).sum()

#                     results['SF 4'][i,j,(0,1)] = (subset['dm']**2).mean(), (2*subset['de']**4).sum()/n

#                     results['SF weighted'][i,j,(0,1)] = ( ((subset['dm']/subset['de'])**2) ).sum()/( (subset['de']**-2).sum() ), 1/(subset['de']**-2).sum()
#                     results['SF corrected'][i,j,(0,1)] = subset['dm2_de2'].mean(), subset['dm2_de2'].var()
#                     results['mean'][i,j,(0,1)] = subset['dm'].mean(), subset['dm'].std()
                    weights = subset['de']**-2
                    results['mean weighted a'][i,j,(0,1)] = np.average(subset['dm'], weights = weights), 1/weights.sum()
                    results['mean weighted b'][i,j,(0,1)] = np.average(subset['dm'], weights = weights), subset['dm'].var()

#                         results['SF corrected weighted'][i,j, group_idx, (0,1)] = ( subset['dm2_de2']/(subset['de']**2) ).sum()/( (subset['de']**-2).sum() ), 1/(subset['de']**-2).sum()
                    weights = 0.5*subset['de']**-4
                    results['SF cwf a'][i,j,(0,1)] = np.average(subset['dm2_de2'], weights = weights), 1/weights.sum()
                    results['SF cwf b'][i,j,(0,1)] = np.average(subset['dm2_de2'], weights = weights), subset['dm2_de2'].var()
                    results['SF cwf c'][i,j,(0,1)] = np.average(subset['dm2_de2'], weights = weights), (2*subset['de']**4).sum()/(n**2)

                else:
                    print('number of points in bin:', n)
        return results

