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

import pandas as pd
import numpy as np
from multiprocessing import Pool
import time


# ## Apply func to cores

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# ## Reading

# +
def reader(n_subarray, *args):
    band = ar
    return pd.read_csv('../lcs_merged/lc_{}_{}.csv'.format(band, n_subarray))

def read_csv_mp(n_cores = 4):
    pool = Pool(n_cores) # number of cores you want to use
    df_list = pool.map(reader, [1,2,3,4]) #creates a list of the loaded df's
    
    return pd.concat(df_list) # concatenates all the df's into a single df


# -

start = time.time()
band = 'r'
df1 = read_csv_mp(n_cores = 4)
end = time.time()
print(end - start)

start = time.time()
df2 = pd.read_csv('../lcs_merged/lc_r.csv')
end = time.time()
print(end - start)
