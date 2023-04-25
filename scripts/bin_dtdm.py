import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from module.analysis import analysis
from os import listdir
import os
import time
from binning import bin_data

wdir = '/disk1/hrb/python/'
band = 'r'

config = {'obj':'qsos','ID':'uid'  ,'t_max':6751,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':248, 'n_t_chunk':19, 'width':2, 'steepness':0.005, 'leftmost_bin':-0.244}
# config = {'obj':'calibStars','ID':'uid_s','t_max':7772,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':235,'n_t_chunk':19, 'width':1, 'steepness':0.005, 'leftmost_bin':-0.21}

width   = config['width']
steepness = config['steepness']
obj = config['obj']
ID  = config['ID']
t_max = config['t_max']
n_bins_t = config['n_bins_t']
n_bins_m = config['n_bins_m']
n_bins_m2 = config['n_bins_m2']
leftmost_bin = config['leftmost_bin']

data_path = wdir+'data/computed/{}/dtdm/raw/{}/'.format(obj,band)

# sort based on filesize, then do ordered shuffle so that each core recieves the same number of large files
fnames = [a for a in listdir(data_path) if (len(a)>27)]
size=[]
for file in fnames:
    size.append(os.path.getsize(data_path+file))
    
fnames = [name for i in [0,1,2,3] for sizename, name in sorted(zip(size, fnames))[i::4]]

###################################

dr = analysis(ID, obj)
dr.properties = pd.read_csv(wdir+'data/catalogues/qsos/dr12q/SDSS_DR12Q_BH_matched.csv', index_col=dr.ID)
key = 'Lbol'
prop_range_all = {'Mi':(-30,-20),'mag_mean':(15,23.5),'mag_std':(0,1),'redshift':(0,5),'Lbol':(44,48),'nEdd':(-3,0.5), 'MBH_CIV':(6,12)}
prop_range_any = {key:prop_range_all[key]}
# mask_all = np.array([(bound[0] < dr.properties[key]) & (dr.properties[key] < bound[1]) for key, bound in prop_range_all.items()])
mask_any  = np.array([(bound[0] < dr.properties[key]) & (dr.properties[key] < bound[1]) for key, bound in prop_range_any.items()])
mask = mask_any.any(axis=0)
dr.properties = dr.properties[mask]
bounds_z = np.array([-3.5,-1.5,-1,-0.5,0,0.5,1,1.5,3.5])
bounds_tuple, z_score_val, bounds_values, mean, std, ax = dr.bounds(key, bounds = bounds_z)
z_score = z_score_val['z_score']

def bin_chunks(indicies):
	dts_binned_tot = np.zeros((n_t_chunk,n_bins_t), dtype = 'int64')
	dms_binned_tot = np.zeros((n_t_chunk,n_bins_m), dtype = 'int64')
	des_binned_tot = np.zeros((n_t_chunk,n_bins_m), dtype = 'int64')
	dm2_de2_binned_tot = np.zeros((n_t_chunk,n_bins_m2), dtype = 'int64')
	
	dcat_tot = np.zeros((n_t_chunk,122), dtype = 'int64')
	boolean = lambda x: np.any([(x == cat) for cat in cat_list], axis=0)
	def boolean(df):
		if cat_list==0:
			# return whole dataframe if cat_list=0 (ie all)
			return df.values
		else:
			# return subset of dataframe for given survey pair
			return df[np.any([(df['cat'] == cat) for cat in cat_list], axis=0)].values
	for i in indicies:
		print('core {}: {}/{}'.format(int((i-lower) // (( upper-lower) / n_cores))+1, i-indicies[0], indicies[-1]-indicies[0] ))
#		 print(i)
		fpath = data_path+fnames[i]
#		 print('reading: {}\n'.format(fnames[i]))
		df = pd.read_csv(fpath, index_col = ID, dtype = {ID: np.uint32, 'dt': np.float32, 'dm': np.float32, 'de': np.float32, 'dm2_de2': np.float32, 'cat': np.uint8});
		uid_uniq = df.index.unique()
		sub_uids = uid_uniq[uid_uniq.isin(uids)]
        ############################
		df = df.loc[sub_uids]
        ##########################
        
		dtdms = boolean(df)
		dms_binned, dts_binned, des_binned, dcat = bin_data(dtdms, n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, compute=True, steepness=0.005, width=2, leftmost_bin=leftmost_bin)
		dts_binned_tot += dts_binned.astype('int64')
		dms_binned_tot += dms_binned.astype('int64')
		dm2_de2_binned_tot += dm2_de2_binned.astype('int64')
		des_binned_tot += des_binned.astype('int64')
		dcat_tot	   += dcat.astype('int64')

        ########


	return dts_binned_tot, dms_binned_tot, dm2_de2_binned_tot, des_binned_tot, dcat_tot

##########################################

lower   = 0
upper   = 4
n_cores = 4
# 200 files for stars, 36 for qsos

t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, m_bin_widths, e_bin_edges, t_dict = bin_data(None, n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, steepness=steepness, width=width, compute=False, leftmost_bin=leftmost_bin)
indicies = np.array_split(np.arange(lower,upper), n_cores)

# cat_list_dict = {'sdss_sdss':[4],'sdss_ps':[5,7],'sdss_ztf':[6,10],'ps_ps':[8],'ps_ztf':[9,11],'ztf_ztf':[12]}#, 'all':0}
cat_list_dict = {'all':0}


##############################################

for name, cat_list in cat_list_dict.items():
    
    groups = [z_score[(lower < z_score) & (z_score < upper)] for lower, upper in bounds_tuple]
    start = time.time()
    
    for i, subgroup in enumerate(groups):
        uids = subgroup.index
        
        if __name__ == '__main__':
            p = Pool(n_cores)
            result = p.map(bin_chunks, indicies);
        #     result = result.sum(axis=0)
        # print(np.array_s plit(np.arange(lower,upper),4))
        # bin_chunks(np.array_split(np.arange(lower,upper),4)[3]) 

        a, b, c, d = result # unpack each subresult from multiproc
        dtdmde_result = (np.array(a[:-1]) + np.array(b[:-1]) + np.array(c[:-1]) + np.array(d[:-1]))
        dcat_result  = a[-1] + b[-1] + c[-1] + d[-1]
        dts_binned_tot, dms_binned_tot, des_binned_tot = dtdmde_result
        np.savetxt('../data/computed/{}/binned/{}/dc/dcs_binned_{}_{}_{}_{}.csv'.format(obj, key, obj, name, key, i), dcat_result, fmt='%i', delimiter=',')
        np.savetxt('../data/computed/{}/binned/{}/dt/dts_binned_{}_{}_{}_{}.csv'.format(obj, key, obj, name, key, i), dts_binned_tot, fmt='%i', delimiter=',')
        np.savetxt('../data/computed/{}/binned/{}/dm/dms_binned_{}_{}_{}_{}.csv'.format(obj, key, obj, name, key, i), dms_binned_tot, fmt='%i', delimiter=',')
        np.savetxt('../data/computed/{}/binned/{}/de/des_binned_{}_{}_{}_{}.csv'.format(obj, key, obj, name, key, i), des_binned_tot, fmt='%i', delimiter=',')
        
    end = time.time()
    print('time taken: {:.2f} minutes'.format((end-start)/60.0))