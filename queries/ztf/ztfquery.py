import pandas as pd
import numpy as np
import sys

def split_save(oid_batch,i):
    print('URL too long, splitting and saving')
    oid_batch_half = np.array_split(oid_batch,2)
    for j, oid_batch_half in enumerate(oid_batch):
        url = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{}&FORMAT=csv&BAD_CATFLAGS_MASK=32768'.format(''.join(['&ID='+str(oid) for oid in oid_batch_half])[1:])
        t2 = pd.read_csv(url, usecols = ['oid', 'mjd', 'mag','magerr','filtercode','magzp','clrcoeff','clrcounc'])
        t2.to_csv('dr2_clr/batch_{:02d}/lc_{}_{}_{}.csv'.format(n,n,i,j), index=False)

def obtain_ztf_lightcurves(n):
    oids = np.array_split(np.loadtxt('/disk1/hrb/python/data/surveys/ztf/meta_data/ztf_oids_ngoodobsrel_nonzero.txt',dtype=np.uint64),2**4)[n]
    for idx, oid_batch in enumerate(np.array_split(oids,2**8)):
        url = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{}&FORMAT=csv&BAD_CATFLAGS_MASK=32768'.format(''.join(['&ID='+str(oid) for oid in oid_batch])[1:])
        try:
            t2 = pd.read_csv(url, usecols = ['oid', 'mjd', 'mag','magerr','filtercode','magzp','clrcoeff','clrcounc'])
            t2.to_csv('dr2_clr/batch_{:02d}/lc_{}_{}.csv'.format(n,n,idx), index=False)
            print('saving',idx,'/',2**8)
        except:
            print('URL too long, splitting and saving')
            try:
                split_save(oid_batch, idx)
            except:
                print('Cannot save',idx)

    print('finished')
    
n = int(sys.argv[1])
print(n)
obtain_ztf_lightcurves(n)
