import numpy as np
import matplotlib.pyplot as plt

qso = '70'
path = '/disk1/hrb/Python' + '/QSO_S82/' + qso

lcs = np.loadtxt(path)[:,:2]


lcs = np.delete(lcs,tuple(np.where(lcs[:,1] == -99.99)[0]),axis=0)

plt.scatter(lcs[:,0],lcs[:,1])


