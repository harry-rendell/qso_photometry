import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import parse, color_transform
from module.assets import load_vac
import pandas as pd
from scipy.stats import skewnorm
from nautilus import Prior, Sampler

def likelihood(param_dict):
    x = [param_dict[p] for p in params]
    models = [x[i+1] + x[0]*np.log10(1+fits[i][:,3]) for i in range(len(x)-1)]
    t_pdfs = [np.sum(skewnorm.logpdf(models[i], a=fits[i][:,0], loc=fits[i][:,1], scale=fits[i][:,2])) for i in range(len(models))]
    return sum(t_pdfs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="One or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
    else:
        ID = 'uid_s'

    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    vac = load_vac('qsos', usecols=['z','Lbol'])
    skewfits = []
    for band in args.band:
        s = pd.read_csv(cfg.D_DIR + f'computed/qsos/mcmc_fits/skewfit_{band}.csv', index_col=ID)
        s['band'] = band
        vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
        s = s.join(vac, on=ID)
        skewfits.append(s)
    skewfits = pd.concat(skewfits).dropna().sort_index()
    skewfits = parse.filter_data(skewfits, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5)}, verbose=True)

    Lbol_edges   = np.linspace(45.2, 47.2, 5)
    lambda_edges = np.linspace(900, 4900, 5)
    # create a series of 2d bins from the edges
    Lbol_bins = pd.cut(skewfits['Lbol'], Lbol_edges, labels=False)
    lambda_bins = pd.cut(skewfits['wavelength'], lambda_edges, labels=False)

    # change groups so that the y axis is flipped
    groups = {1: (0,3), 2: (0,2), 3: (1,3), 4: (1,2), 5: (1,1), 6: (2,3), 7: (2,2), 8: (2,1), 9: (2,0), 10: (3,2), 11: (3,1), 12: (3,0)}

    masks = [(Lbol_bins == L).values & (lambda_bins == l).values for l,L in groups.values()]
    fits = [skewfits[['a','loc','scale','z']].values[mask] for mask in masks]

    start = time.time()
    os.makedirs(cfg.W_DIR + 'temp/nautilus_test_0.5_1.5/', exist_ok=True)
    os.chdir(cfg.W_DIR + 'temp/nautilus_test_0.5_1.5/')

    params = ['n'] + [f'c_{1+i}' for i in range(12)]
    prior = Prior()
    for key in params:
        if key == 'n':
            prior.add_parameter(key, dist=(0.5,1.5))
        else:
            prior.add_parameter(key, dist=(2,4))

    sampler = Sampler(prior, likelihood, n_dim=13, pool=args.n_cores, filepath='checkpoint.hdf5')
    sampler.run(verbose=True)
    # points, log_w, log_l, t = sampler.posterior(return_blobs=True)
    points, log_w, log_l = sampler.posterior()
    
    np.savez('posterior.npz', points=points, log_w=log_w, log_l=log_l)
    # results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/{args.model}_fits_{band}_{args.survey}_{args.nobs_min}_{phot_str}.csv')
    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)),'\n')