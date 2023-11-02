import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from scipy.stats import skewnorm, norm
from nautilus import Prior, Sampler


def likelihood(param_dict):
    x = [param_dict[p] for p in params]
    models = [x[i+1] + x[0]*np.log10(1+zfits[i]) for i in range(len(x)-1)]
    t_pdfs = [np.sum(skewnorm.logpdf(models[i], a=tfits[i][:,0], loc=tfits[i][:,1], scale=tfits[i][:,2])) for i in range(len(models))]
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

    import pickle as pkl
    file = open(cfg.W_DIR + 'temp/zfits.pkl','rb')
    zfits = pkl.load(file)
    file.close()
    file = open(cfg.W_DIR + 'temp/tfits.pkl','rb')
    tfits = pkl.load(file)
    file.close()

    # Cannot parallelise this? :(
    start = time.time()
    os.makedirs(cfg.W_DIR + 'temp/nautilus_test_brewerdata/', exist_ok=True)
    os.chdir(cfg.W_DIR + 'temp/nautilus_test_brewerdata/')

    params = ['n'] + [f'c_{1+i}' for i in range(12)]
    prior = Prior()
    for key in params:
        if key == 'n':
            prior.add_parameter(key, dist=norm(loc=1.0, scale=0.5))
        else:
            prior.add_parameter(key, dist=(0,5))

    sampler = Sampler(prior,
                      likelihood,
                      n_live=3000,
                      n_dim=13,
                      pool=args.n_cores,
                      filepath='checkpoint.hdf5',
                      resume=False)
    sampler.run(verbose=True)
    points, log_w, log_l = sampler.posterior()
    
    np.savez('posterior.npz', points=points, log_w=log_w, log_l=log_l)
    # results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/{args.model}_fits_{band}_{args.survey}_{args.nobs_min}_{phot_str}.csv')
    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)),'\n')