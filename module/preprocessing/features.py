import numpy as np
import pandas as pd
from ..config import cfg

def welch_stetson_j(mag, mag_err, mag_mean, n):
    """
    Welch-Stetson variability index J (Stetson 1996)
    
    Might be better to make this a function of ∆m, ∆t

    mag : array of magnitudes
    magerr : array of magnitude errors
    mag_mean : float of magnitude mean
    n : int of number of observations
    """

    delta = (n/(n-1)) ** 0.5 * (mag-mag_mean)/mag_err

    p = delta * delta[:, np.newaxis]
    k = int(n*(n-1)//2)
    unique_pair_indicies = np.triu_indices(n,1)
    p = p[unique_pair_indicies]

    J = ( np.sign(p)*(np.abs(p)**0.5) ).sum()/k

    return J

def welch_stetson_j(mag, mag_err, mag_mean, n):
    """
    Welch-Stetson variability index K (Stetson 1996)

    mag : array of magnitudes
    magerr : array of magnitude errors
    mag_mean : float of magnitude mean
    n : int of number of observations
    """

    delta = (n/(n-1)) ** 0.5 * (mag-mag_mean)/mag_err

    K = n**-0.5 * np.abs(delta).sum() / ((delta**2).sum()**0.5)

    return K