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
import matplotlib.pyplot as plt
from numpy.random import normal

sigma = 2
X = normal(0, sigma, size=5000000)
Y = X**2
x = np.linspace(0.1,5*sigma, 100)


# +
def pdf_Y(x, sigma):
    return (2*np.pi*x*sigma**2)**-0.5*np.exp(-0.5*x/(sigma**2))
    
def pdf_X(x, sigma):
    return (2*np.pi*sigma**2)**-0.5*np.exp(-0.5*(x/sigma)**2)


# +
fig, ax = plt.subplots(1,1, figsize = (17,8))

ax.hist(X, bins=301, range=(-6*sigma,6*sigma), alpha=0.5, density=True);
ax.hist(X**2, bins=301, range=(0,20*sigma), alpha=0.5, density=True);
ax.plot(x, pdf_X(x, sigma), color = 'b')
ax.plot(x, pdf_Y(x, sigma), color = 'orange')
ax.set(ylim=(0,0.3), xlim=(-3,3))

# +
from scipy.integrate import quad
from scipy.optimize import minimize

def err(upper, sigma):
    return abs(quad(pdf_Y, 0, upper, args=(sigma))[0] - 0.6826894)

ci = [] # confidence interval
var = []
for sigma in sigmas:
    upper = minimize(err, 1, args=(sigma), tol=1e-3)['x'][0]
    ci.append(upper)
    Y = normal(0, sigma, size=5000000) ** 2
    var.append(np.var(Y))
    
ci = np.array(ci)
var = np.array(var)
# -

var**0.5

ci

var**0.5/ci

var/(2*sigmas**4)

Z = X**4

Z.var()

96*X.var()**4


