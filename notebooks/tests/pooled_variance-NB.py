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
np.random.seed(42)



true_var = a.var()
true_var

n = np.random.randint(10,10000, size=(shape[0]))
n = (np.round(n/n.sum()*shape[1]))

n = np.random.randint(10,10000, size=(shape[0]))
n/n.sum()*shape[1]


def calculate_pooled_mean_and_variance(a,shape):
    """
    Calculate pooled mean and variance for an array.
    """
    # Calculate pooled mean and variance
    b = a.reshape(shape)
    # n = np.full(shape[0], shape[1])
    n = np.random.randint(10,10000, size=(shape[0]))
    n = (np.round(n/n.sum(axis=0)*shape[1]*1000))
    b_mean = b.mean(axis=-1)
    b_var  = b.var(axis=-1)
    # https://en.wikipedia.org/wiki/Law_of_total_variance
    # https://arxiv.org/pdf/1007.1012.pdf
    # https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    # Pooled mean is the weighted average of the means
    pooled_mean = np.average(b_mean, weights = n, axis=0)
    # Pooled variance is the mean of the variances plus the variance of the means
    pooled_var  = np.average(b_var,  weights = n, axis=0) + np.average((b_mean-pooled_mean)**2, weights = n, axis=0)
    return pooled_mean, pooled_var


# +
a = np.random.normal(2,0.2, size=10000)
# a += np.random.normal(8,0.2, size=10000)
print('pooled mean and variance:')
for shape in [(20,500),(2,5000),(1,20*500),(10*500,2),(25, 400)]:
    print(calculate_pooled_mean_and_variance(a, shape))

print(f'true mean and variance:\n({a.mean()}, {a.var()})')
# -

# We see that the pooled variances are identical. Thus it does not matter if we group a into 20 groups of 500 or 200 groups of 50.

# to test the above in practice, the code below splits the 


