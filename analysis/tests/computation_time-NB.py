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



from time import time

# x = [int(a) for a in [1e1,5e1,1e2,5e2,1e3,5e3]]
x = np.logspace(1,np.log10(len(uids)),10)
x = x.astype('int')
n = len(x)

start = time()
subdf = dr.df.loc[uids,['mjd','mag','catalogue']]
end = time()
end - start

times1 += end-start

times1 = np.zeros(n)
start = time()
subdf = dr.df.loc[uids,['mjd','mag','catalogue']]
for i in range(n):
    for uid in uids[:x[i]]:
        a = subdf.loc[uid]
    end = time()
    times1[i] = end - start

times2 = np.zeros(n)
for i in range(n):
    start = time()
    for uid in uids[:x[i]]:
        b = dr.df.loc[uid,['mjd','mag','catalogue']]
    end = time()
    times2[i] = end - start

plt.plot(times1, color='r', label = 'subdf')
plt.plot(times2, color='b', label = 'wholedf')
plt.legend()

# for len(uids) = 5e4
plt.plot(times1, color='r', label = 'subdf')
plt.plot(times2, color='b', label = 'wholedf')
plt.legend()

a=1



