import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('PS_secondary.csv')

so = df[df['uid'] == df.uid.value_counts().index[1]]

plt.scatter((so['ra']-so['ra'].mean())*3600,(so['dec']-so['dec'].mean())*3600)

fig, ax = plt.subplots(1,1,figsize=(10,5))

for band, color in zip('grizy','grkby'):
    ax.plot(so[so['filter'] == band]['obsTime'],so[so['filter'] == band]['psfFlux'], color = color, lw = 0.5)