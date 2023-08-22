# not sure if it'll look any good 3D, but try it anyway

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# parameters
# ---

# where the results are stored
results_dir = '../../results/compute_time/'
results_fname = 'deltap_time_taken.csv'


# create plot
# ---

# read in csv
df = pd.read_csv(results_dir + results_fname)

# split it into vectors
mV = list(df['m'])
nV = list(df['n'])
tV = list(df['time_taken'])

# meshgrid it

ms = sorted(set(mV))
ns = sorted(set(nV))

m2idx = {m: idx for idx, m in enumerate(ms)}
n2idx = {n: idx for idx, n in enumerate(ns)}
len_m = len(ms)
len_n = len(ns)
tM = np.zeros((len_n, len_m))
for m, n, t in zip(mV, nV, tV):
    tM[n2idx[n], m2idx[m]] = t

mM, nM = np.meshgrid(ms, ns)

# plot it

fig = plt.figure(figsize=(5, 3.8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(mM, nM, tM, edgecolor='black', color='blue', alpha=0.5)
ax.set_xlabel('number of strategies, $m$')
ax.set_ylabel('number of players, $n$')
ax.set_zlabel('computation time (s)')
ax.set_zlim((0,20))
plt.savefig(results_dir + 'deltap_time_taken_3D.pdf')
plt.close('all')
