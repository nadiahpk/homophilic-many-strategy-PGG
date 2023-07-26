# Plot the time taken to find steady states for two methods, the "direct" approach we newly 
# devised, and the "transformed payoff-matrix" approach. 

import matplotlib.pyplot as plt
import pandas as pd


# parameters
# ---

# where the results are stored
results_dir = '../../results/compute_time/'
results_fname = 'time_taken.csv'


# create plot
# ---

# read in csv
df = pd.read_csv(results_dir + results_fname)

# plot each component

# plot size I like
plt.figure(figsize=(5, 3.8))

# the overhead for calculating the transformed payoff matrix
plt.scatter(df['group_size'], df['transmat_overhead'], s=20, facecolors='none', edgecolors='red', label='transformed payoff-matrix overhead')
plt.plot(df['group_size'], df['transmat_overhead'], ls='dashed', color='red')

# the total using the transformed payoff matrix approach
plt.scatter(df['group_size'], df['transmat_overhead'] + df['transmat_steadystate'], s=20, color='red', label='transformed payoff-matrix total')
plt.plot(df['group_size'], df['transmat_overhead'] + df['transmat_steadystate'], ls='solid', color='red')

# the total using the new direct approach approach
plt.scatter(df['group_size'], df['newdirect_total'], s=20, color='blue', label='direct approach total')
plt.plot(df['group_size'], df['newdirect_total'], ls='solid', color='blue')

# decorations
plt.legend(loc='best')
plt.xlabel('group size $n$', fontsize='x-large')
plt.ylabel('computation time (s)', fontsize='x-large')

plt.tight_layout()
plt.savefig(results_dir + 'time_taken.pdf')
plt.close('all')
