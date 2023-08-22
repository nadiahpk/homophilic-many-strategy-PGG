# The purpose of this script is to record the time required to evaluate delta p once
# as it varies with group size (n) and number of strategies (m). The payoff() function
# I'm using isn't a real function, it always just returns 1.

import os
import pandas as pd

import sys
sys.path.append('../../functions/')
import tetraplot4class as tp

from my_model import MyModel

import time

# ----------------------------------------------------------------------------------------------

# fixed parameters
# ---

# results directory
results_dir = '../../results/compute_time/'

# filename suffix in case I want to append data later
suffix = '_append'

# number of strategies
n_s_start = 3
n_s_end = 9

# start and end range for number of players in the group
n_start = 3
n_end = 9


# loop through (n_s, n) combinations timing how long delta p takes
# ---

res = list() # a place to store results [[n_s, n, time_taken]]
for n_s in range(n_s_start, n_s_end+1):
    for n in range(n_start, n_end+1):

        # initialise the model
        # ---

        # fake parameter names for each strategy
        game_pars = {'strat_names': [str(i) for i in range(n_s)]}

        # using the members attract model with alpha = 5 so we are calculating terms for every F
        evol_pars = {
                'group_formation_model': 'members attract',
                'alpha': 5,                                 # parameter for members-attract model
                'n': n,                                     # group size
                'n_s': n_s,                                 # number of strategies
                'partn2prob_dir': os.path.abspath('../../results/partn2prob/'),
                }

        # initialise the model
        print(f'initialising model with n = {n} and m = {n_s}')
        my_model = MyModel(evol_pars, game_pars, calc_minus1_system=False, precalc_payoffs=False)


        # record time taken to calculate delta p
        # ---

        start = time.time()
        strat_ps = [1/n_s for i in range(n_s)] # make every strategy present so every calculation must be made
        deltap = my_model.deltap_fnc(strat_ps)
        time_taken = time.time() - start

        # store
        res.append([n_s, n, time_taken])


# write it to a csv
# ---

df = pd.DataFrame(res, columns=['m', 'n', 'time_taken'])
df.to_csv(results_dir + 'deltap_time_taken' + suffix + '.csv', index=False)
