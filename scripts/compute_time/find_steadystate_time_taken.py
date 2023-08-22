# The purpose of this script is to compare the time taken by two methods to find the evolutionary 
# steady states, the "direct" approach we newly devised, and the "transformed payoff-matrix" 
# approach. The time taken by the latter is split into the time taken to evaluate the transformed 
# matrix and to find the steady states. This is is done to show that, for smaller group sizes, the 
# overhead costs of evaluating the transformed matrix can be worth the trade-off against the 
# efficiency of using matrix multiplications to evaluate the dynamics (done repeatededly).

import os
import numpy as np
import pandas as pd
import time

import sys
sys.path.append('../../functions/')
import tetraplot4class as tp

sys.path.append('../')
from sigmoid_UDCL.sigmoidUDCL_model import SigmoidUDCL as newdirectModel
from transmat_sigmoid_UDCL.sigmoidUDCL_model import SigmoidUDCL as transPaymModel


# parameter values
# ---

results_dir = '../../results/compute_time/'

# keep the ratio tau/n as close to this value as possible
tau_on_n = 5/8

# range of group sizes to find computational time for
n_min = 5
n_max = 11

# default parameter values, which we'll be changing as the group size increases

game_pars = {
        'strat_names': ['D', 'C', 'L', 'U'],
        'tau': 5,               # lottery quorum and midpoint of benefits function minus 0.5
        'steep': 10,            # steepness of sigmoid benefits function (\infty is threshold game)
        'cognitive_cost': -0.02,# a small cost to understanding coordination
        'contrib_cost': -0.25   # cost of contributing to public good
        }

# evolutionary parameters
evol_pars = {
        'group_formation_model': 'leader driven',   # group formation model
        'q': 0.85,                                  # parameter for leader-driven model, match main figure
        'n': 8,                                     # group size
        'n_s': len(game_pars['strat_names']),       # number of strategies
        'partn2prob_dir': os.path.abspath('../../results/partn2prob/'),
        }


# record times taken for range of group sizes n
# ---

# choose start points for finding the steady states
lV_coarse = tp.net_get_face_mesh(2**3)

time_takens = list()
for n in range(n_min, n_max+1):

    print(f'doing n = {n}...')

    # update parameter values

    evol_pars['n'] = n
    game_pars['tau'] = int(np.round(n*tau_on_n))

    # time taken using the transformed payoff-matrix approach

    # time taken to initialise and calculate transformed payoff matrix
    start = time.time()
    transpaym_model = transPaymModel(evol_pars, game_pars)
    time_transmat_overhead = time.time() - start

    # time taken to find the steady states
    start = time.time()
    transmat_fp_baryV = tp.net_find_fixed_points(lV_coarse, transpaym_model.deltap_fnc)
    time_transmat_steadystates = time.time() - start

    # time taken using our "newdirect" approach

    start = time.time()
    newdirect_model = newdirectModel(evol_pars, game_pars)
    newdirect_fp_baryV = tp.net_find_fixed_points(lV_coarse, newdirect_model.deltap_fnc)
    time_newdirect_total = time.time() - start

    # record all three

    time_taken = [n, time_transmat_overhead, time_transmat_steadystates, time_newdirect_total]
    time_takens.append(time_taken)


# write to csv file
# ---

df = pd.DataFrame(time_takens, columns=['group_size', 'transmat_overhead', 'transmat_steadystate', 'newdirect_total'])
df.to_csv(results_dir + 'steadystate_time_taken.csv', index=False)
