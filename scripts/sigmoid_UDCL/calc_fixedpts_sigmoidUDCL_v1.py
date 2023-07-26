import os
import numpy as np
import pandas as pd

import sys
sys.path.append('../../functions/')
import tetraplot4class as tp

from sigmoidUDCL_model import SigmoidUDCL

# ----------------------------------------------------------------------------------------------

# fixed-point finding parameters
# ---

ngrid = 9 # number of points on each axis
results_dir = '../../results/sigmoid_UDCL/'
qV = [0.81, 0.89, 0.8962, 0.8970, 1] # NOTE list of homophily parameters goes here


# model parameters
# ---

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
        'q': 1,                                     # parameter for leader-driven model
        'n': 8,                                     # group size
        'n_s': len(game_pars['strat_names']),       # number of strategies
        'partn2prob_dir': os.path.abspath('../../results/partn2prob/'),
        }



# initialise the model
# ---

print('initialising model')
sigmoidUDCL = SigmoidUDCL(evol_pars, game_pars)


# for each q value in qV, find the fixed points
# ---

for q in qV:

    print('-----')
    print(f'doing q = {q}')

    # update homophily parameter q
    sigmoidUDCL.evol_pars['q'] = q
    sigmoidUDCL.update_F()


    # find the fixed points in barycentric coordinates
    # ---

    print('finding fixed points')
    # divide the tetrahedron into a grid with ngrid points on each axis
    lV = tp.tet_get_mesh(ngrid)

    # find the fixed points numerically
    fps = tp.tet_find_fixed_points(lV, sigmoidUDCL.deltap_fnc)

    # zeros and ones are usually slightly off, so correct them
    for fp in fps:
        fp[np.isclose(fp, 0, atol=1e-7)] = 0
        fp[np.isclose(fp, 1, atol=1e-7)] = 1

    # I prefer a list of lists
    fps = [list(fp) for fp in fps]


    # check the stability of each fixed point
    # ---

    stabs = list()
    for fp in fps:

        # it doesn't make sense to talk about stability of monomorphic except wrt an invader
        stab = 'na' if fp.count(0) == 3 else sigmoidUDCL.calc_stability(fp)

        # store stability result
        stabs.append(stab)


    # check invasion fitness of each missing strategy into stable fixed points
    # ---

    inv_fitM = list()

    for fp, stab in zip(fps, stabs):

        if stab == 'unstable':

            # we're not interested in invasion fitness into unstable equilibria
            inv_fitM.append([np.nan]*4)

        else:

            inv_fitV = list()

            for inv_idx, inv_name in enumerate(game_pars['strat_names']):

                # only new strategies can invade
                inv_fit = np.nan if fp[inv_idx] != 0 else sigmoidUDCL.calc_invasion_fitness(inv_name, fp)

                # store invasion fitness for this invader
                inv_fitV.append(inv_fit)

            # store all invasion fitnesses for this fixed point
            inv_fitM.append(inv_fitV)


    # write the fixed points to a file
    # ---

    # create a large list of lists where each row corresponds to a fixed point
    M = [fp + [stab] + inv_fit for fp, stab, inv_fit in zip(fps, stabs, inv_fitM)]

    # write the corresponding column names
    colnames = ['p_' + strat_name + '*' for strat_name in game_pars['strat_names']]  # fixed points
    colnames += ['stability']
    colnames += ['inv_fit_' + strat_name for strat_name in game_pars['strat_names']] # invasion fitnesses

    # put it into a big dataframe
    df = pd.DataFrame(M, columns=colnames)

    # write it to a csv
    suffix = sigmoidUDCL.evol_pars['group_formation_model'].replace(' ', '_') + '_ngrid_' + str(ngrid) + '_q_' + str(int(10000*evol_pars['q'])) 
    fname = 'fixedpts_stability_sigmoidUDCL_v1_' + suffix + '.csv'
    df.to_csv(results_dir + fname, index=False)


