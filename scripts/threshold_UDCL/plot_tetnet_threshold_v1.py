
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append('../../functions/')
import tetraplot4class as tp

from threshold_UDCL_model import ThresholdUDCL

import time

# ----------------------------------------------------------------------------------------------

# parameters defaults that sometimes get changed below
# ----

extra_suffix = ''

# qV = [0.6]
qV = [0.8]


# parameters
# ---

game_pars = {
        'strat_names': ['D', 'C', 'L', 'U'],
        'tau': 5,               # lottery quorum and midpoint of benefits function minus 0.5
        'cognitive_cost': -0.1, # a small cost to understanding coordination
        'contrib_cost': -1,     # cost of contributing to public good
        'benefit_thresh_met': 3 # benefit when threshold is met
        }

# evolutionary parameters
evol_pars = {
        'group_formation_model': 'members recruit', # group formation model
        'sumprodm_dir': os.path.abspath('../../members_recruit_comb_term/'),
        'q': 1,                                     # parameter for leader-driven model
        'n': 8,                                     # group size
        'n_s': len(game_pars['strat_names']),       # number of strategies
        'partn2prob_dir': os.path.abspath('../../results/partn2prob/'),
        }



# initialise the model
# ---

print('initialising model')
thresholdUDCL = ThresholdUDCL(evol_pars, game_pars, calc_minus1_system=True, precalc_payoffs=False)


# for each q value in qV, plot the dynamics on the net
# ---

for q in qV:

    print('-----')
    print(f'doing q = {q}')

    # update homophily parameter q
    thresholdUDCL.evol_pars['q'] = q
    thresholdUDCL.update_F()

    # initialise the plot and mesh
    ax = plt.figure().add_subplot(1,1,1)                    # get the plot axis object
    tris = tp.net_plot_initialise(ax, thresholdUDCL.strat_names)   # defines the corners of the triangles in the net plot
    lV = tp.net_get_face_mesh(2**5)                         # get a list of barycentric grid points for each face of the tetrahedron
    lV_coarse = tp.net_get_face_mesh(2**3)                  # get a list of barycentric grid points for each face of the tetrahedron

    # get dynamics

    print('calculating fixed points')
    start = time.time()
    fp_baryV = tp.net_find_fixed_points(lV_coarse, thresholdUDCL.deltap_fnc)              # find fixed points
    print(f'Time: {time.time() - start}')

    print('finding seln gradient')
    strengthsV, dirn_normV = tp.net_calc_grad_on_mesh(tris, lV, thresholdUDCL.deltap_fnc) # find gradient of selection


    # plot dynamics
    tp.net_plot_dynamics_rate(ax, tris, lV, strengthsV)     # plot rate of change (colour contour)
    tp.net_plot_dynamics_dirn(ax, tris, lV, dirn_normV)     # plot direction of selection (quiver plot)
    tp.net_plot_fixed_points_w_stability(ax, tris, fp_baryV, thresholdUDCL.calc_stability) # plot fixed points

    # name for the file
    suffix = thresholdUDCL.evol_pars['group_formation_model'].replace(' ', '_') + '_'.join(sorted(game_pars['strat_names'])) + '_q_' + str(int(10000*evol_pars['q'])) + extra_suffix
    fname = 'tetnet_thresholdUDCL_v1_' + suffix + '.pdf'

    plt.tight_layout()
    plt.savefig('../../results/threshold_UDCL/' + fname)
    plt.close('all')


    # write the fixed points to a file
    # ---

    # clean up the zeros and 1s
    for fp_bary in fp_baryV:
        for fp in fp_bary:
            fp[np.isclose(fp, 0, atol=1e-7)] = 0
            fp[np.isclose(fp, 1, atol=1e-7)] = 1

    # put into a big dataframe
    df_list = []
    for idx_omit, fp_bary in enumerate(fp_baryV):

        df = pd.DataFrame(fp_bary, columns = ['s1', 's2', 's3'])
        df.insert(0, 'idx_omit', [idx_omit]*len(fp_bary)) # add omitted index column
        df_list.append(df)

    df_all = pd.concat(df_list)

    # write to csv
    fname = 'fp_bary_threshold_v1_' + suffix + '.csv'
    df_all.to_csv('../../results/threshold_UDCL/' + fname, index=False)

