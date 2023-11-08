# similar to v1, but the slope ('steep') has been reduced so that tau contributors is not a Nash eqm

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append('../../functions/')
import tetraplot4class as tp

from sigmoidUDCL_model import SigmoidUDCL

import time

# ----------------------------------------------------------------------------------------------

# parameters defaults that sometimes get changed below
# ----

extra_suffix = ''

qV = [0, 0.41, 0.42, 0.46, 0.54, 0.6, 0.64, 0.68]
qV = [0.716, 0.724, 0.76, 0.7680, 0.8, 0.81, 0.82, 0.83, 0.84, 0.88, 0.92, 0.96, 1]
qV = [0.69, 0.70]
qV = [0.685, 0.71]
qV = [0.73, 0.74, 0.75, 0.735]
qV = [0.825, 0.827, 0.826]
qV = [0.9, 0.89]
qV = [0.4, 0.405]
qV = [0.805, 0.807]
qV = [0.806]

# fixed parameters
# ---

game_pars = {
        'strat_names': ['D', 'C', 'L', 'U'],
        'tau': 5,               # lottery quorum and midpoint of benefits function minus 0.5
        'steep': 6,             # steepness of sigmoid benefits function (\infty is threshold game)
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


# for each q value in qV, plot the dynamics on the net
# ---

for q in qV:

    print('-----')
    print(f'doing q = {q}')

    # update homophily parameter q
    evol_pars['q'] = q
    sigmoidUDCL = SigmoidUDCL(evol_pars, game_pars)


    # initialise the plot and mesh
    ax = plt.figure().add_subplot(1,1,1)                    # get the plot axis object
    tris = tp.net_plot_initialise(ax, sigmoidUDCL.strat_names)   # defines the corners of the triangles in the net plot
    lV = tp.net_get_face_mesh(2**5)                         # get a list of barycentric grid points for each face of the tetrahedron
    lV_coarse = tp.net_get_face_mesh(2**3)                  # get a list of barycentric grid points for each face of the tetrahedron

    # get dynamics

    print('calculating fixed points')
    start = time.time()
    fp_baryV = tp.net_find_fixed_points(lV_coarse, sigmoidUDCL.deltap_fnc)              # find fixed points
    print(f'Time: {time.time() - start}')

    print('finding seln gradient')
    strengthsV, dirn_normV = tp.net_calc_grad_on_mesh(tris, lV, sigmoidUDCL.deltap_fnc) # find gradient of selection


    # plot dynamics
    tp.net_plot_dynamics_rate(ax, tris, lV, strengthsV)     # plot rate of change (colour contour)
    tp.net_plot_dynamics_dirn(ax, tris, lV, dirn_normV)     # plot direction of selection (quiver plot)
    tp.net_plot_fixed_points_w_stability(ax, tris, fp_baryV, sigmoidUDCL.calc_stability) # plot fixed points

    # name for the file
    suffix = sigmoidUDCL.evol_pars['group_formation_model'].replace(' ', '_') + '_'.join(sorted(game_pars['strat_names'])) + '_q_' + str(int(10000*evol_pars['q'])) + extra_suffix
    fname = 'tetnet_transmat_sigmoidUDCL_v2_' + suffix + '.pdf'

    plt.tight_layout()
    plt.savefig('../../results/transmat_sigmoid_UDCL/' + fname)
    plt.close('all')
