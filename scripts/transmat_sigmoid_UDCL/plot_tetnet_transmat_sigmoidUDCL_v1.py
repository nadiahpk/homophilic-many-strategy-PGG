# This script replicates the results in ../sigmoid_UDCL/ using the transformed payoff-matrix approach.
# Compare the result this script produces:
#   ../../results/transmat_sigmoid_UDCL/tetnet_transmat_sigmoidUDCL_v1_leader_drivenC_D_L_U_q_8500.pdf
# with the result produced by the "efficient" method:
#   ../../results/sigmoid_UDCL/tetnet_sigmoidUDCL_v1_leader_drivenC_D_L_U_q_8500.pdf
# to verify the methods are equivalent.

import matplotlib.pyplot as plt
import os
import numpy as np
import time

import sys
sys.path.append('../../functions/')
import tetraplot4class as tp

from sigmoidUDCL_model import SigmoidUDCL


# parameters
# ----

extra_suffix = ''

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


# initialise the model
# ---

print('initialising model')
sigmoidUDCL = SigmoidUDCL(evol_pars, game_pars) # calculates the transformed payoff matrix


# plot the dynamics
# ---

# initialise the plot and mesh
ax = plt.figure().add_subplot(1,1,1) # create the plot axis object
tris = tp.net_plot_initialise(ax, sigmoidUDCL.strat_names) # define the corners of the triangles in the net plot

# get a list of barycentric grid points for each face of the tetrahedron
lV = tp.net_get_face_mesh(2**5)        # fine mesh
lV_coarse = tp.net_get_face_mesh(2**3) # coarse mesh


# get dynamics

print('calculating fixed points')
start = time.time()
fp_baryV = tp.net_find_fixed_points(lV_coarse, sigmoidUDCL.deltap_fnc)
print(f'Time: {time.time() - start}')

# find gradient of selection at each point on fine grid
print('finding seln gradient')
strengthsV, dirn_normV = tp.net_calc_grad_on_mesh(tris, lV, sigmoidUDCL.deltap_fnc)


# plot dynamics
tp.net_plot_dynamics_rate(ax, tris, lV, strengthsV) # rate of change (colour contour)
tp.net_plot_dynamics_dirn(ax, tris, lV, dirn_normV) # direction of selection (arrows)
tp.net_plot_fixed_points_w_stability(ax, tris, fp_baryV, sigmoidUDCL.calc_stability) # fixed points 

# name for the file
suffix = sigmoidUDCL.evol_pars['group_formation_model'].replace(' ', '_') + '_'.join(sorted(game_pars['strat_names'])) + '_q_' + str(int(10000*evol_pars['q'])) + extra_suffix
fname = 'tetnet_transmat_sigmoidUDCL_v1_' + suffix + '.pdf'

plt.tight_layout()
plt.savefig('../../results/transmat_sigmoid_UDCL/' + fname)
plt.close('all')

