# plot of trajectories starting near U

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append('../../functions/')
import tetraplot4class as tp

from threshold_UDCL_model import ThresholdUDCL

# ----------------------------------------------------------------------------------------------

# parameters
# ---

results_dir = '../../results/threshold_UDCL/'
fixedpts_file = 'fixedpts_stability_threshold_v1_members_recruit_ngrid_7_q_8000.csv'

# where the fixed points are stored

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
        'q': 0.8,                                     # parameter for leader-driven model
        'n': 8,                                     # group size
        'n_s': len(game_pars['strat_names']),       # number of strategies
        'partn2prob_dir': os.path.abspath('../../results/partn2prob/'),
        }


# initialise the model
# ---

print('initialising model')
thresholdUDCL = ThresholdUDCL(evol_pars, game_pars)


# initialise the plot
# ---

ax = plt.figure().add_subplot(projection='3d')
vertexs = tp.tet_plot_initialise(ax, thresholdUDCL.strat_names)


# find and plot the fixed points
# ---

if fixedpts_file:

    # get the fixed points and information about them stored in the file
    df = pd.read_csv(results_dir + fixedpts_file)

    # create a list of the fixed points
    fpV = list(df[['p_' + strat_name + '*' for strat_name in game_pars['strat_names']]].to_numpy())

    # corresponding list of stability information
    stabV = list(df['stability'])

    # encode each stability as a colour
    colourV = ['red' if stab == 'unstable' else 'blue' if stab == 'stable' else 'black' for stab in stabV]

    # convert bary coordinates to xyz coords
    xyzV = [ list(tp.tet_bary2xyz(vertexs, ls)) for ls in fpV ]
    xV, yV, zV = zip(*xyzV)

    # plot fixed point in colour corresponding to stability
    ax.scatter(xV, yV, zV, color=colourV)

else:

    print('finding fixed points')
    lV = tp.tet_get_mesh(3) # mesh for starting points
    fp_bary = tp.tet_find_fixed_points(lV, thresholdUDCL.deltap_fnc)
    tp.tet_plot_fixed_points(ax, vertexs, fp_bary)


# plot some trajectories
# ---

# choose starting points near U

pert = 0.01
nopts = 5
vals = pert*np.linspace(0, 1, nopts)[1:]

nopts -= 1

v0_tspans = list()
for ptD in range(nopts):

    for ptC in range(nopts-ptD):

        ptL = nopts-ptD-ptC-1
        pts = [ptD, ptC, ptL]
        v0 = [vals[i] for i in pts] + [1-pert]
        v0_tspans.append((v0, [0, 40]))

print('finding trajectories')

xM = list()
yM = list()
zM = list()

for v0, tspan, in v0_tspans:

    print(f'doing v0 = {v0}')
    xV, yV, zV = tp.get_trajectory(vertexs, v0, tspan, thresholdUDCL.deltap_fnc)
    xM.append(xV)
    yM.append(yV)
    zM.append(zV)


# plot them
for xV, yV, zV in zip(xM, yM, zM):
    ax.plot(xV, yV, zV, color='black', lw=1, alpha=0.5)


# finalise plot and save
# ---

#plt.legend(loc='best')
plt.tight_layout()
suffix = thresholdUDCL.evol_pars['group_formation_model'].replace(' ', '_') + '_'.join(sorted(game_pars['strat_names'])) + '_q_' + str(int(10000*evol_pars['q'])) 
fname = 'tet3d_thresholdUDCL_v1_startU_' + suffix + '.pdf'
plt.savefig(results_dir + fname)
plt.close()
