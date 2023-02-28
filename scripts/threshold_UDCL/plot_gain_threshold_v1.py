import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

import sys
sys.path.append('../../functions/')
from threshold_UDCL_model import ThresholdUDCL


# fixed parameters
# ---

# list of pairs to plot, [ [name, strats] ]
L = [['C', 'D'], ['C', 'U'], ['C', 'L'], ['U', 'D']]

# where to save figures to
savedir = '../../results/threshold_UDCL/'

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

# a mapping from the strategy name codes to a pretty name for plotting
pretty_names = {
        'D': 'Unconditional Defectors',
        'C': 'Coordinating Cooperators',
        'U': 'Unconditional Cooperators',
        'L': 'Liars',
        }


# initialise the model
# ---

print('initialising model')
thresholdUDCL = ThresholdUDCL(evol_pars, game_pars, calc_minus1_system=False, precalc_payoffs=False)


# for each pair in the list L, plot the payoffs, gain sequence, and gain function
# ---

# possible strategy counts of focal and non-focal
n = evol_pars['n']
strat_countsV = [ [i, n-i] for i in range(1, n+1) ]

for strats in L:


    # calculate payoffs given k cooperators in n-1 other members
    # ---

    # strategy 1 payoffs 
    aV = [thresholdUDCL.payoff(strats, strat_counts) for strat_counts in strat_countsV]

    # strategy 2 payoffs 
    r_strats = list(reversed(strats))
    bV = [thresholdUDCL.payoff(r_strats, strat_counts) for strat_counts in reversed(strat_countsV)]

    # gain sequence
    dV = [ ai-bi for ai, bi in zip(aV, bV) ]

    # gain function
    pV = np.linspace(0, 1, 50)
    gV = [ sum( binom(n-1, k) * p**k * (1-p)**(n-1-k) * dV[k] for k in range(n) ) for p in pV ]


    # plot
    # ---

    # get the short names and pretty names of the two strategies
    focal_strat = strats[0]
    other_strat = strats[1]
    focal_pretty = pretty_names[focal_strat]
    other_pretty = pretty_names[other_strat]

    fig = plt.figure()
    f, (ax3, ax1) = plt.subplots(2, 1, figsize=(5,7))

    # plot gain sequence and function
    ax1.plot([i/(n-1) for i in range(n)], dV, '-o', color='black', label=r'gain sequence')
    ax1.plot(pV, gV, color='brown', label=r'gain function')
    ax1.axhline(0, color='black', alpha=0.5)
    ax1.set_xlabel(r'fraction of ' + focal_pretty + ' in the population', color='brown', fontsize='medium')
    ax1.set_ylabel(r'gain sequence & function', fontsize='medium')
    ax1.legend(loc='best')

    # add top axis for gain sequence
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(np.linspace(0,1,n))
    ax2.set_xticklabels(range(n))

    # plot payoffs
    ax3.plot(range(n), aV, '-o', color='blue', label=focal_strat)
    ax3.plot(range(n), bV, '-o', color='red', label=other_strat)
    ax3.xaxis.set_ticks_position('both')
    # ax3.xaxis.set_label_position('top')
    ax3.tick_params(labelbottom=False,labeltop=True)
    ax3.set_xlabel(r'no. of ' + focal_pretty + ' among $n-1$ others', fontsize='medium')
    ax3.xaxis.set_label_position('top')
    ax3.set_ylabel(r'payoff', fontsize='medium')
    ax3.legend(loc='best')

    # save
    plt.suptitle(focal_pretty + ' vs ' + other_pretty, fontsize='large')
    plt.tight_layout()
    plt.savefig(savedir + 'gain_threshold_v1_' + '_Vs_'.join(strats) + '.pdf')
    plt.close('all')

