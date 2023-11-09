# a utility function to make latex tables of all the fixed points in the directory

import sys
sys.path.append('../../functions/')
from utilities import write_fixedpts_latex


# parameters
# ---

# directory containing fixed-point files and where the latex tables will be saved
results_dir = '../../results/transmat_sigmoid_UDCL'

# which file prefixes I want to make latex tables for
file_prefixV = [
    'fixedpts_stability_transmat_sigmoidUDCL_v2_leader_driven_ngrid_9',
        ]


# write latex table file for each
# ---

for file_prefix in file_prefixV:
    write_fixedpts_latex(results_dir, file_prefix)
