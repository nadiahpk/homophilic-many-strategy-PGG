# writes compressed CM*.csv for group size n and number of strategies n_s

import pandas as pd
import numpy as np
import itertools as it
from math import factorial

import sys
sys.path.append('../../functions')

from utilities import partitionInteger


# parameters
# ---

# a list of group sizes and number of strategies to calculate, (group_size, no_strategies)
# n_nsV = [(3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
# n_nsV = [(4, 2)]
n_nsV = [(5, 2)]
n_nsV = [(10, 2)]


# for each group size + number of strategies combination, calculate and store the matrices
# ---

for n, n_s in n_nsV:

    print(f'doing group size = {n} and number of strategies = {n_s}')
    # e.g., 
    # n = 5       # group size
    # n_s = 3     # three strategies, e.g., cooperate, defect, punish


    # find all the partitions of n, which are the possible family-partition structures
    # ---

    # make a mapping from F to index
    partn2idx = { tuple(partn): i for i, partn in enumerate(partitionInteger(n)) }
    # -- {(1, 1, 1, 1, 1): 0, (1, 1, 1, 2): 1, (1, 1, 3): 2, (1, 2, 2): 3, (1, 4): 4, (2, 3): 5, (5,): 6}


    # find all the possible strategy outcomes in a group where tag of non-focal can be swapped
    # ---

    '''
    # doing it the long way, a reminder
    strat_countsV = list()
    for no_focal in range(1, n+1):

        non_focal_outcomes = [ outcome for outcome in it.combinations_with_replacement(range(n_s-1), n-no_focal) ]
        non_focal_counts = set([ tuple(sorted( outcome.count(s) for s in range(n_s-1) )) for outcome in non_focal_outcomes ])
        for non_focal_count in non_focal_counts:
            strat_countsV.append( (no_focal,) + non_focal_count )
    '''

    strat_countsV = [ (no_focal,) + non_focal_count for no_focal in range(1, n+1) 
            for non_focal_count in set([ tuple(sorted( outcome.count(s) for s in range(n_s-1) )) 
                for outcome in it.combinations_with_replacement(range(n_s-1), n-no_focal) ]) ]
    # -- [(1, 1, 3), (1, 0, 4), (1, 2, 2), (2, 1, 2), (2, 0, 3), (3, 1, 1), (3, 0, 2), (4, 0, 1), (5, 0, 0)]


    # create matrices for storing results
    # ---

    no_rows = len(strat_countsV)
    no_cols = len(partn2idx)

    # coefficients matrix
    CM = [ [ [] for col in range(no_cols) ] for row in range(no_rows) ]

    # powers matrix
    WM = [ [ [] for col in range(no_cols) ] for row in range(no_rows) ]

    for row, strat_counts_ in enumerate(strat_countsV):

        # remove the zeros, i.e., the strategies that aren't in this outcome
        strat_counts = tuple( cnt for cnt in strat_counts_ if cnt != 0 )


        # find all partitions of each strategy
        # ---

        partn_sV = list()
        for strat_count in strat_counts:

            partn_s = list(partitionInteger(strat_count))
            partn_sV.append(partn_s)

        # create all combinations of all partitions of each strategy
        partnV = list(it.product(*partn_sV))

        # flatten to ID later
        partnV_flat = [ tuple(sorted(it.chain(*v))) for v in partnV ]


        # find the powers of each partition
        # ---

        pwrsV = [ [ len(partn_s) for partn_s in partn ] for partn in partnV ]


        # find the coefficient for each partition, \gamma_x C(Z)
        # ---

        # the number of ways the strategies can be rearranged into same-sized partitions

        coeffV = list()
        for partn_stratV, partn in zip(partnV, partnV_flat):

            '''
            # old version that is inefficient, but I keep it in comments to match my old notes

            # calculate the phis
            i_phi = [ (i, partn.count(i)) for i in set(partn) ]
            i_phi_stratV = [ [ (i, partn_strat.count(i)) for i in set(partn_strat) ] for partn_strat in partn_stratV ]

            # calculate constant term prod_{i=1}^n ( phi_i !) / prod_{s in S} phi_{s,i}! 
            prod_term = np.prod([ factorial(phi) for i, phi in i_phi ]) \
                    / np.prod([ factorial(phi_strat)  for i_phi_strat in i_phi_stratV for i, phi_strat in i_phi_strat ])

            # calculate sum_j j phi_{Aj}
            sum_term = sum( i*phi for i, phi in i_phi_stratV[0] ) # NOTE - this is just kx, you can replace it

            coeff_sum = sum_term*prod_term

            #print('coeff_sum = ' + str(coeff_sum))
            coeffV.append(int(np.round(coeff_sum))) # just trying to be cautious about rounding
            '''

            # \sum_{i=1}^m z_ij terms in the numerator
            sum_z_js = [partn.count(size) for size in set(partn)]

            # z_{i,j} terms in the denominator
            z_ijs = [partn_strat.count(size) for partn_strat in partn_stratV for size in set(partn_strat)]

            # \gamma_x C(Z) always an integer, so can use integer divide here
            coeff = strat_counts[0] * np.prod([factorial(sum_z_j) for sum_z_j in sum_z_js]) \
                    // np.prod([factorial(z_ij) for z_ij in z_ijs])

            coeffV.append(coeff)


        # store coefficients and powers in their matrices
        # ---

        for coeff, pwrs, partn in zip(coeffV, pwrsV, partnV_flat):

            col = partn2idx[partn]
            CM[row][col].append(coeff)
            WM[row][col].append(pwrs)



    # write the matrices CM and WM to a csv file
    # ---

    # no_strat_1, no_strat_2, ..., coef_1|1|1|1, coef_1|1|2, ..., pwrs_1|1|1|1
    # 2,          2,               1           , 1|1,           , 2|1*1|2

    # write strings for the headers
    no_strat_headers = [ 'no_strat_' + str(i) for i in range(1, n_s+1) ]
    coef_headers = [ 'coef_' + '|'.join([str(v) for v in partn]) for partn in partn2idx.keys() ]
    pwrs_headers = [ 'pwrs_' + '|'.join([str(v) for v in partn]) for partn in partn2idx.keys() ]


    # create a matrix of strings for the coefficients

    CM_str = [ [ '' for col in range(no_cols) ] for row in range(no_rows) ]

    for row in range(no_rows):
        for col in range(no_cols):

            coefs = CM[row][col]
            CM_str[row][col] = '|'.join([str(v) for v in coefs])


    # create a matrix of strings for the powers

    WM_str = [ [ '' for col in range(no_cols) ] for row in range(no_rows) ]

    for row in range(no_rows):
        for col in range(no_cols):

            pwrsV = WM[row][col]
            WM_str[row][col] = '*'.join([ '|'.join([str(w) for w in pwrs]) for pwrs in pwrsV ])


    # construct the dataframe

    df_no_strat = pd.DataFrame(strat_countsV, columns = no_strat_headers)
    df_coef = pd.DataFrame(CM_str, columns = coef_headers, dtype="string")
    df_pwrs = pd.DataFrame(WM_str, columns = pwrs_headers, dtype="string")

    df = pd.concat([df_no_strat, df_coef, df_pwrs], axis=1) # merge


    # write
    df.to_csv('../../results/partn2prob/CW_groupsize' + str(n) + '_numstrategies' + str(n_s) + '.csv', index=False)

