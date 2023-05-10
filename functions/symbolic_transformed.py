# Helpful functions for doing analytic work in Jupyter with the transformed payoff matrix.

import sympy as sp
import numpy as np
import itertools as it
from sympy.utilities.iterables import multiset_permutations
import pandas as pd

from utilities import partitionInteger


def calc_switching_gains(A, k_idx, o_idx):
    '''
    Calculate symbolic expressions for the switching gain for each k = 0, ..., n-1,
    where k is the number of nonfocal individuals pursuing strategy s_{k_idx}
    against n-1-k individuals pursuing strategy s_{o_idx}. These switching gains can
    give qualitative insights into the dynamics between the two strategies,
    as described in Peña et al. (2014, J Theor Biol).


    Inputs:
    ---

    A, symbolic matrix of size m^n

        The payoff matrix where each dimension corresponds to an individual in the group, each 
        index corresponds to the strategy played by the individual, and the leading index corresponds
        the focal individual.

    k_idx, int
        
        The index of the focal strategy.

    o_idx, int

        The index of the other strategy.


    Outputs:
    ---

    dk, list of n symbolic expressions
        
        The switching gains, from strategy o_idx to k_idx, for k = 0, ..., n-1

    pi_focV, list of n symbolic expressions

        The payoff to the focal-strategy when grouped with k of n-1 others playing
        the focal strategy (s_{k_idx}) and n-1-k others playing the nonfocal strategy (s_{o_idx})

    pi_nonV, list of n symbolic expressions

        The payoff to the nonfocal-strategy when grouped with k of n-1 others playing
        the focal strategy
    '''

    # number of individuals in the group
    n = len(A.shape)

    # k is the number of focal-strategists among the n-1 other players
    kV = [i for i in range(n)]
    
    # each k in kV corresponds to indices in A, and this is 1 possibility
    non_idxsV = [[k_idx]*k + [o_idx]*(n-1-k) for k in kV]
    # -- e.g., k_idx = 3, o_idx = 0, n = 3, 
    #          then non_idxsV = [[0, 0], [3, 0], [3, 3]]
    
    # payoffs to focal
    pi_focV = [A[tuple([k_idx] + non_idxs)] for k, non_idxs in zip(kV, non_idxsV)]
    
    # payoffs to nonfocal
    pi_nonV = [A[tuple([o_idx] + non_idxs)] for k, non_idxs in zip(kV, non_idxsV)]
    
    # switching gains
    dk = [pi_foc - pi_non for pi_foc, pi_non in zip(pi_focV, pi_nonV)]
    
    return (dk, pi_focV, pi_nonV)


def create_payoff_matrix(n, m, pi):
    '''
    Create the payoff matrix corresponding to the payoff function pi.


    Inputs:
    ---

    n, int

        Number of individuals in the group

    m, int

        Number of strategies in the game

    pi, function pi(gamma_focal, gamma_nonfoc)

        Payoff function called $pi(gamma_0, \boldsymbol{gamma}_{-0})$ in the main text.
        - gamma_focal is a symbolic matrix of length m with a 1 in the i-th position indicating
          that the focal individual plays strategy s_i
        - gamma_nonfoc is a symbolic matrix of length m where the value of each entry j is how
          many individuals among the n-1 nonfocal group members play strategy s_j


    Outputs:
    ---

    A, symbolic matrix of size m^n

        The payoff matrix where each dimension corresponds to an individual in the group, each 
        index corresponds to the strategy played by the individual, and the leading index corresponds
        the focal individual.
    '''

    # initialise the payoff matrix with all zeros
    A = sp.MutableDenseNDimArray(np.zeros([m]*n, dtype=int))

    # a list of possible strategy combinations played by the nonfocal members
    nonf_stratsV = list(it.combinations_with_replacement(range(m), n-1))
    # -- e.g., [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    # the strategy indicator vector, \boldsymbol{e} in the text
    e = lambda i: sp.Matrix([1 if idx == i else 0 for idx in range(m)])

    # populate the A matrix with payoffs

    for foc_strat in range(m):
        for nonf_strats in nonf_stratsV:

            # this is just one of the A indices corresponding to this group strategy composition
            idxs = tuple([foc_strat] + list(nonf_strats))
            
            # the gamma_{-0} for nonfocals
            gamma_n = sp.Matrix([nonf_strats.count(strat_idx) for strat_idx in range(m)])

            # calculate the payoff for this
            payoff = pi(e(foc_strat), gamma_n)

            # because the payoff matrix is symmetric around each leading index value,
            # this payoff can be inserted at every location in A corresponding to permutation
            # of the nonfocal indices

            for perm_nonf_strats in multiset_permutations(nonf_strats):

                idxs = tuple([foc_strat] + perm_nonf_strats)
                A[idxs] = payoff

    return A


def create_transformed_payoff_matrix(A):
    '''
    Create the transformed payoff matrix that will produce the same dynamics in a well-mixed population
    as the original payoffs in an homophilic population.


    Inputs:
    ---

    A, symbolic matrix of size m^n

        The original payoff matrix. Each dimension corresponds to an individual in the group, each 
        index corresponds to the strategy played by the individual, and the leading index corresponds
        the focal individual.


    Outputs:
    ---

    B, symbolic matrix of size m^n

        The transformed payoff matrix.
    '''

    # get the dimensions of A, which correspond to the no. individuals in the group and strategies in the game
    A_shape = A.shape
    n = len(A_shape)    # number of individuals in the group
    m = A_shape[0]      # number of strategies in the game

    # the possible family partition structures of the group
    partnV = list(partitionInteger(n))
    len_partn = len(partnV)

    # a list of possible strategy combinations played by the nonfocal members
    nonf_stratsV = list(it.combinations_with_replacement(range(m), n-1))

    # populate the B matrix
    # ---

    B = sp.MutableDenseNDimArray(np.zeros([m]*n, dtype=int))

    for foc_strat in range(m):
        for nonf_strats in nonf_stratsV:
            
            # this is just one of the B indices corresponding to this group strategy composition
            idxs = tuple([foc_strat] + list(nonf_strats))

            # calculate the payoff for this entry of B and others symmetric with it
            payoff = 0
            
            for F_idx, partn in enumerate(partnV):

                # we need to go through each possible partition size the focal could be in
                partnS = set(partn)

                for foc_partn_size in partnS:

                    # the probability that the focal is in a partition of this size is
                    P_foc_psize = sp.Rational(partn.count(foc_partn_size)*foc_partn_size, n)

                    # the non-focal's kin partitions remaining
                    nonf_partn = [sz for sz in partn] # deep copy
                    nonf_partn.remove(foc_partn_size)

                    # to find all possible allocations of nonfocals j, k, l, etc. into diff sized partns,
                    # we need to find all possible multiset permutations of (the zeros 
                    # are when there are none)

                    find_perms_of = nonf_partn + [0]*(n-1-len(nonf_partn))
                    nonf_allocs = list(multiset_permutations(find_perms_of))
                    P_alloc = sp.Rational(1, len(nonf_allocs))

                    for nonf_alloc in nonf_allocs:

                        # this vector says how many having the strategy of i, j, k, etc.
                        whole_alloc = [foc_partn_size] + nonf_alloc

                        # the index that whole_alloc is referring to is
                        ref_idxs = tuple(idxs[i] for i in range(n) for rep in range(whole_alloc[i]))

                        # so append to B
                        payoff += sp.Symbol('F_'+str(F_idx+1))*P_foc_psize*P_alloc*A[ref_idxs]
                        # here, we created on the fly symbolic parameters like $F_2$, which is the probability
                        # of a group forming with family partition structure "2"


            # because the payoff matrix is symmetric around each leading index value,
            # this payoff can be inserted at every location in B corresponding to permutation
            # of the nonfocal indices

            for perm_nonf_strats in multiset_permutations(nonf_strats):

                idxs = tuple([foc_strat] + perm_nonf_strats)
                B[idxs] = payoff

    return B


def tabulate_switching_gains(A, B, strat_names, strat_pairs):
    '''
    Returns latex tables of the switching gains for all pairs of strategies in strat_pairs for 
    (1) in a well-mixed population, (2) with some homophily, (3) with perfect homophily

    Inputs:
    ---

    A, symbolic matrix of size m^n

        The payoff matrix where each dimension corresponds to an individual in the group, each 
        index corresponds to the strategy played by the individual, and the leading index corresponds
        the focal individual.

    B, symbolic matrix of size m^n

        The transformed payoff matrix.

    strat_names, list of strings of length m

        Name of each strategy, in order, with indices corresponding to indices of A and B.

    strat_pairs, list of pairs

        A list of strategies to calculate the switching gains 

    Outputs:
    ---

    ltx, long string
        
        A series of latex of latex tables, one for each pair in strat_pairs
    '''

    # don't abbreviate my strings
    pd.set_option('max_colwidth',10000)

    # get the dimensions of A, which correspond to the no. individuals in the group and strategies in the game
    A_shape = A.shape
    n = len(A_shape)    # number of individuals in the group
    m = A_shape[0]      # number of strategies in the game

    # the number of partitions of n determine how many F parameters there are
    partnV = list(partitionInteger(n))
    num_partns = len(partnV)
    partn_strV = ['[' + ','.join([str(v) for v in reversed(partn)]) + ']' for partn in partnV]

    ltx_all = '' # we'll append the string here

    for strat_pair in strat_pairs:

        # get the focal- and other-strategy indices
        foc, oth = strat_pair
        k_idx, o_idx = [strat_names.index(strat_pair[i]) for i in range(2)]


        # switching gains 

        # well-mixed population
        mix_dk, _, _ = calc_switching_gains(A, k_idx, o_idx)

        # some homophily
        hom_dk, _, _ = calc_switching_gains(B, k_idx, o_idx)

        # tidy the dk expressions so the terms are grouped by F_i
        hom_dk = [sp.collect(expr, [sp.Symbol('F_'+str(i)) for i in range(1, num_partns+1)]) 
                for expr in hom_dk]

        # perfect homophily
        subsD = {'F_'+str(i): 0 for i in range(1, num_partns)} # all F_i = 0 except F_{num_partns}
        subsD['F_'+str(num_partns)] = 1
        per_dk = [expr.subs(subsD) for expr in hom_dk]


        # create latex table

        # write as latex strings
        mix_dk = [clean_latex_expr(expr) for expr in mix_dk]
        hom_dk = [clean_latex_expr(expr, partn_strV) for expr in hom_dk]
        per_dk = [clean_latex_expr(expr) for expr in per_dk]

        # create the string of signs (will need to address by hand)
        sgns = ['$+/-$' for i in range(n)]

        # the list of k values
        kV = list(range(n))

        # put into a dataframe
        columns = ['$k$', 'expression', 'sign', 'expression', 'sign', 'expression', 'sign']
        df = pd.DataFrame(list(zip(kV, mix_dk, sgns, hom_dk, sgns, per_dk, sgns)), columns=columns)

        caption = f'Switching gains $d_k$ for {foc} versus {oth} where $k$ is the number of {foc}-strategists among the $n-1$ non-focal group members.'
        label = f'switch_{foc}_{oth}'
        ltx = df.to_latex(index=False, escape=False, column_format='|c|c|c|c|c|c|c|', caption=caption, label=label)

        # tidy up latex table directly
        ltx = ltx.replace("\\begin{table}", "\\begin{table}[h]")
        ltx = ltx.replace("\\midrule", "\\hline")
        ltx = ltx.replace("\\bottomrule", "\\hline")

        # use "\toprule" as an opportunity to put in the more complex headings
        complex_heading = "\\hline \n & \\multicolumn{2}{c|}{\\bf well-mixed} & \\multicolumn{2}{c|}{\\bf some homophily} & \\multicolumn{2}{c|}{\\bf perfect homophily} \\\\ \n \\cline{2-7}"
        ltx = ltx.replace("\\toprule", complex_heading)

        # append to big string
        ltx_all += ltx
        ltx_all += '\n'

    return ltx_all
        

def clean_latex_expr(expr, partn_strV=None):
    '''
    Just a utility to turn latex expressions into the kinds of strings I'll use for
    my Latex table. Called by tabulate_switching_gains().
    '''

    expr = '$' + sp.latex(expr).replace(' ','') + '$' 

    if not partn_strV is None:

        for idx, partn_str in enumerate(partn_strV):

            oldF = 'F_{' + str(idx+1) + '}'
            newF = 'F_{' + partn_str + '}'
            expr = expr.replace(oldF, newF)

    return expr


