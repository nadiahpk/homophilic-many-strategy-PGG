# Functions used to calculate the family partition probabilities 
# resulting from 3 group-formation models: leader drive, members 
# recruit, and members attract. For details on the 3 models, please
# refer to 
#
#   Kristensen, N.P., Ohtsuki, H., Chisholm, R.A. (2022) Ancestral 
#   social environments plus nonlinear benefits can explain 
#   cooperation in human societies, Scientific Reports, 12: 20252, 
#   https://doi.org/10.1038/s41598-022-24590-y

import pandas as pd
import numpy as np
from math import factorial
from scipy.special import binom

def get_FV_members_attract(n, partns, alpha):
    '''
    Using the 'members attract' homophilic group-formation model, 
    calculate the vector of partition probabilities F.


    Inputs:
    ---

    n, int
        Group size

    partns, list of list of ints
        List of all possible partitions of n

    alpha, float (0, infty)
        The 'stranger weighting', the weighting given to recruitment 
        of a nonkin new member.


    Outputs:
    ---

    FV, list of floats
        The probability that a group is formed with family partition 
        structure corresponding to structures in partns
    '''


    # place to store probabilities of each partition
    FV = [0]*len(partns)

    if alpha == 0:

        # all individuals will be in the same partition
        FV = [ 1 if len(partn) == 1 else 0 for partn in partns ]

    elif np.isinf(alpha):

        # all individuals will be in separate partitions
        FV = [ 1 if len(partn) == n else 0 for partn in partns ]

    else:

        # we'll need to calculate using Ewen's formula

        fact_n = factorial(n)
        for i, partn in enumerate(partns):

            Phi = len(partn)

            # phi_i i s the number of species with i individuals
            # create a list of non-zero (i, phi_i) pairs

            iV = set(partn) # which i's occur
            i_phiV = [ (i, partn.count(i)) for i in iV ]

            # get terms in denom
            AA = np.prod([ i**phi_i for i, phi_i in i_phiV ])
            BB = np.prod([ factorial(phi_i) for _, phi_i in i_phiV ])
            CC = np.prod([ alpha + k - 1 for k in range(1, n+1) ])

            # calc F
            #FV[i] = fact_n * alpha**Phi / (AA*BB*CC)
            FV[i] = fact_n * alpha**Phi / AA / BB / CC # rewritten to prevent overflow errors

    return FV


def get_FV_members_recruit(n, partns, q, fname):
    '''
    Using the 'members recruit' homophilic group-formation model, 
    calculate the vector of partition probabilities F.

    Inputs:
    ---

    n, int
        Group size

    partns, list of list of ints
        List of all possible partitions of n

    q, float (0,1)
        The probability that the recruitor makes a mistake and 
        recruits a nonkin new member


    Outputs:
    ---

    FV, list of floats
        The probability that a group is formed with family partition 
        structure corresponding to structures in partns
    '''


    if q == 0:

        # special case: never mistakenly choose a stranger
        FV = [ 1 if partn == [n] else 0 for partn in partns ]

    elif q == 1: 

        # special case: always choose a stranger
        FV = [ 1 if partn == [1]*n else 0 for partn in partns ]

    else:

        # read in and prepare the needed info
        # ---

        df = pd.read_csv(fname)
        df.set_index('partition', inplace=True)


        # for each partition, calculate the probability
        # ---

        FV = list()
        for partn in partns:

            # get sum_prod_mistakes and other info

            Phi = len(partn) # the total number of families
            partn_str = '|'.join([ str(partn_ij) for partn_ij in partn]) 
            sum_prod_mistakes = df.loc[partn_str]['sum_product_mistake_indices']

            # make calculation

            F = (np.prod([factorial(phi-1) for phi in partn]) / factorial(n-1)) * q**(Phi-1) * (1-q)**(n-Phi) * sum_prod_mistakes
            FV.append(F)

    return FV


def get_FV_leader_driven(n, partns, q):
    '''
    Using the 'leader driven' homophilic group-formation model, 
    calculate the vector of partition probabilities F.


    Inputs:
    ---

    n, int
        Group size

    partns, list of list of ints
        List of all possible partitions of n

    q, float (0,1)
        The probability that the recruitor makes a mistake and 
        recruits a nonkin new member


    Outputs:
    ---

    FV, list of floats
        The probability that a group is formed with family partition 
        structure corresponding to structures in partns
    '''

    FV = list()
    for partn in partns:

        if len(partn) == 1:

            Fi = (1-q)**(n-1)

        else:

            # rely on the largest group being at the end in code below
            partn.sort()

            if partn[-2] != 1:

                # this cannot happen in the leader model because all strangers are unrelated
                Fi = 0

            else:

                s = len(partn)-1
                Fi = binom(n-1, s) * q**s * (1-q)**(n-1-s)

        FV.append(Fi)

    return FV 

