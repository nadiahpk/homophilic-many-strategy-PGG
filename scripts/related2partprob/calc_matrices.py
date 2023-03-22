# coefficients needed to calculate relatedness coefficients from partition probabilities and visa versa
# calculate and store matrices as csv files

import numpy as np
import pandas as pd
from math import gcd


import sys
sys.path.append('../../functions/')

from get_allocations import get_allocations
from utilities import partitionInteger


# parameters
# ---

# where to store the coefficients
results_dir = '../../results/related2partprob/'

# list of group sizes we wish to calculate for
nV = [3, 4, 5, 6, 7, 8]


# coefficients for converting between r_rho and F_psi
# ---

for n in nV:

    # get a list of psis, the family partition structures
    # ---

    psisV = list(partitionInteger(n))
    # -- e.g., n=3, psisV = [[1, 1, 1], [1, 2], [3]]

    # skip the first one
    psisV = psisV[1:]


    # get a list of rhos
    # ---

    # the rhos are all partitions in psisV above minus the elements equal to 1
    rhosV = [[psi for psi in psis if psi != 1] for psis in psisV]
    # -- e.g., n=3, rhosV = [[2], [3]]

    '''
    # another way to generate rhos, just for interest

    # the rhos are all partitions of integers 2 ... n
    # that don't have a 1 in them

    rhosV = list()
    for i in range(2, n):

        # all partitions
        partns = list(partitionInteger(i))

        # filter out those with ones in them
        for partn in partns:
            if 1 not in partn:
                rhosV.append(partn)

    # we already have the nth one
    for partn in psisV:
        if 1 not in partn:
            rhosV.append(partn)

    # -- rhosV = [[2], [3]]
    '''


    # create a big matrix to store
    # ---

    num_psis = len(psisV)
    num_rhos = len(rhosV)

    # this will store the probability numerator
    M = np.zeros((num_rhos, num_psis), dtype=int)

    # this will store the probability denominator
    # D = np.zeros(num_rhos, dtype=int)
    # skipped bc the denominator is always the same for each rho
    #   denoms = np.prod(range(n+1-sum(rhos), n+1))
    # and will also be equal to the final column of M
    # corresponding to the partition {n}

    for row, rhos in enumerate(rhosV):

        for col, psis in enumerate(psisV):

            # get the allocations dictionary
            psiDV = get_allocations(rhos, psis)

            # Each dictionary represents a way of allocating the rhos into partitions.
            # The key is the index of the partition in psis, and the value is the sum of rho values.

            # calculate the numerator to the probability of each allocation occurring

            numers = 0
            for psiD in psiDV:

                # calculate this numerator
                numer = 1
                for psi_idx, rhosum in psiD.items():

                    psi = psis[psi_idx]
                    numer *= np.prod(range(psi+1-rhosum, psi+1))

                # add it to the others
                numers += numer

            # put it where it belongs
            M[row, col] = numers


    # create the matrix of actual probabilities and invert it
    # ---

    # the denominators
    D = [np.prod(range(n+1-sum(rhos), n+1)) for rhos in rhosV]

    # matrix of probabilities
    P = M / np.tile(D, (num_psis, 1)).transpose()

    invP = np.linalg.inv(P)

    # the inverse will be all whole numbers
    # coefficients to convert in the reverse direction, from r_rho to F_psi
    invP_i = invP.astype(int)

    # the greatest common divisor also appears as the leftmost nonzero term
    hcfs = [gcd(*V) for V in invP_i]

    # lets factor it out
    invP_factored = invP_i // np.tile(hcfs, (num_rhos, 1)).transpose()


    # write files
    # ---

    # create headers
    header_psis = ['|'.join([str(v) for v in psis]) for psis in psisV]
    header_rhos = ['|'.join([str(v) for v in rhos]) for rhos in rhosV]

    # write coefficients for rho to psi conversion file
    df_psi = pd.DataFrame(header_psis, columns = ['psi'])
    df_hcf = pd.DataFrame(hcfs, columns = ['common_factor'])
    df_bod = pd.DataFrame(invP_factored, columns = header_rhos)
    df_final = pd.concat([df_psi, df_hcf, df_bod], axis=1)
    fname = results_dir + 'rho2psi_coeffs_n' + str(n) + '.csv'
    df_final.to_csv(fname, index=False)

    # write numerators matrix M to a file
    df_rho = pd.DataFrame(header_rhos, columns = ['rho'])   # the first column is the rho values
    df_fac = pd.DataFrame(D, columns = ['common_denom'])
    df_bod = pd.DataFrame(M, columns = header_psis)         # body
    df_final = pd.concat([df_rho, df_fac, df_bod], axis=1)  # put them together
    fname = results_dir + 'psi2rho_numerators_n' + str(n) + '.csv'
    df_final.to_csv(fname, index=False)

