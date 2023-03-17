# the purpose of this code is to check the code using examples numerically, 
# that the probability of choosing a given rho from a give psi calculated 
# using the code is equal to the proportion of samples that satisfy rho drawn
# randomly from psi

import numpy as np
import random
from get_allocations import get_allocations

# some of the examples I checked
# ---

# uncomment or write your own to check
'''
rhos = [2, 2, 2, 2]
psis = [4, 4] # -- correct

rhos = [2, 2]
psis = [4, 1] # -- correct

rhos = [2, 2, 3]
psis = [4, 3, 1] # -- correct

rhos = [2, 2, 3]
psis = [5, 3, 1] # -- correct

rhos = [2, 2, 3]
psis = [5, 2, 2, 1] # -- correct

'''

rhos = [2, 2, 3]
psis = [1, 3, 5] # -- correct


# parameters
# ---

# how many times to sample to calculate the probability numerically
num_samples = 100000

# secondary parameter, number of individuals in the group
n = sum(psis)


# calculation according to my equation using my code
# ---

# get the allocations dictionary
# a list dictionaries, where each dictionary represents one way of allocating the rhos into partitions
# each key is the index of the partition in psis, and the value is the sum of rho values allocated to that partition in psi
psiDV = get_allocations(rhos, psis)

# calculate the numerator to the probability of each allocation occurring

numers = 0
for psiD in psiDV:

    # for each possible allocation, calculate the probability's numerator

    numer = 1
    for psi_idx, rhosum in psiD.items():

        psi = psis[psi_idx]
        numer *= np.prod(range(psi+1-rhosum, psi+1)) # the numerator in our equation

    # the total probability is the sum of probabilities
    # of all possible allocations
    numers += numer

# the probability's denominator is always the same
denom = np.prod(range(n+1-sum(rhos), n+1))

probability = numers/denom
print(f'According to my maths and code, the probability is {probability}')


# numerical calculation by randomly drawing from a group with partition structure psi
# ---

# create a group with family IDs matching partition structure psi
group = [i for i, psi in enumerate(psis) for individuals in range(psi)]
# e.g., psis = [1, 3, 5] gives group [0, 1, 1, 1, 2, 2, 2, 2, 2]

# rewrite r_sub as the start and end indices
len_rhos = len(rhos)
r_inds = [sum(rhos[:i]) for i in range(len_rhos+1)]
# e.g., rhos = [2, 2, 3] gives r_inds = [0, 2, 4, 7]

# count how many random draws satisfy the structure of rho
cnt_satisfied = 0
for i in range(num_samples):

    # shuffle our group of family members
    random.shuffle(group)

    # initialise by saying this random draw satisfied the structure of rho ...
    rhos_satisfied = True

    # ... and check each rho_i drawn for falsification

    for i in range(len_rhos):

        # start and end index for this rho_i
        start = r_inds[i]
        end = r_inds[i+1]

        # check if any of our rho_i are not all from the same family ID
        if any(v != group[start] for v in group[start+1:end]):
            
            rhos_satisfied = False
            break

    # if the sample we drew satisfied the structure of rho, add to count
    if rhos_satisfied:
        cnt_satisfied += 1

# the probability calculated numerically is the proportion of our samples satisfying rho
print(f'I find numerically, the proportion satisfying rho is {cnt_satisfied/num_samples}')
