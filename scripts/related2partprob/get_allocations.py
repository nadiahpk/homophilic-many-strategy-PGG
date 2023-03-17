def get_allocations(rhos, psis):
    '''

    Inputs:
    ---

    rhos, list of ints
        A list of the integers that are in the subscript of Hisashi's r, we call rho

    psis, list of ints
        The family partition structure of the group, 
        each integer is the number of group members in that family


    Outputs:
    ---

    psiDV, list of dictionaries
        Each dictionary represents a way of allocating the rhos into partitions.
        The key is the index of the partition in psis, and the value is the sum of rho values.

    '''

    # find the possible ways to allocate the rhos into psis
    # ---

    allocations = [[]]

    for rho in rhos:

        # it is a cartesian product with the restriction that,
        # for the set of rhos allocated to a family partition,
        # the sum of rhos must be <= the size of that partition

        allocations = [alloc_sofar+[j] for alloc_sofar in allocations for j, psi in enumerate(psis)
                if rho <= psi-sum(rhos[i_rho] for i_rho, i_psi in enumerate(alloc_sofar) if i_psi == j)]

    # e.g., allocations = [ [0, 0], [0, 1] ] means
    # allocation 0 is [0, 0], and it has rho_0 in psi_0 and rho_1 in psi_0
    # allocation 1 is [0, 1], and it has rho_0 in psi_0 and rho_1 in psi_1

    # this way of storing the allocations is equivalent what is called \boldsymbol{a} in the SI


    # turn the allocations into a dictionary {psi_idx: sum_of_rhos_allocated}
    # ---

    # this way of storing the allocations is more useful for calculating the probabilities 
    # i.e., we calculate the sum of \rho values for each \psi_i

    psiDV = list()
    for allocation in allocations:

        # the psi indices to which the allocations are made
        psiD = {psi_idx: 0 for psi_idx in set(allocation)}

        # finds the sum rhos that go into each partition
        for psi_idx, rho in zip(allocation, rhos):
            psiD[psi_idx] += rho

        # append the dictionary for this allocation
        psiDV.append(psiD)

    return psiDV
