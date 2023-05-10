# define the model base class for the transformed payoff matrix approach

import os
import abc
import numpy as np
import itertools as it
from sympy.utilities.iterables import multiset_permutations
from copy import deepcopy
from scipy.special import binom

from family_partn_prob import get_FV_members_recruit, get_FV_members_attract, get_FV_leader_driven
from utilities import partitionInteger

class TransmatBase(metaclass=abc.ABCMeta):

    def __init__(self, evol_pars, game_pars):
        '''

        Inputs:
        ---

        evol_pars, dict

            A dictionary of the parameters needed to do evolutionary dynamics under homophilic group 
            formation.

            Must contain keys:
                'group_formation_model': str, either 'members attract', 'members recruit', or 
                    'leader driven'
                'n': int, group size
                'n_s': int, number of strategies
                'partn2prob_dir': str os.path.abspath('dir/filename'), locn of C and W matrices

            If 'group_formation_model': 'members attract', 
                must also contain: 'alpha': float (0, np.inf), stranger weighting

            If 'group_formation_model': 'members recruit' or 'leader driven',
                must also contain: 'q': float [0,1], probability to recruit/attract nonkin

            If 'group_formation_model': 'members recruit',  must also contain:
                'sumprodm_dir': str os.path.abspath('dir/filename'), where sum of products of mistakes 
                terms stored

        game_pars, dict

            A dictionary of the parameters needed to calculate payoffs 
            from a game. Write payoff() function in the subclass to use this dictionary.

            Must contain keys:
                'strat_names': list of strs, to name the strategies to distinguish their payoffs
        '''

        # extract key parameter values
        # ---

        self.evol_pars = evol_pars
        self.game_pars = game_pars
        self.n = evol_pars['n'] 
        self.n_s = evol_pars['n_s']
        self.strat_names = game_pars['strat_names']


        # create the transformed payoff matrix
        # ---

        # matrix A is untransformed, assumes no homophily
        self.A = self.create_payoff_matrix()

        # matrix B is the transformed payoff matrix

        # depends on F

        self.partnV = list(partitionInteger(self.n)) # a list of all the partitions of n
        self.F = self.calc_F() # probability distribution of family partition structures
        self.B = self.create_transformed_payoff_matrix()


    def __str__(self):

        s = 'Model of deterministic evolutionary dynamics using the transformed payoff matrix \n'
        s += 'Group formation model: ' + self.group_formation_model + '\n'

        return s


    def update_F_and_B(self):
        '''
        '''

        self.F = self.calc_F()
        self.B = self.create_transformed_payoff_matrix()


    def calc_F(self):
        '''
        Calculates the family partition structure probability distribution F. 
        '''

        self.group_formation_model = self.evol_pars['group_formation_model']

        if self.group_formation_model == 'members attract':

            alpha = self.evol_pars['alpha']
            F = get_FV_members_attract(self.n, self.partnV, alpha)

        elif self.group_formation_model == 'members recruit':

            q = self.evol_pars['q']
            sumprodm_dir = self.evol_pars['sumprodm_dir']
            fname = os.path.join(sumprodm_dir, 'sum_prod_mistakes' + str(self.n) + '.csv')
            F = get_FV_members_recruit(self.n, self.partnV, q, fname)

        elif self.group_formation_model == 'leader driven':

            q = self.evol_pars['q']
            F = get_FV_leader_driven(self.n, self.partnV, q)

        else:

            raise ValueError('Group formation model must be either: members attract, members recruit, or leader driven')

        return F


    def create_payoff_matrix(self):
        '''
        Create the payoff matrix corresponding to the payoff function calc_payoff().
        This is the untransformed payoff matrix.

        Outputs:
        ---

        A, np matrix of size m^n

            The payoff matrix where each dimension corresponds to an individual in the group, each 
            index corresponds to the strategy played by the individual, and the leading index corresponds
            the focal individual.

        '''

        # get key parameter values
        n = self.n      # Number of individuals in the group
        m = self.n_s    # Number of strategies in the game

        # A is an m^n matrix
        A = np.zeros([m]*n)

        # possible nonfocal individual's strategies
        nonf_stratsV = list(it.combinations_with_replacement(range(m), n-1))

        for foc_strat in range(m):
            for nonf_strats in nonf_stratsV:

                # calculate the payoff for this
                all_strats = [foc_strat] + list(nonf_strats)
                strat_counts = [all_strats.count(strat_idx) for strat_idx in range(m)]
                payoff = self.calc_payoff(foc_strat, strat_counts)

                # allocate this payoff to every corresponding entry of A
                for perm_nonf_strats in multiset_permutations(nonf_strats):

                    idxs = tuple([foc_strat] + perm_nonf_strats)
                    A[idxs] = payoff

        return A


    def create_transformed_payoff_matrix(self):
        '''
        Create the transformed payoff matrix that will produce the same dynamics in a well-mixed population
        as the original payoffs in an homophilic population.

        Outputs:
        ---

        B, matrix of size m^n

            The transformed payoff matrix. This will produce the same dynamics in a well-mixed population
            as the homophilic model with payoffs A and family partition structure distribution FV
        '''

        # get parameters
        A = self.A
        m = self.n_s
        n = self.n
        partnV = self.partnV
        FV = self.F

        # B is the transformed matrix of A
        B = np.zeros([m]*n)

        # possible nonfocal individual's strategies
        nonf_stratsV = list(it.combinations_with_replacement(range(m), n-1))

        for foc_strat in range(m):
            for nonf_strats in nonf_stratsV:

                # one of the indexes of B to which this refers
                idxs = tuple([foc_strat] + list(nonf_strats))

                # calculate the payoff for this
                payoff = 0

                for F, partn in zip(FV, partnV):

                    # we need to go through each possible partition size the focal could be in
                    partnS = set(partn)

                    for foc_partn_size in partnS:

                        # the probability that the focal is in a partition of this size is
                        P_foc_psize = partn.count(foc_partn_size)*foc_partn_size / n

                        # the non-focal's kin partitions remaining
                        nonf_partn = [sz for sz in partn] # deep copy
                        nonf_partn.remove(foc_partn_size)

                        # to find all possible allocations of nonfocals j, k, l, etc. into diff sized partns,
                        # we need to find all possible multiset permutations of (the zeros 
                        # are when there are none)

                        find_perms_of = nonf_partn + [0]*(n-1-len(nonf_partn))
                        nonf_allocs = list(multiset_permutations(find_perms_of))
                        P_alloc = 1/len(nonf_allocs)

                        # this loop here takes up a fair amount of time but can't be shortened
                        for nonf_alloc in nonf_allocs:

                            # this vector says how many having the strategy of i, j, k, etc.
                            whole_alloc = [foc_partn_size] + nonf_alloc

                            # the index that whole_alloc is referring to is
                            ref_idxs = tuple(idxs[i] for i in range(n) for rep in range(whole_alloc[i]))

                            # so append to B
                            payoff += F*P_foc_psize*P_alloc*A[ref_idxs]

                # allocate this payoff to every corresponding entry of B
                for perm_nonf_strats in multiset_permutations(nonf_strats):

                    idxs = tuple([foc_strat] + perm_nonf_strats)
                    B[idxs] = payoff

        return B

    def deltap_fnc(self, p, idx_omit=None):
        '''
        Using the matrix multiplication, return dp/dt for each strategy.

        Inputs:
        ---

        p, 1 x m vector of floats
            The proportion of each strategy in the population

        idx_omit, int
            Optional. Let's us know that one of the strategies is fixed at 0.

        Outputs:
        ---

        delta_p, 1 x m vector of floats
            The change in each strategy frequency in the population
        '''

        # get parameters
        # ---

        B = self.B
        n = self.n
        m = self.n_s


        # if an index of p was skipped, add a zero to p where needed
        # ---

        if not idx_omit is None:

            # need to add a zero to the p vector at index idx_omit
            p = list(p)
            p = np.array(p[:idx_omit] + [0] + p[idx_omit:])


        # calculate delta_p vector
        # ---

        # the expected payoff to each strategist
        E_piV = B
        for dim in range(n-1):

            E_piV = E_piV @ p

        # the total expected payoff
        E_pi = E_piV @ p

        # delta p_x = p_x (\overline{pi}_x - \overline{pi})
        deltap = p * (E_piV - E_pi)


        # if an index of p was skipped, skip the same in deltap
        # ---

        if not idx_omit is None:

            deltap = list(deltap)
            deltap = deltap[:idx_omit] + deltap[idx_omit+1:]

        return deltap


    def calc_stability(self, strat_ps, abs_tol=1e-10):
        '''
        Use the Jacobian matrix to determine if a given steady state is stable.

        Inputs:
        ---

        strat_ps, list of floats
            The proportion of each strategy in the population at the steady state,
            with values given in the same order as in strat_names

        Outputs:
        ---

        ans, string
            Returns 'stable' if the steady state is stable or 'unstable' if it is not.
            If the leading eigenvalue is close to 0, the stability cannot be determined
            from the linearised dynamics, and this function will return 'undetermined'
        '''

        Jac = self.calc_jacobian(strat_ps)            # get the Jacobian
        max_eig = max(np.real(np.linalg.eig(Jac)[0])) # maximum eigenvalue determines stability

        if abs(max_eig) <= abs_tol:

            # if max eig is 0, stability can't be determined from linearised system
            ans = 'undetermined'

        else:

            if max_eig < 0:
                ans = 'stable'
            else: # elif max_eig > 0:
                ans = 'unstable'

        return ans


    def calc_jacobian(self, ps):
        '''
        Return the Jacobian matrix at the steady-state defined by ps. Note that, if any 
        entry of ps is 0, it will be treated as not included in the system (you might want 
        calc_invasion_fitness() instead).

        Inputs:
        ---

        ps, list of floats
            The population proportion of each strategy at the steady state in the same order as 
            self.strat_names.


        Outputs:
        ---

        Jac, numpy matrix of floats
            The Jacobian matrix evaluated at ps.

        '''

        # get basic parameters
        # ---

        B = deepcopy(self.B) # I may need to remove rows and columns below and work with subsystem
        n = self.n
        m = self.n_s


        # if any strategies are absent, work with the subsystem
        # ---

        nzero_idxs = [ i for i in range(m) if not ps[i] == 0 ]
        new_m = len(nzero_idxs)

        assert new_m > 1, 'Only 1 strategy present at strat_ps, use calc_invasion_fitness() instead'

        if new_m < m:

            B = B[np.ix_(*[nzero_idxs]*n)]
            ps = [ps[i] for i in nzero_idxs]
            m = new_m


        # create the Jacobian
        # ---

        J = np.zeros((m-1, m-1))

        for col in range(m-1):

            derivs = [self.calc_d_Epay_dpx(B, ps, row, col) for row in range(m)]
            pi_derivs = [ps[row] * derivs[row] for row in range(m)]
            sum_pi_derivs = sum(pi_derivs)

            for row in range(m-1):

                J[row, col] = ps[row]*(derivs[row] - sum_pi_derivs)

        return J


    def calc_d_Epay_dpx(self, B, ps, u0, x):
        '''
        Calculate the derivative of the expected payoff to a u0 strategist wrt the proportion of x-strategists
        in the population

            $$ \frac{\partial \overline{\pi}_{u_0}}{\partial p_x}$$

        This calculation is called by calc_jacobian() to calculate the Jacobian matrix and ultimately the
        stability of a steady state.

        Inputs:
        ---

        B, matrix of size l^n where l <= m
            The transformed payoff matrix for only those l strategies present at the steady state.

        ps, list of floats
            The population proportion of each strategy at the steady state.

        u0, integer
            The strategy whose expected payoff we are finding the derivative

        x, integer
            The strategy whose proportion in the population the derivative is being found with respect to

        Outputs:
        ---

        deriv, float
            The value of the derivative, \frac{\partial \overline{\pi}_{u_0}}{\partial p_x}
        '''

        B_shape = B.shape
        m = B_shape[0]
        n = len(B_shape)

        deriv = 0
        for kappa in range(1, n):

            coeff = binom(n-1, kappa)*kappa

            for v_others in it.product(range(m-1), repeat=kappa-1):

                v = [u0] + list(v_others)
                beta = self.calc_beta(B, v, x)
                deriv += coeff*beta*np.prod([ps[vi] for vi in v[1:]])

        return deriv


    def calc_beta(self, B, v, x):
        '''
        Calculates beta, which is a particular sum of elements of the transformed matrix B that is
        needed to calculate the Jacobian matrix. Please see the supplementary information of the paper
        for full details, but it is basically a combinatorial problem to find this quantity.

        This calculation is called by calc_d_Epay_dpx(), which in turn is called by calc_jacobian() 
        to calculate the Jacobian matrix and ultimately the stability of a steady state.

        Inputs:
        ---

        B, matrix of size l^n where l <= m
            The transformed payoff matrix for only those l strategies present at the steady state.

        v, list of integers
            These are indices of a sum corresponding to strategies played in the group and also elements
            of the matrix B.

        x, integer
            The same x as used in calc_d_Epay_dpx(). It is the strategy whose proportion in the population 
            the derivative is being found with respect to.

        Outputs:
        ---

        beta, float
            A quantity needed to calculate elements of the Jacobian matrix
        '''

        kappa = len(v)

        B_shape = B.shape
        m = B_shape[0]
        n = len(B_shape)

        beta = 0
        for h in range(0, kappa+1):

            sign = (-1)**h

            for inner_js in it.combinations(list(range(1, kappa)) + [None], kappa-h):

                g = [v[0]] + [x if j is None else v[j] for j in inner_js] + [m-1]*(n-(kappa-h)-1)
                beta += sign*B[tuple(g)]

        return beta


    def calc_invasion_fitness(self, focal_name, strat_ps):
        '''
        Calculate the invasion fitness of a strategy given population proportions of other strategies
        at steady state.

        Inputs:
        ---

        focal_name, string
            The name of the focal strategy that we're calculating the invasion fitness of.  It 
            needs to be one of the names in self.strat_names.

        strat_ps, list of floats
            The population proportion of each strategy in the same order as self.strat_names.


        Outputs:
        ---

        invasion_fitness, float
            The invasion fitness of focal_name into the system at strat_ps.

        '''

        # get needed info and check I've got a sensible system
        # ---

        n = self.n

        # the focal strategy must be at 0 proportion to calculate its invasion fitness
        focal_idx = self.strat_names.index(focal_name)
        assert strat_ps[focal_idx] == 0

        # we'll need one of the strategies that are present to make the expected fitness comparison with
        other_idxs = [idx for idx, p_idx in enumerate(strat_ps) if p_idx > 0]
        assert len(other_idxs) > 0
        other_idx = other_idxs[0]


        # calculate the expected payoff for all strategies, including invader
        # ---

        B = self.B
        E_piV = B
        for dim in range(n-1):

            E_piV = E_piV @ strat_ps


        # calculate invasion fitness
        # ---

        # invasion fitness is difference between invader's expected payoff and another strategy's
        invasion_fitness = E_piV[focal_idx] - E_piV[other_idx]

        return invasion_fitness


        @abc.abstractmethod
        def calc_payoff(self, foc_strat, strat_counts):
            '''
            Write a function that accepts strategy distribution and returns payoff.  Ensure the game_pars 
            dictionary has all the parameters it needs.
            

            Inputs:
            ---

            focal_strat, int
                The strategy played by the focal player, where the value correspond to index of strat_names

            strat_counts, list of ints
                Whole-group distribution of strategies, including the focal, in the same order as strat_names

            Outputs:
            ---

            payoff_total, float
                Payoff to the focal strategist given the whole-group distribution
            '''
