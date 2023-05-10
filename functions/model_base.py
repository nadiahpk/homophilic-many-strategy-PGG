import abc
import pandas as pd
import os
import itertools as it
import numpy as np

from family_partn_prob import get_FV_members_recruit, get_FV_members_attract, get_FV_leader_driven

class ModelBase(metaclass=abc.ABCMeta):

    def __init__(self, evol_pars, game_pars, calc_minus1_system = False, precalc_payoffs = False):
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

        cacl_minus1_system, boolean

            This is particularly used when m = 4 for plotting the strategy dynamics on the face of 
            the tetrahedron, to save time. If true, we pre-calculate parameters needed for a reduced 
            m -> m-1 system.

        precalc_payoffs, boolean

            Switch to True to precalculate payoffs for all possible focal players in all possible
            strategy outcomes (may save time for expensive payoff calculations)
        '''

        self.evol_pars = evol_pars
        self.game_pars = game_pars
        self.n = evol_pars['n']     # the evolutionary parameters, 
        self.n_s = evol_pars['n_s'] # which are indept of game type
        self.strat_names = game_pars['strat_names']


        # get the stored variables C and W matrices to convert F -> Z
        # ---

        partn2prob_dir = evol_pars['partn2prob_dir']
        fname = os.path.join(partn2prob_dir, 'CW_groupsize' + str(self.n) + '_numstrategies' + str(self.n_s) + '.csv')

        self.CM, self.WM, comp_strat_countsV, self.partnV = self.get_CM_WM(fname)

        # Info above is stored in a compressed form that treats non-focal strategy tags as arbitrary.
        # But when we do the calculations, we need to distinguish between strategies.
        # Therefore, we make the full list of strategy outcomes, and put the mapping between them 
        # and the compressed list in look-up dictionaries

        self.full_strat_countsV, self.full_row2comp_row, self.full_row2order = \
            self.get_full_list_strategy_outcomes(self.n, self.n_s, comp_strat_countsV)

        # for the minus-1-strategy system (for calculating faces)
        if calc_minus1_system:

            self.get_minus1_pars()

        else:

            self.CM_m = None; self.WM_m = None; comp_strat_countsV_m = None
            self.full_strat_countsV_m = None; self.full_row2comp_row_m = None; self.full_row2order_m = None


        # calculate the partition probabilies F
        # ---

        self.update_F()


        # precalculate payoffs dictionary (useful for expensive payoff calculations)
        # ---

        # for a list of names stratV and a list of counts strat_counts, use this to get payoff out of dictionary:
        #  payoff = payoffD[self.strat_names.index(stratV[0])]
        #                  [tuple([strat_counts[stratV.index(x)] if x in stratV else 0 for x in self.strat_names])]

        self.payoffD = dict()
        if precalc_payoffs:
            self.payoffD = self.calc_payoffD()


    def __str__(self):

        s = 'Model of deterministic evolutionary dynamics \n'
        s += 'Group formation model: ' + self.group_formation_model + '\n'

        return s


    def get_minus1_pars(self):

        fname = os.path.join(self.evol_pars['partn2prob_dir'], 
                'CW_groupsize' + str(self.n) + '_numstrategies' + str(self.n_s-1) + '.csv')

        self.CM_m, self.WM_m, comp_strat_countsV_m, _ = self.get_CM_WM(fname)

        self.full_strat_countsV_m, self.full_row2comp_row_m, self.full_row2order_m = \
                self.get_full_list_strategy_outcomes(self.n, self.n_s-1, comp_strat_countsV_m)


    def calc_payoffD(self):

        # a list of all nonfocal strategy distributions
        nonfoc_distns = [[outcome.count(s) for s in range(self.n_s)] 
                for outcome in it.combinations_with_replacement(range(self.n_s), self.n-1)]

        # a place to store payoffs, a dictionary structured as:
        # [focal's strategy index][whole group distribution]: payoff to focal
        payoffD = {focal_idx: dict() for focal_idx in range(self.n_s)}

        # calculate payoff for each focal strategy and for each non-focal strategy distribution
        # ---

        for focal_idx, foc_strat_name in enumerate(self.strat_names):

            # payoff() function needs the focal's strategy as the first strategy name in the list
            strat_names = [self.strat_names[focal_idx]] + self.strat_names[:focal_idx] + self.strat_names[focal_idx+1:]

            for nonfoc_distn in nonfoc_distns:

                # whole-group strategy distribution
                whole_distns = [v+1 if idx == focal_idx else v for idx, v in enumerate(nonfoc_distn)]

                # payoff() function needs the focal's count as the first count in the list
                strat_counts = [whole_distns[focal_idx]] + whole_distns[:focal_idx] + whole_distns[focal_idx+1:]


                # find and store payoff
                payoffD[focal_idx][tuple(whole_distns)] = self.payoff(strat_names, strat_counts)

        return payoffD


    def clear_payoffD(self):

        self.payoffD = dict()


    def update_F(self):

        # calculate the partition probabilies F
        # can update F with a change in homophily without touching anything else
        # ---

        self.group_formation_model = self.evol_pars['group_formation_model']

        if self.group_formation_model == 'members attract':

            alpha = self.evol_pars['alpha']
            self.F = get_FV_members_attract(self.n, self.partnV, alpha)

        elif self.group_formation_model == 'members recruit':

            q = self.evol_pars['q']
            sumprodm_dir = self.evol_pars['sumprodm_dir']
            fname = os.path.join(sumprodm_dir, 'sum_prod_mistakes' + str(self.n) + '.csv')
            self.F = get_FV_members_recruit(self.n, self.partnV, q, fname)

        elif self.group_formation_model == 'leader driven':

            q = self.evol_pars['q']
            self.F = get_FV_leader_driven(self.n, self.partnV, q)

        else:

            raise ValueError('Group formation model must be either: members attract, members recruit, or leader driven')


    def get_CM_WM(self, fname):
        '''
        Using the file, reconstruct the C and W matrices used to construct matrix P, and also return 
        the strategy outcomes corresponding to rows, and family partition structures corresponding 
        to columns.  The rows of C and W correspond to the strategy outcomes and the columns to 
        family partitions.

        Inputs:
        ---

        fname, str
            The location where the CW csv file is stored.  
            See examples in /partn2prob_compress/results/.

        Outputs:
        ---

        CM, list of list of tuples
            The coefficients C used to construct the P matrix compressed so permutations of 
            non-focal (idx > 0) are not considered.

        WM, list of list of tuples of tuples
            The powers of p used to construct the P matrix compressed so permutations of non-focal 
            (idx > 0) are not considered.

        strat_countsV, list of tuples
            A list of the strategy outcomes for a group corresponding to rows of CM and WM. For 
            example, (1, 0, 0, 4) means a group containing 1 individual with the focal-strategy, 
            0 with another strategy, 0 with another strategy, and 4 with another strategy.  If the 
            strategies are labelled A, B, C, D, then it is also maps to [1, 0, 4, 0] 
            (i.e., 1 Strategy A, 0 Strategy B, 4 Strategy C, 0 Strategy D) and [1, 4, 0, 0].  Use 
            get_full_list_strategy_outcomes() to deal with this transformation.
            
        partnsV, list of tuples
            A list of all possible family-partitions corresponding to cols of CM and WM.

        '''


        # read in the dataframe
        # ---

        # read the whole thing as strings
        df = pd.read_csv(fname, dtype='string')

        # replace all the NA with an empty string
        df.fillna('', inplace=True)

        # convert all the no_strat_x columns to integers
        no_strat_headers = [s for s in df.columns if s[:8] == 'no_strat']
        df[no_strat_headers] = df[no_strat_headers].astype(int)


        # get the group size and number of strategies info
        # ---

        # n = max(df['no_strat_1'])
        n_s = len(no_strat_headers)


        # recreate WM and CM
        # ---

        # the strategy-count outcomes
        strat_countsV = df[no_strat_headers].values
        strat_countsV = [tuple(v) for v in strat_countsV]
        # -- e.g., [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]

        # the family partitions
        partn_strV = [s[5:] for s in df.columns if s[:4] == 'coef']
        # -- e.g., ['1|1|1|1', '1|1|2', '1|3', '2|2', '4']

        partnsV = [list( int(v) for v in s.split('|') ) for s in partn_strV]
        # -- e.g., [[1, 1, 1, 1], [1, 1, 2], [1, 3], [2, 2], [4,]]

        no_rows = len(strat_countsV)
        no_cols = len(partnsV)


        # coefficients matrix C

        CM = [[(0,) for col in range(no_cols)] for row in range(no_rows)]

        for row in range(no_rows):
            for col, partn_str in enumerate(partn_strV):

                coef_str = df.iloc[row]['coef_' + partn_str]

                if coef_str:

                    CM[row][col] = tuple(int(v) for v in coef_str.split('|'))


        # powers matrix W

        WM = [[(tuple([0]*n_s),) for col in range(no_cols)] for row in range(no_rows)]

        for row, strat_counts in enumerate(strat_countsV):

            idxs = [i for i, cnt in enumerate(strat_counts) if cnt > 0]
            for col, partn_str in enumerate(partn_strV):

                pwrs_str = df.iloc[row]['pwrs_' + partn_str]

                if pwrs_str:

                    WMi = list()
                    for s in pwrs_str.split('*'):

                        WMij = [0]*n_s

                        for idx, v in zip(idxs, s.split('|')):

                            WMij[idx] = int(v)

                        WMi.append(tuple(WMij))

                    WM[row][col] = tuple(WMi)

        return(CM, WM, strat_countsV, partnsV)


    def get_full_list_strategy_outcomes(self, n, n_s, comp_strat_countsV=None):
        '''
        Get a list of all possible strategy outcomes where different orderings of non-focal strategies
        are counted. If the compressed list is provided, also return a mapping from the full to
        compressed list.

        Inputs:
        ---

        n, integer
            Group size

        n_s, integer
            Number of strategies

        comp_strat_countsV, list of tuples, optional
            A compressed list of the possible strategy outcomes where the non-focal strategies
            (index > 0) have arbitrary indices, i.e., non-focal combinations not included.
            e.g., [1, 0, 0, 5] means '1 individual of focal strategy, 0 of another strategy, 0 of another strategy, 
            5 of another strategy. If the strategies are labelled A, B, C, D, then it is also maps to
            [1, 0, 5, 0] (1 Strategy A, 0 Strategy B, 5 Strategy C, 0 Strategy D) and [1, 5, 0, 0].

        Outputs:
        ---

        full_strat_countsV, list of tuples
            A full list of the possible strategy outcomes where all combinations of the non-focal strategies
            (index > 0) are included. e.g., [1, 0, 0, 5] means only 1 Strategy A (focal), 0 Strategy B, 0 Strategy C, 
            5 Strategy D.

        full_row2comp_row, dict int -> int
            Maps from rows of full_strat_countsV to equivalent entry in comp_strat_countsV

        full_row2order, dict int -> tuple of ints
            Maps from rows of full_strat_countsV to the order of the equivalent entry in comp_strat_countsV
        '''

        # a full list of strategy outcomes that distinguishes non-focal tags 
        full_strat_countsV = [tuple( outcome.count(s) for s in range(n_s) )
                for outcome in it.combinations_with_replacement(range(n_s), n)]

        # remove the strat_counts that start with a zero because I am assuming the first strategy is the focal
        full_strat_countsV = [strat_counts for strat_counts in full_strat_countsV if strat_counts[0] != 0]

        if comp_strat_countsV is None:

            full_row2comp_row = None
            full_row2order = None

        else:

            # dictionaries mapping from the full list of strategy outcomes to the compressed list

            full_row2comp_row = dict()
            full_row2order = dict()

            for full_row, full_strat_counts in enumerate(full_strat_countsV):

                comp_strat_counts = (full_strat_counts[0],) + tuple(sorted( full_strat_counts[1:] ))
                comp_row = comp_strat_countsV.index(comp_strat_counts)      # which row of compressed strat_countsV
                full_row2comp_row[full_row] = comp_row

                order = (0,) + tuple(np.argsort(full_strat_counts[1:])+1)   # order strategies appear in comp_strat_countsV
                full_row2order[full_row] = order

        return(full_strat_countsV, full_row2comp_row, full_row2order)


    def deltap_fnc(self, strat_ps_, idx_omit=None):
        '''
        Given a vector of strategy proportions in the population, calculate the change in frequency 
        of each strategy.

        Inputs:
        ---

        strat_ps_, list of floats
            The population proportion of each strategy in the same order as self.strat_names

        idx_omit, int, optional
            The index in self.strat_names of a strategy proportion omitted from strat_ps_ implying 
            that it should have its population held at 0. Useful for plotting dynamics on faces.

        Outputs:
        ---

        deltapV, list of floats
            The relative change in proportion of each strategy.
        '''

        strat_ps = list(strat_ps_)

        if idx_omit is None:

            # the default where strat_ps_ has all strategies
            strat_names = self.strat_names
            full_strat_countsV = self.full_strat_countsV

        else:

            if self.CM_m is None:
                self.get_minus1_pars()

            # special case where strat_ps_ omits strategy idx_omit (i.e., idx_omit's population held at 0)
            strat_names = self.strat_names[:idx_omit] + self.strat_names[idx_omit+1:]
            full_strat_countsV = self.full_strat_countsV_m # minus-1 system marked with _m in superclass

        n_s = len(strat_names) # if idx_omit, this equals self.n_s-1 


        # calculate and store results for each focal index (rows of *M matrices below)
        # ---

        # name ordering of strategies (focal is always index 0) 
        stratM = [[strat_names[focal_idx]] + strat_names[:focal_idx] + strat_names[focal_idx+1:] 
                     for focal_idx in range(n_s)]

        # population proportions of strategies
        pM = [[strat_ps[focal_idx]] + strat_ps[:focal_idx] + strat_ps[focal_idx+1:] for focal_idx in range(n_s)]

        # probability of each possible strategy-count outcome in full_strat_countsV.
        ZM = list()
        for pV in pM:
            P = self.calc_P(pV, idx_omit)
            ZV = list(P @ np.array(self.F))
            ZM.append(ZV)


        # calculate the rate of change of each focal strategy
        # ---

        # get payoffs
        if self.payoffD:

            # retrive precalculated payoffs from dictionar
            payoffM = [[self.payoffD[self.strat_names.index(stratV[0])][tuple([strat_counts[stratV.index(x)] if x in stratV else 0 for x in self.strat_names])] for strat_counts in full_strat_countsV] for stratV in stratM]

        else:

            # calculate now
            payoffM = [[self.payoff(stratV, strat_counts) for strat_counts in full_strat_countsV] for stratV in stratM]


        # \pi_i(k) Z(s_i, k) terms 
        pi_ZV = [ sum( payoff* Z for payoff, Z in zip(payoffV, ZV) ) for payoffV, ZV in zip(payoffM, ZM) ]

        # second part of the deltap equation is always the same, \sum_{i=1}^{n_s} \pi_i(k) Z(s_i, k)
        second_bit = sum(pi_ZV)

        # calculate this delta p according to the main equation
        deltapV = [ pi_ZV[focal_idx] - strat_ps[focal_idx]*second_bit for focal_idx in range(n_s) ]

        return(deltapV)


    def calc_P(self, pV, idx_omit=None):
        '''
        The P matrix that's used to calculate the probability of each strategy outcome, i.e., Z = PF.  
        This function calculates P using the compressed C and W matrices (coefficients and powers).

        Inputs:
        ---

        pV, list of floats
            Population proportions for each strategy, where order corresponds to strategy outcomes 
            in full_strat_countsV (below)

        idx_omit, int, optional
            The index in self.strat_names of a strategy proportion omitted from strat_ps_ implying 
            that it should have its population held at 0. Useful for plotting dynamics on faces.

        Outputs:
        ---

        P, np.array of floats
            The matrix that converts family partition probabilies (F) into strategy outcome probabilities 
            (Z)
        '''

        if idx_omit == None:

            # use the full system 
            CM = self.CM
            WM = self.WM
            full_strat_countsV = self.full_strat_countsV
            full_row2comp_row = self.full_row2comp_row
            full_row2order = self.full_row2order

        else:

            if self.CM_m is None:
                get_minus1_pars()

            # use the minus-1-strategy system, which is marked with _m suffix
            CM = self.CM_m
            WM = self.WM_m
            full_strat_countsV = self.full_strat_countsV_m
            full_row2comp_row = self.full_row2comp_row_m
            full_row2order = self.full_row2order_m

        # group size
        n = sum(full_strat_countsV[0])

        # size of the P matrix
        rows = len(full_strat_countsV)  # number of strategy outcomes including permuations of non-focal strategy
        cols = len(CM[0])               # number of family partitions

        P = np.zeros((rows, cols))
        for row in range(rows):

            # match the order of strategy proportions to the order in compressed C and W
            ps = [pV[i] for i in full_row2order[row]]

            # get the row of the compressed C and W corresponding to this row of the full strategy-outcomes list
            CM_row = CM[full_row2comp_row[row]]
            WM_row = WM[full_row2comp_row[row]]

            for col in range(cols):

                # the equation that constructs P from C and W while accounting for compression
                P[row, col] = sum( (coeff/n)*np.prod([p**pwr for p, pwr in zip(ps, pwrs)])
                                  for coeff, pwrs in zip(CM_row[col], WM_row[col]) )

        return(P)


    # for compatibility with old code
    def inv_fit(self, focal_name, strat_ps):
        return self.calc_invasion_fitness(focal_name, strat_ps)


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

        strat_ps = list(strat_ps)

        strat_names = self.strat_names
        n_s = len(strat_names)

        focal_idx = strat_names.index(focal_name)
        assert strat_ps[focal_idx] == 0 # must be evaluated at 0 for invasion


        # calculate the second part of the invasion fitness equation
        # ---

        full_strat_countsV = self.full_strat_countsV

        # name ordering of strategies (this is always index 0) 
        stratM = [[strat_names[this_idx]] + strat_names[:this_idx] + strat_names[this_idx+1:] 
                     for this_idx in range(n_s)]

        # population proportions of strategies
        pM = [[strat_ps[this_idx]] + strat_ps[:this_idx] + strat_ps[this_idx+1:] for this_idx in range(n_s)]

        # probability of each possible strategy-count outcome in full_strat_countsV.
        ZM = list()
        for pV in pM:
            P = self.calc_P(pV, idx_omit=None)
            ZV = list(P @ np.array(self.F))
            ZM.append(ZV)

        # payoffs
        if self.payoffD:

            # retrieve precalculated payoffs from dictionary
            payoffM = [[self.payoffD[self.strat_names.index(stratV[0])][tuple([strat_counts[stratV.index(x)] if x in stratV else 0 for x in self.strat_names])] for strat_counts in full_strat_countsV] for stratV in stratM]

        else:

            # calculate
            payoffM = [[self.payoff(stratV, strat_counts) for strat_counts in full_strat_countsV] for stratV in stratM]

        # calculate second part of the invasion fitness equation
        second_bit = sum( sum( payoff * Z 
            for payoff, Z in zip(payoffV, ZV) ) 
            for payoffV, ZV in zip(payoffM, ZM) )

        # NOTE: this could be shortened by observing that the expected payoff for each strategy 
        # is equal to the expected payoff at the steady state. Unecessary to take the average. TODO

        # calculate the P/p_focal
        # ----

        # arrange p so that the focal idx is first
        pV = [strat_ps[focal_idx]] + strat_ps[:focal_idx] + strat_ps[focal_idx+1:]

        # get the needed variables
        n = sum(full_strat_countsV[0])
        CM = self.CM
        WM = self.WM
        full_strat_countsV = self.full_strat_countsV
        full_row2comp_row = self.full_row2comp_row
        full_row2order = self.full_row2order

        # size of the P matrix
        rows = len(full_strat_countsV)  # number of strategy outcomes including permuations of non-focal strategy
        cols = len(CM[0])               # number of family partitions

        # create a modified P i.e., P/p_focal

        P = np.zeros((rows, cols))
        for row in range(rows):

            # match the order of strategy proportions to the order in compressed C and W
            ps = [pV[i] for i in full_row2order[row]]

            # get the row of the compressed C and W corresponding to this row of the full strategy-outcomes list
            CM_row = CM[full_row2comp_row[row]]
            WM_row = WM[full_row2comp_row[row]]

            for col in range(cols):

                # here, I have to turn all p[0]^1 -> 1, and all others into 0
                P[row, col] = sum( (coeff/n)*np.prod([p**pwr if i != 0 else 1 if pwr == 1 else 0 
                                    for i, (p, pwr) in enumerate(zip(ps, pwrs))])
                                  for coeff, pwrs in zip(CM_row[col], WM_row[col]) )


        # use P to calculate first bit of invasion fitness
        # ---

        ZV = list(P @ np.array(self.F))

        # get payoffs
        if self.payoffD:

            stratV = stratM[focal_idx]
            payoffV = [self.payoffD[focal_idx][tuple([strat_counts[stratV.index(x)] if x in stratV else 0 for x in self.strat_names])] for strat_counts in full_strat_countsV]

        else:

            # calculate payoffs
            payoffV = [self.payoff(stratM[focal_idx], strat_counts) for strat_counts in full_strat_countsV]

        # first part of the invasion fitness equation
        first_bit = sum( payoff*Z for payoff, Z in zip(payoffV, ZV) )


        # calculate invasion fitness
        # ---

        invasion_fitness = first_bit - second_bit

        return invasion_fitness

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

        Jac = self.calc_jacobian(strat_ps)              # get the Jacobian
        max_eig = max(np.real(np.linalg.eig(Jac)[0]))   # maximum eigenvalue determines stability

        if abs(max_eig) <= abs_tol:

            # if max eig is 0, stabiltiy can't be determined from linearised system
            ans = 'undetermined'

        else:

            if max_eig < 0:
                ans = 'stable'
            else: # elif max_eig > 0:
                ans = 'unstable'

        return ans



    def calc_jacobian(self, strat_ps):
        '''
        Return the Jacobian matrix at the steady-state defined by strat_ps. Note that, if any 
        entry of strat_ps is 0, it will be treated as not included in the system (you might want 
        calc_invasion_fitness() instead).

        Inputs:
        ---

        strat_ps_, list of floats
            The population proportion of each strategy at the steady state in the same order as 
            self.strat_names.


        Outputs:
        ---

        Jac, numpy matrix of floats
            The Jacobian matrix evaluated at strat_ps.

        '''

        strat_names = self.strat_names
        n_s = len(strat_names)


        # create ZM, matrix of Z(s_i, k) where i is strategy and k is strategy-count outcome
        # ---

        # the index reorderings depending on the identity of the focal strategy
        strat_idxs = list(range(n_s))
        idxM = [[strat_idxs[this_idx]] + strat_idxs[:this_idx] + strat_idxs[this_idx+1:] 
                     for this_idx in range(n_s)]

        # names of strategies in an ordering so focal is index 0
        stratM = [[strat_names[this_idx]] + strat_names[:this_idx] + strat_names[this_idx+1:] 
                     for this_idx in range(n_s)]

        # population proportions of strategies
        pM = [[strat_ps[this_idx]] + strat_ps[:this_idx] + strat_ps[this_idx+1:] for this_idx in range(n_s)]

        # probability of each possible strategy-count outcome in full_strat_countsV.
        ZM = list() # ZM[strategy outcome][focal strategy]
        for pV in pM:
            P = self.calc_P(pV)
            ZV = list(P @ np.array(self.F))
            ZM.append(ZV)


        # list of all possible strategy outcomes, including outcomes with strategy not present in system 
        # can include them bc p=0 for those strategies drops them out of Z
        full_strat_countsV = self.full_strat_countsV

        # \sum_{k \in K} \pi_i(k) Z(s_i, k) terms for each focal i
        # ---

        # get payoffs
        if self.payoffD:

            # retrive from payoffs dictionar
            payoffM = [[self.payoffD[self.strat_names.index(stratV[0])][tuple([strat_counts[stratV.index(x)] if x in stratV else 0 for x in self.strat_names])] for strat_counts in full_strat_countsV] for stratV in stratM]

        else:

            # calculate
            payoffM = [[self.payoff(stratV, strat_counts) for strat_counts in full_strat_countsV] for stratV in stratM]

        # term in equation
        sumk_pi_ZV = [ sum( payoff*Z for payoff, Z in zip(payoffV, ZV) ) for payoffV, ZV in zip(payoffM, ZM) ]


        # \sum_{i=1}^{n_s} \sum_{k \in K} \pi_i(k) Z(s_i, k) is always the same
        sumi_sumk_pi_Z = sum(sumk_pi_ZV)


        # identify which strategies are in the system at strat_ps steady state
        # ---

        # strategy indexes that are present at the steady state - I the derivative of \Delta p_i for these
        nzero_idxs = [ i for i in range(n_s) if not strat_ps[i] == 0 ]

        assert len(nzero_idxs) > 1, 'Only 1 strategy present at strat_ps, use calc_invasion_fitness() instead'

        # strategies that are not present in the system
        zero_idxs = [ i for i in range(n_s) if strat_ps[i] == 0 ]

        # always remove the last strategy of the present ones bc \sum_i p_i = 1
        remov_idx = nzero_idxs[-1]

        # these y I'll take the derivative with respect to, \Delta p_i / d p_y
        deriv_idxs = [ i for i in nzero_idxs if i != remov_idx ]


        # calculate each \sum_{k \in K} \pi_i(k) d Z(s_i,k) / d p_y term
        # ---

        sumk_pi_dZM = list() # sumk_pi_dZM: entries by focal-strategy, entries of entries by deriv-strategy

        for focal_idx in nzero_idxs:

            # shift indices and p to account for the focal index being shifted to index 0

            shifted_idxs = idxM[focal_idx]                      # shifted indexing of strategies
            shifted_pV = pM[focal_idx]                          # new ordering of p, proportions each strategy in population
            shifted_remov_idx = shifted_idxs.index(remov_idx)   # new index for strategy removed

            # if there are any strategies not present, we don't include strategy outcomes that involve those strategies
            # so the indices in fsc_idxs are the only indexes we consider from self.full_strat_countsV

            if zero_idxs:

                shifted_zero_idxs = [shifted_idxs.index(i) for i in zero_idxs]
                fsc_idxs = [i for i, fsc in enumerate(full_strat_countsV) if all(fsc[i] == 0 for i in shifted_zero_idxs)]
                poss_strat_countsV = [ fsc for j, fsc in enumerate(full_strat_countsV) if all(fsc[j]==0 for j in shifted_zero_idxs)]

            else:

                fsc_idxs = None
                poss_strat_countsV = full_strat_countsV

            # calculate \pi_i(k) for this i for each k \in K (possible strategy count outcomes)

            shifted_stratV = stratM[focal_idx] # names of strategies in correct order for this focal index i
            piV = [self.payoff(shifted_stratV, strat_counts) for strat_counts in poss_strat_countsV] # piV[k]

            # \sum_{k \in K} \pi_i(k) d Z(s_i,k) / d p_y for each y

            sumk_pi_dZV = list() # sumk_pi_dZM entries by deriv-strategy
            for deriv_idx in deriv_idxs:

                shifted_deriv_idx = shifted_idxs.index(deriv_idx)   # new index for what derivative is wrt

                # calculate d Z(s_i, k) / d p_y for each k \in K
                dP = self.calc_dP(shifted_pV, shifted_deriv_idx, shifted_remov_idx, fsc_idxs)
                dZV = list(dP @ np.array(self.F)) # dZV[k]

                # \sum_{k \in K} \pi_i(k) d Z(s_i,k) / d p_y for this y
                sumk_pi_dZ = sum(pi*dZ for pi, dZ in zip(piV, dZV))

                # append
                sumk_pi_dZV.append(sumk_pi_dZ)

            sumk_pi_dZM.append(sumk_pi_dZV)

        # nicer as an array
        sumk_pi_dZM = np.array(sumk_pi_dZM) # sumk_pi_dZM[focal-strategy, deriv-strategy]

        # for each derivative d p_y, the sum over strategies \sum_i (\sum_k \pi_i(k) dZ(s_i,k) / d p_y)
        sumi_sumk_pi_dZ = np.sum(sumk_pi_dZM, axis=0)


        # construct the Jacobian
        # ---

        # J(i, j) = d \Delta p_i / d p_j
        sz = len(deriv_idxs)
        Jac = np.zeros((sz, sz)) # rows are focal strategy, columns are p_strategies the derivative was taken wrt

        for i in range(sz):

            focal_idx = nzero_idxs[i]
            p = strat_ps[focal_idx]

            for j in range(sz):

                deriv_idx = deriv_idxs[j]

                if focal_idx == deriv_idx:

                    Jac[i, j] = sumk_pi_dZM[i, j] - p*sumi_sumk_pi_dZ[j] - sumi_sumk_pi_Z

                else:

                    Jac[i, j] = sumk_pi_dZM[i, j] - p*sumi_sumk_pi_dZ[j]

        return Jac


    def calc_dP(self, pV, deriv_idx, remov_idx, fsc_idxs=None):
        '''
        Find the partial derivative wrt p_y of the P matrix, the matrix that's used to calculate 
        the probability of each strategy outcome (i.e., Z = PF), at the steady state provided. The 
        derivative is useful for finding the Jacobian.

        Inputs:
        ---

        pV, list of floats
            Population proportions for each strategy at the steady state, p_i^*, where the order 
            corresponds to strategy outcomes in self.full_strat_countsV.

        deriv_idx, int
            The index in pV of the strategy we take the derivative with respect to.

        remov_idx, int
            The index of the strategy removed to reduce the dimensionality of the system.  Because 
            \sum_i p_i = 1, one of the variables is superfluous.

        fsc_idxs, list of ints
            The indexes of self.full_strat_countsV, corresponding to strategy distribution outcomes 
            in the group, that can be skipped because that strategy is 0 at the steady state. 
            Doing it this way means we don't have to define a new model in order to test the 
            stability of steady states that only involve a subset of the strategies.


        Outputs:
        ---

        dP, np.array of floats
            The matrix that converts family partition probabilies (F) into the derivative of strategy 
            outcome probabilities (dZ).
        '''

        # matrix of coefficients C and powers of p_i used to construct the P matrix, 
        # compressed so permutations of non-focal (idx > 0) are not considered
        CM = self.CM
        WM = self.WM

        # maps from the compressed to full list fo strategy outcomes with non-focal permutations included
        full_row2comp_row = self.full_row2comp_row
        full_row2order = self.full_row2order

        # group size
        n = self.n

        if fsc_idxs is None:
            rows = list(range(len(full_row2comp_row))) # do every row of full_strat_countsV, strategy outcomes
        else:
            rows = fsc_idxs

        cols = len(CM[0]) # number of family partitions, depends on group size

        dP = np.zeros((len(rows), cols))
        for irow, row in enumerate(rows):

            # match the order of strategy proportions to the order in compressed C and W
            ps = [pV[i] for i in full_row2order[row]]

            # the derivative and removal index will shift as well
            d_idx = full_row2order[row].index(deriv_idx)
            r_idx = full_row2order[row].index(remov_idx)

            # get the row of the compressed C and W corresponding to this row of the strategy-outcomes list
            CM_row = CM[full_row2comp_row[row]]
            WM_row = WM[full_row2comp_row[row]]

            for col in range(cols): # for each family partition outcome

                # the equation that constructs P from C and W while also accounting for compression

                first_term = [ 0 if pwrs[d_idx] == 0 
                        else pwrs[d_idx]*np.prod([p**(pwr-(i==d_idx)) for i, (p, pwr) in enumerate(zip(ps, pwrs))])
                        for pwrs in WM_row[col] ]

                second_term = [ 0 if pwrs[r_idx] == 0 
                        else pwrs[r_idx]*np.prod([p**(pwr-(i==r_idx)) for i, (p, pwr) in enumerate(zip(ps, pwrs))])
                        for pwrs in WM_row[col] ]

                # \sum_{possible partition n} C(n) (first term - second term)
                dP[irow, col] = (1/n) * sum( coeff * (frst-scnd) for coeff, frst, scnd in zip(CM_row[col], first_term, second_term) )

        return dP


    @abc.abstractmethod
    def payoff(self, strats, strat_counts):
        '''
        Write a function that accepts strategy distribution and returns payoff.  Ensure the game_pars 
        dictionary has all the parameters it needs.

        Inputs:
        ---

        strats, list of str:
            A list of the strategy names in the same order as strat_counts.  Index 0 in strats is 
            treated as the focal strategy.

        strat_counts:
            How many individuals in the group are pursuing which strategies.  For example, if 
            strats = ['C', 'D', 'P'] and strat_counts = [2, 1, 3], then the group has 2 
            cooperators, 1 defector, and 3 punishers.

        Outputs:
        ---

        payoff, float
            Payoff to and individual playing the focal strategy (strats[0], cooperators in example 
            above) given the distribution of strategies in strat_counts.
        '''

