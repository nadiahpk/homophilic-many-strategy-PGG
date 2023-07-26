# define the sigmoid game with strategies:
#  U Unconditional Cooperator, 
#  D Unconditional Defector, 
#  C Conditional Cooperator, and 
#  L Liar

import numpy as np
from scipy.special import binom
import sys

sys.path.append('../../functions/')
from transmat_base import TransmatBase

class SigmoidUDCL(TransmatBase):

    def __init__(self, *args, **kwargs):

        # inherit everything
        super(self.__class__, self).__init__(*args, **kwargs)

        # we now have names for the strategies
        self.strat_names = self.game_pars['strat_names']
        assert(len(self.strat_names) == self.n_s)    # check size matches what expecting


    def __str__(self):

        s = super().__str__()
        s += 'Game played: A sigmoid game with the option of a lottery/straw-drawing mechanic.\n'

        return s


    def calc_payoff(self, focal_strat_idx, strat_counts):

        # parameters to set up game
        # ---

        n = self.n                      # group size
        tau = self.game_pars['tau']     # both quorum and inflection point of benefits function
        steep = self.game_pars['steep'] # steepness of the sigmoid (s = 0 is a linear game, s = infty a threshold game)

        cognitive_cost = self.game_pars['cognitive_cost']
        contrib_cost = self.game_pars['contrib_cost']

        focal_strat = self.game_pars['strat_names'][focal_strat_idx]

        # benefit from the public good given that k are contributors (inspired by Archetti's function Eq. 9 & 10)
        L = lambda k: 1 / (1 + np.exp(steep*(tau-0.5-k)/n))


        # calculate the payoff to the focal
        # ---

        # count strategies in group
        strat_idxs = [self.strat_names.index('D'), self.strat_names.index('C'), self.strat_names.index('L'), self.strat_names.index('U')]
        countD, countC, countL, countU = [strat_counts[idx] for idx in strat_idxs]

        # payoff calculation depends on if lottery quorum is met

        if countC + countL < tau: # if the lottery quorum is not met

            k = countU                              # only U will contribute
            benefit = (L(k) - L(0))/(L(n) - L(0))   # benefit from PGG

            # payoff returned depending on the strategy of the focal player
            if focal_strat == 'U':

                payoff_total = benefit + contrib_cost

            elif focal_strat == 'D':

                payoff_total = benefit

            else: # focal_strat == C or L

                payoff_total = benefit + cognitive_cost

        else:

            # probability that j Coordinating Cooperators will be designated as contributors by the lottery
            denom = binom(countC + countL, tau)
            PjV = [binom(countC, j)*binom(countL, tau-j)/denom for j in range(0, tau+1)]

            # the benefit returned for each j
            benefitjV = [(L(j+countU) - L(0))/(L(n) - L(0)) for j in range(0, tau+1)]

            # payoff returned depending on the strategy of the focal player
            if focal_strat == 'U':

                payoff_total = sum(Pj*benefitj for Pj, benefitj in zip(PjV, benefitjV)) + contrib_cost

            elif focal_strat == 'D':

                payoff_total = sum(Pj*benefitj for Pj, benefitj in zip(PjV, benefitjV))

            elif focal_strat == 'C':

                payoff_total = sum(PjV[j]*(benefitjV[j] + contrib_cost*j/countC) for j in range(0, tau+1)) + cognitive_cost

            else: # focal_strat == 'L':

                payoff_total = sum(Pj*benefitj for Pj, benefitj in zip(PjV, benefitjV)) + cognitive_cost

        return payoff_total


