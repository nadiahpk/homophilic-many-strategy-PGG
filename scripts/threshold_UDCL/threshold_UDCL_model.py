# define the threshold game with options:
#  U Unconditional Cooperator, 
#  D Unconditional Defector, 
#  C Coordinating Cooperator, and 
#  L Liar

import numpy as np
from scipy.special import binom
import sys

sys.path.append('../../functions/')
from model_base import ModelBase

# create the lottery model building on ModelBase
class ThresholdUDCL(ModelBase):

    def __init__(self, *args, **kwargs):

        # inherit everything
        super(self.__class__, self).__init__(*args, **kwargs)

        # we now have names for the strategies
        self.strat_names = self.game_pars['strat_names']
        assert(len(self.strat_names) == self.n_s)    # check size matches what expecting


    def __str__(self):

        s = super().__str__()
        s += 'Game played: A threshold game with the option of a lottery/straw-drawing mechanic.\n'

        return s


    def payoff(self, strats, strat_counts):
        '''
        A threshold-game benefits function combined with straw-drawing.

        Inputs:
        ---

        strats, list of str
            A list of the strategy names in the same order as strat_counts.
            Index 0 in strats is treated as the focal strategy.

        strat_counts, list of ints
            How many individuals in the group are pursuing which strategies.
            e.g., if strats = ['U', 'D', 'L'] and strat_counts = [2, 1, 3]
            then the group has 2 Unconditional Cooperators, 1 Unconditional Defector, and 3 Liars.

        Outputs:
        ---

        payoff, float
            Payoff to and individual playing the focal strategy (strats[0])
            (cooperators in example above) given the distribution of strategies in strat_counts
        '''


        # parameters to set up game
        # ---

        # unpack needed parameters
        n = self.n

        # unpack needed parameters
        tau = self.game_pars['tau'] # both the quorum and the threshold

        cognitive_cost = self.game_pars['cognitive_cost']
        contrib_cost = self.game_pars['contrib_cost']
        benefit_thresh_met = self.game_pars['benefit_thresh_met']

        focal_strat = strats[0]


        # check we got sensible inputs
        # ---

        assert(sum(strat_counts) == n)
        assert(strat_counts[0] > 0)


        # calculate the payoff to the focal
        # ---

        # count the number of each strategy in the group

        countDict = dict(zip(strats, strat_counts))
        countU = 0 if 'U' not in countDict else countDict['U'] # Unconditional Cooperators
        countD = 0 if 'D' not in countDict else countDict['D'] # Unconditional Defectors
        countC = 0 if 'C' not in countDict else countDict['C'] # Coordinating Cooperators
        countL = 0 if 'L' not in countDict else countDict['L'] # Liars


        # payoff calculation depends on if lottery quorum is met

        if countC + countL < tau: # if the lottery quorum is not met

            k = countU                                      # only U will contribute
            benefit = benefit_thresh_met if k >= tau else 0 # benefit from PGG

            # payoff returned depending on the strategy of the focal player
            if focal_strat == 'U':

                payoff_total = benefit + contrib_cost

            elif focal_strat == 'D':

                payoff_total = benefit

            else: # focal_strat == C or L

                payoff_total = benefit + cognitive_cost

        else: # lottery quorum is met

            # probability that j Coordinating Cooperators will be designated as contributors by the lottery
            denom = binom(countC + countL, tau)
            PjV = [binom(countC, j)*binom(countL, tau-j)/denom for j in range(0, tau+1)]

            # the benefit returned for each j
            benefitjV = [benefit_thresh_met if j+countU >= tau else 0 for j in range(0, tau+1)]

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
