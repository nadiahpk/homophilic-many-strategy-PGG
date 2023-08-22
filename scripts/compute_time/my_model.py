# defines the payoff for the pretend model I'm using to calculate how the computation time varies
# with group size and number of strategies

import sys
sys.path.append('../../functions/')
from model_base import ModelBase

# create the lottery model building on ModelBase
class MyModel(ModelBase):

    def __init__(self, *args, **kwargs):

        # inherit everything
        super(self.__class__, self).__init__(*args, **kwargs)

        # we now have names for the strategies
        self.strat_names = self.game_pars['strat_names']
        assert(len(self.strat_names) == self.n_s)    # check size matches what expecting


    def __str__(self):

        s = super().__str__()
        s += 'This is a fake model used to calculate how long it takes to evaluate delta p.\n'

        return s


    def payoff(self, strats, strat_counts):
        '''
        Always returns a 1. 
        '''

        return 1
