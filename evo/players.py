from pdb import set_trace as TT

import numpy as np
from qdpy.phenotype import Features
import torch as th
from torch import nn

from evo.individuals import Fitness, Individual
from evo.models import PlayNCA, FullObsPlayNCA


class Player(Individual):
    def __init__(self, obs_width, obs_n_chan, *args, player_chan=3, **kwargs):
        Individual.__init__(self, fitness=Fitness((0,), weights=(1,)), features=Features(0,0))
        self.model = PlayNCA(obs_width=obs_width, n_chan=obs_n_chan, player_chan=player_chan)

    def get_actions(self, obs):
        act = self.model(th.Tensor(obs).permute(0, 3, 1, 2))
        act -= act.min()
        act = act / th.norm(act, p=1, dim=1)[:,None]
        act = act.cpu().detach().numpy()
        acts = [np.random.choice(np.arange(len(act[0])), p=act[i]) for i in range(len(act))]

        return acts

    def reset(self):
        self.model.reset()

    def mutate(self, *args, **kwargs):
        self.model.mutate(*args, **kwargs)

        return self, 

    def __eq__(self, other):
        # Don't bother comparing network weights. All distinct objects are different.
        return False
        # return (self.__class__ == other.__class__ and hash(self) == hash(other))
