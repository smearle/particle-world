from pdb import set_trace as TT

import numpy as np
import torch as th
from torch import nn

from evo.individuals import Individual
from evo.models import PlayNCA


class Player(Individual):
    def __init__(self, obs_width, obs_n_chan, player_chan=3):
        self.model = PlayNCA(n_chan=obs_n_chan, player_chan=player_chan)

    def get_actions(self, obs):
        act = self.model(th.Tensor(obs).permute(0, 3, 1, 2))
        act -= act.min()
        act = act / th.norm(act, p=1, dim=1)[:,None]
        act = act.cpu().detach().numpy()
        acts = [np.random.choice(np.arange(len(act[0])), p=act[i]) for i in range(len(act))]

        return acts


