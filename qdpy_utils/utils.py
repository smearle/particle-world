import os
import pickle
from pdb import set_trace as TT
from timeit import default_timer as timer

import deap.algorithms
from qdpy.containers import *
from qdpy.phenotype import *
from rllib_utils.eval_worlds import rllib_evaluate_worlds
from rllib_utils.trainer import train_players
from rllib_utils.utils import IdxCounter

from utils import update_individuals


def qdpy_save_archive(container, play_itr, gen_itr, net_itr, logbook, save_dir):
    with open(os.path.join(save_dir, 'latest-0.p'), 'wb') as f:
        pickle.dump(
            {
                'container': container,
                'net_itr': net_itr,
                'gen_itr': gen_itr,
                'play_itr': play_itr,
                'logbook': logbook,
            }, f)

