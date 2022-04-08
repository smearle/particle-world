import math
from pdb import set_trace as TT
from timeit import default_timer as timer

import numpy as np
import ray
import torch as th

from utils import discrete_to_onehot, get_solution


def get_env_world_complexity(trainer):
    """Compute (a proxy) for the complexity of all of the worlds currently loaded in rllib environments in the trainer.
    """
    # TODO: This flood performs fast parallelized BFS. Validate it against BFS and use it instead!
    # flood_model = trainer.get_policy('oracle').model

    # This is conveniently parallelized by rllib for us.
    path_lengths = trainer.workers.foreach_worker(
        # lambda w: w.foreach_env(lambda e: flood_model.get_solution_length(th.Tensor(e.world[None,...]))))
        lambda w: w.foreach_env(lambda e: len(get_solution(e.world_flat))))

    path_lengths = [p for worker_paths in path_lengths for p in worker_paths]
    mean_path_length = np.mean(path_lengths)
    min_path_length = np.min(path_lengths)
    max_path_length = np.max(path_lengths)
    # std_path_length = np.std(path_lengths)
    return {
        'mean_path_length': mean_path_length,
        'min_path_length': min_path_length,
        'max_path_length': max_path_length,
    }
    # return mean_path_length, min_path_length, max_path_length  #, std_path_length


@ray.remote
class IdxCounter:
    ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
    responsible for providing unique indices to parallel environments.'''

    def __init__(self):
        self.count = 0
        self.world_keys = None

    def get(self, hsh):
        world_key_queue = self.hashes_to_idxs[hsh]

        return world_key_queue

        # if self.idxs is None:
        #     Then we are doing inference and have set the idx directly
        #
        # return self.count
        #
        # idx = self.idxs[self.count % len(self.idxs)]
        # self.count += 1
        #
        # return idx

    def set(self, i):
        # For inference
        self.count = i

    def set_idxs(self, idxs):
        self.count = 0
        self.world_keys = idxs

    def set_hashes(self, hashes, allow_one_to_many: bool=True):
        """
        Args:
            hashes: A list of hashes, one per environment object.
            allow_one_to_many (bool): Whether we allow one environment/hash to be mapped to multiple worlds (self.idxs).
        """
        hashes_to_idxs = {h: [] for h in hashes}
        for i, wk in enumerate(self.world_keys):
            h = hashes[i % len(hashes)]
            hashes_to_idxs[h].append(wk)
        
#        if not allow_one_to_many:
#            assert len(hashes) >= len(self.world_keys)
#            # If we have more hashes than indices, map many-to-one
##           if len(hashes) > len(idxs):
##               n_repeats = math.ceil(len(hashes) / len(idxs))
##               idxs = np.tile(idxs, n_repeats)
#
#            # Map hashes of each environment to world keys/names/IDs
#            self.hashes_to_idxs = {hsh: id for hsh, id in zip(hashes, self.world_keys[:len(hashes)])}
#       else:
        # Just for efficiency.
#       assert len(self.world_keys) % len(hashes) == 0
#       n_worlds_per_env = len(self.world_keys) // len(hashes)
#       hashes_to_idxs = {}
#       for i, hsh in enumerate(hashes):
#           hashes_to_idxs[hsh] = self.world_keys[i * n_worlds_per_env:(i + 1) * n_worlds_per_env]

        self.hashes_to_idxs = hashes_to_idxs


    def scratch(self):
        return self.hashes_to_idxs