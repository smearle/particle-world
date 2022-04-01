import math
import numpy as np
from utils import get_solution

import ray


def get_env_world_heuristics(trainer):
    """Get heuristics for all the worlds currently loaded in the environments managed by the trainer."""
    flood_model = trainer.get_policy('oracle').model

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
        self.idxs = None

    def get(self, hsh):
        return self.hashes_to_idxs[hsh]

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
        self.idxs = idxs

    def set_hashes(self, hashes, allow_one_to_many: bool=False):
        if not allow_one_to_many:
            assert len(hashes) >= len(self.idxs)
        idxs = self.idxs

        # If we have more hashes than indices, map many-to-one
        if len(hashes) > len(idxs):
            n_repeats = math.ceil(len(hashes) / len(idxs))
            idxs = np.tile(idxs, n_repeats)
        self.hashes_to_idxs = {hsh: id for hsh, id in zip(hashes, idxs[:len(hashes)])}

    def scratch(self):
        return self.hashes_to_idxs