import math
from pdb import set_trace as TT
import numpy as np
from utils import discrete_to_onehot, get_solution

import ray
import torch as th


def get_archive_world_complexity(archive, trainer):
    """Compute (a proxy) for the complexity of all of the worlds currently stored in the archive.
    """
    # TODO: This flood performs fast parallelized BFS. Validate it against BFS and use it instead!
    flood_model = trainer.get_policy('oracle').model

    # Compute path-length of each individual where it has not been computed previously.
#   path_lengths_old = [ind.path_length if hasattr(ind, "path_length") else len(get_solution(ind.discrete)) \
#       for ind in archive]
    path_lengths = [ind.path_length if hasattr(ind, "path_length") else \
        flood_model.get_solution_length(th.Tensor(discrete_to_onehot(ind.discrete)[None,...])) \
        for ind in archive]
    
    # Assign individuals their corresponding path-length if it was not computed previously.
    [setattr(ind, "path_length", pl) for ind, pl in zip(archive, path_lengths) if not hasattr(ind, "path_length")]

    mean_path_length = np.mean(path_lengths)
    min_path_length = np.min(path_lengths)
    max_path_length = np.max(path_lengths)
    # std_path_length = np.std(path_lengths)
    return {
        'mean_path_length': mean_path_length,
        'min_path_length': min_path_length,
        'max_path_length': max_path_length,
    }


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