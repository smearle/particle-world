from timeit import default_timer as timer
import os
import pickle

import numpy as np
import torch as th

from utils import discrete_to_onehot


def get_archive_world_heuristics(archive, trainer):
    """Compute metrics (e.g. a proxy for the complexity) of all of the worlds currently stored in the archive.

    These are heuristics that reflect the quality/features of the worlds in the archive, without requiring simulation
    with a learning agent, or which may use an "oracle" policy to approximate world complexity (e.g. shortes path-length
    between spawn and goal-point in maze).
    """
    start_time = timer()
    # TODO: This flood performs fast parallelized BFS. Validate it against BFS and use it instead!
    flood_model = trainer.get_policy('oracle').model

    get_path_length = lambda ind: flood_model.get_solution_length(th.Tensor(discrete_to_onehot(ind.discrete)[None,...]))
    # get_path_length_old = lambda ind: get_solution(ind.discrete)

    # Compute path-length of each individual where it has not been computed previously, and store this information with 
    # the individual.
    # print(f"{len([ind for ind in archive if 'path_length' not in ind.stats['heuristics']])} new individuals.")
    [ind.stats["heuristics"].update({"path_length": get_path_length(ind)}) 
        for ind in archive if "path_length" not in ind.stats['heuristics']]
#   path_lengths_old = [ind.path_length if hasattr(ind, "path_length") else len(get_solution(ind.discrete)) \
#       for ind in archive]

    path_lengths = [ind.stats["heuristics"]["path_length"] for ind in archive]
    mean_path_length = np.mean(path_lengths)
    min_path_length = np.min(path_lengths)
    max_path_length = np.max(path_lengths)
    # std_path_length = np.std(path_lengths)
    # print("Computed path-lengths of {} individuals in {:.2e} seconds.".format(len(archive), timer() - start_time))

    return {
        'mean_path_length': mean_path_length,
        'min_path_length': min_path_length,
        'max_path_length': max_path_length,
    }


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

