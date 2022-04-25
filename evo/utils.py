import os
from operator import attrgetter
from pdb import set_trace as TT
import pickle
import random
from timeit import default_timer as timer

import numpy as np
import torch as th

from utils import discrete_to_onehot


def compute_archive_world_heuristics(archive, trainer):
    """Compute metrics (e.g. a proxy for the complexity) of all of the worlds currently stored in the archive.

    These are heuristics that reflect the quality/features of the worlds in the archive, without requiring simulation
    with a learning agent, or which may use an "oracle" policy to approximate world complexity (e.g. shortes path-length
    between spawn and goal-point in maze).
    """
    start_time = timer()
    # TODO: This flood performs fast parallelized BFS. Validate it against BFS and use it instead!
    flood_model = trainer.get_policy('oracle').model

    get_path_length = lambda ind: flood_model.get_solution_length(th.Tensor(discrete_to_onehot(ind.discrete, ind.n_chan)[None,...]))
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


def selRoulette(individuals, k, fit_attr="fitness"):
    """Select *k* individuals from the input *individuals* using *k*
    spins of a roulette. The selection is made by looking only at the first
    objective of each individual. The list returned contains references to
    the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    This has been adapted from qdpy to support maximizing single objectives with values less than or equal to 0.
    """

    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    fits = [getattr(ind, fit_attr) for ind in individuals]
    min_fit = min([f.values[0] for f in fits])
    sum_fits = sum(getattr(ind, fit_attr).values[0] + min_fit for ind in individuals)
    if sum_fits == 0:
        return [random.choice(individuals) for i in range(k)]

    chosen = []
    for i in range(k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in s_inds:
            sum_ += getattr(ind, fit_attr).values[0] + min_fit
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


def save(world_archive, player_archive, play_itr, gen_itr, net_itr, logbook, save_dir, adversarial_archive=False):
    arch_name = 'latest-0.p' if not adversarial_archive else 'adversarial_worlds.p'
    with open(os.path.join(save_dir, arch_name), 'wb') as f:
        pickle.dump(
            {
                'world_archive': world_archive,
                'player_archive': player_archive,
                'net_itr': net_itr,
                'gen_itr': gen_itr,
                'play_itr': play_itr,
                'logbook': logbook,
            }, f)

