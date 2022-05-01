from argparse import Namespace
import math
from pdb import set_trace as TT
from timeit import default_timer as timer

import numpy as np
import ray
from ray.rllib.evaluation.worker_set import WorkerSet
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
        self.keys = None

    def get(self, hsh):
        world_key_queue = self.hashes_to_keys[hsh]

        if not world_key_queue:
            raise Exception("No world keys provided.")

        return world_key_queue

    def set(self, i):
        # For inference
        self.count = i

    def set_keys(self, keys):
        self.count = 0
        self.keys = keys

    def set_hashes(self, hashes):
        """
        Note that we may assign multiple worlds to a single environment, or a single world to multiple environments.

        We will only assign a single world to multiple environments if duplicate keys were provided to `set_idxs()`.

        Args:
            hashes: A list of hashes, one per environment object.
        """
        hashes_to_keys = {h: [] for h in hashes}

        for i, wk in enumerate(self.keys):
            h = hashes[i % len(hashes)]
            hashes_to_keys[h].append(wk)
        
        self.hashes_to_keys = hashes_to_keys

    def scratch(self):
        return self.hashes_to_keys


def set_worlds(worlds: dict, workers: WorkerSet, idx_counter: IdxCounter, cfg: Namespace, load_now: bool = False):
    """Assign worlds to environments to be loaded at next reset."""
    # keys = np.random.permutation(list(worlds.keys()))
    keys = list(worlds.keys())
    world_gen_sequences = {k: world.gen_sequence for k, world in worlds.items() if hasattr(world, 'gen_sequence')} \
        if cfg.render else None

    # If not `fixed_worlds`, these will be Individuals. Otherwise, they are already simply arrays.
    if not isinstance(list(worlds.values())[0], np.ndarray):
        worlds = {k: np.array(world.discrete) for k, world in worlds.items()}

    # Have to get hashes on remote workers. Objects have different hashes in "envs" above.
    hashes = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
    hashes = [h for wh in hashes for h in wh]
    n_envs = len(hashes)

    # Pad the list of indices with duplicates in case we have more than enough eval environments
    keys = keys * math.ceil(n_envs / len(keys))
    keys = keys[:n_envs]

    idx_counter.set_keys.remote(keys)
    idx_counter.set_hashes.remote(hashes)

    # FIXME: Sometimes hash-to-idx dict is not set by the above call?
    assert ray.get(idx_counter.scratch.remote())

    # Assign envs to worlds
    workers.foreach_worker(
        lambda worker: worker.foreach_env(
            lambda env: env.queue_worlds(worlds=worlds, idx_counter=idx_counter, 
                world_gen_sequences=world_gen_sequences, load_now=load_now)))


def get_world_stats_from_hist_stats(hist_stats: dict, cfg: Namespace):
    world_stats = [{k: hist_stats[k][i] for k in hist_stats} for i in range(len(hist_stats['world_key']))]

    # Take sum of each stat for each world. Can have nested dicts (2 layers deep).
    mean_world_stats = {}
    for ws in world_stats:
        wk = ws.pop('world_key')
        if wk not in mean_world_stats:
            ws['n_episodes'] = 1
            mean_world_stats[wk] = ws
        else:
            for k in ws:
                if isinstance(ws[k], dict):
                    for k1 in ws[k]:
                        mean_world_stats[wk][k][k1] = mean_world_stats[wk][k][k1] + ws[k][k1]
                else:
                    mean_world_stats[wk][k] = mean_world_stats[wk][k] + ws[k]
            mean_world_stats[wk]['n_episodes'] += 1

    # Now take average of each stat.
    for wk in mean_world_stats:
        for k in mean_world_stats[wk]:
            if isinstance(mean_world_stats[wk][k], dict):
                for k1 in mean_world_stats[wk][k]:
                    mean_world_stats[wk][k][k1] = mean_world_stats[wk][k][k1] / mean_world_stats[wk]['n_episodes']
            else:
                mean_world_stats[wk][k] = mean_world_stats[wk][k] / mean_world_stats[wk]['n_episodes']

    [ws.update({'qd_stats': ((ws['obj'],), ws['measures'])}) for ws in mean_world_stats.values()]

    return mean_world_stats


def get_world_qd_stats(world_stats: list, cfg: Namespace, ignore_redundant=False):
    """Get world stats from workers."""
    # world_stats = workers.foreach_worker(
        # lambda worker: worker.foreach_env(lambda env: env.get_world_stats(quality_diversity=cfg.quality_diversity)))
    # world_stats = [s for ws in world_stats for s in ws]
    # Extract QD stats (objectives and features) from the world stats.
    new_qd_stats = {}
    for stats_dict in world_stats:
        world_key = stats_dict["world_key"]
        if world_key in new_qd_stats:
            warn_msg = ("Should not have redundant world evaluations inside this function unless training on "\
                        "fixed worlds or doing evaluation/enjoyment.")
            # assert cfg.fixed_worlds or evaluate_only, warn_msg
            if not cfg.fixed_worlds and not cfg.evaluate and not ignore_redundant:
                print(warn_msg)

            # We'll create a list of stats from separate runs.
            new_qd_stats[world_key] = new_qd_stats[world_key] + [stats_dict["qd_stats"]]

        else:
            # Note that this will be a tuple.
            new_qd_stats[world_key] = [stats_dict["qd_stats"]]
    
    # Take the average of each objective and feature value across all runs.
    aggregate_new_qd_stats = {}
    for world_key, world_qd_stats in new_qd_stats.items():
        n_trials = len(world_qd_stats)
        if n_trials > 1:
            aggregate_new_qd_stats[world_key] = ([], [])

            # Get the mean in each objective value across trials.
            n_objs = len(world_qd_stats[0][0])  # Length of objective tuple from first trial.
            for obj_i in range(n_objs):
                aggregate_new_qd_stats[world_key][0].append(np.mean([world_qd_stats[trial_i][0][obj_i] \
                    for trial_i in range(n_trials)]))

            # Get the mean in each feature value across trials.
            n_feats = len(world_qd_stats[0][1])  # Length of feature list from first trial.
            for feat_i in range(n_feats):
                aggregate_new_qd_stats[world_key][1].append(np.mean([world_qd_stats[trial_i][1][feat_i] \
                    for trial_i in range(n_trials)]))

        else:
            aggregate_new_qd_stats[world_key] = new_qd_stats[world_key][0]

    return aggregate_new_qd_stats
