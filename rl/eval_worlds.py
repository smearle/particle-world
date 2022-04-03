from pdb import set_trace as TT

import numpy as np
import ray
from timeit import default_timer as timer
import torch as th

from ray.tune.logger import pretty_print
from evo.utils import get_archive_world_heuristics


def rllib_evaluate_worlds(trainer, worlds, cfg, start_time=None, net_itr=None, idx_counter=None, evaluate_only=False, 
                          is_training_player=False):
    """
    Simulate play on a set of worlds, returning statistics corresponding to players/generators, using rllib's
    train/evaluate functions.

    If we are running a QD experiment (cfg.quality_diversity), we'll return measures corresponding
    to fitnesses of distinct populations, and an objective corresponding to fitness of an additional "protagonist"
    population. Otherwise, return placeholder measures, and an objective corresponding to a contrastive measure of
    population fitnesses.

    :param trainer: an rllib trainer, holding policy networks, and training/evaluation workers with environments.
    :param worlds: a dictionary of words on which to simulate gameplay.
    :param idx_counter:
    :param evaluate_only: If True, we are not training, just evaluating some trained players/generators. (Normally,
    during training, we also evaluate at regular intervals. This won't happen here.) If True, we do not collect stats
    about generator fitness.
    :param is_training_player: (bool) Whether we are currently training the player (i.e. its weights are unfrozen and 
        rllib is set to train it). If so, we will print/log relevant stats.
    :return:
    """
    if start_time is None:
        start_time = timer()
    idxs = np.random.permutation(list(worlds.keys()))
    if evaluate_only:
        workers = trainer.evaluation_workers
    else:
        workers = trainer.workers
    world_gen_sequences = {k: world.gen_sequence for k, world in worlds.items() if hasattr(world, 'gen_sequence')} \
        if cfg.render else None
    if not isinstance(list(worlds.values())[0], np.ndarray):
        worlds = {k: np.array(world.discrete) for k, world in worlds.items()}
    # fitnesses = {k: [] for k in worlds}
    rl_stats = []

    # Train/evaluate on all worlds n_trials many times each
    # for i in range(n_trials):
    world_stats = {}
    world_id = 0

    # Train/evaluate until we have simulated in all worlds
    while world_id < len(worlds):

        # When running parallel envs, if each env is to evaluate a separate world, map envs to worlds
        if idx_counter:

            # Have to get hashes on remote workers. Objects have different hashes in "envs" above.
            hashes = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
            hashes = [h for wh in hashes for h in wh]
            n_envs = len(hashes)

            sub_idxs = idxs[world_id:min(world_id + n_envs, len(idxs))]
            idx_counter.set_idxs.remote(sub_idxs)
            idx_counter.set_hashes.remote(hashes)

            # FIXME: Sometimes hash-to-idx dict is not set by the above call?
            assert ray.get(idx_counter.scratch.remote())

        # Assign envs to worlds
        workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.set_worlds(worlds=worlds, idx_counter=idx_counter, world_gen_sequences=world_gen_sequences)))

        # If using oracle, manually load the world
        if cfg.oracle_policy:
            flood_model = trainer.get_policy('policy_0').model
            workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env.reset()))

            # Hardcoded rendering
            envs = workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env))
            envs = [e for we in envs for e in we]
            envs[0].render()

            new_world_stats = workers.foreach_worker(
                lambda worker: worker.foreach_env(
                    # NOTE: need to match what is returned by env in get_world_stats here
                    #  i.e., [(world_key, ((qd_objective,), (qd_features...)), [swarm_rewards...]), ...]
                    lambda env: [(env.world_key, ((flood_model.get_solution_length(th.Tensor(env.world).unsqueeze(0)),), (0,0)), [])]))
            rl_stats.append([])

        else:
            # Train/evaluate
            if evaluate_only:
                stats = trainer.evaluate()
            else:
                stats = trainer.train()
#               if is_training_player:
#                   log_result = {k: v for k, v in stats.items() if k in cfg.log_keys}
#                   log_result['info: learner:'] = stats['info']['learner']

#                   # FIXME: sometimes timesteps_this_iter is 0. Maybe a ray version problem? Weird.
#                   log_result['fps'] = stats['timesteps_this_iter'] / stats['time_this_iter_s']

#                   print('-----------------------------------------')
#                   print(pretty_print(log_result))
            # print(pretty_print(stats))
            rl_stats.append(stats)

            # Collect stats for generator
            new_world_stats = workers.foreach_worker(
                lambda worker: worker.foreach_env(
                    lambda env: env.get_world_stats(evaluate=evaluate_only, quality_diversity=cfg.quality_diversity)))
        # assert(len(rl_stats) == 1)  # TODO: bring back this assertion except when we're re-evaluating world archive after training
        last_rl_stats = rl_stats[-1]
        logbook_stats = {'iteration': net_itr}
        stat_keys = ['mean', 'min', 'max']  # , 'std]  # Would need to compute std manually
        # if i == 0:
#       if calc_world_heuristics:
#           # TODO: only getting these heuristics on the subset of maps on which we train, at the moment
#           env_world_heuristics = get_env_world_complexity(trainer)
#           logbook_stats.update({f'{k}Path': env_world_heuristics[f'{k}_path_length'] for k in stat_keys})
        if is_training_player and not evaluate_only:
            logbook_stats.update({
                f'{k}Rew': last_rl_stats[f'episode_reward_{k}'] for k in stat_keys})
        if 'evaluation' in last_rl_stats:
            logbook_stats.update({
                f'{k}EvalRew': last_rl_stats['evaluation'][f'episode_reward_{k}'] for k in stat_keys})
        # logbook.record(**logbook_stats)
        # print(logbook.stream)

        # list with entries (world_key, (fit_vals, bc_vals), policy_rewards)
        new_world_stats = [stats for worker_world_stats in new_world_stats for stats in worker_world_stats]

        # Extract QD stats (objectives and features) from the world stats.
        new_qd_stats = {}
        for stat_lst in new_world_stats:
            for stats_dict in stat_lst:
                world_key = stats_dict["world_key"]
                if world_key in new_qd_stats:
                    warn_msg = ("Should not have redundant world evaluations inside this function unless training on "\
                                "fixed worlds or doing evaluation/enjoyment.")
                    # assert cfg.fixed_worlds or evaluate_only, warn_msg
                    if not cfg.fixed_worlds and not evaluate_only:
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

        # print("Aggregate new qd stats:", aggregate_new_qd_stats)

        # Ensure we have not evaluated any world twice
        for k in aggregate_new_qd_stats:
            if k in world_stats:
                assert cfg.fixed_worlds or evaluate_only
        if cfg.fixed_worlds:
            for k in list(new_qd_stats.keys()):
                if k in world_stats:
                    world_stats[k] = [world_stats[k]] + [new_qd_stats.pop(k)]

        else:
            world_stats.update(aggregate_new_qd_stats)
        
        # print("World stats:", world_stats)

        # If we've mapped envs to specific worlds, then we count the number of unique worlds evaluated (assuming worlds are
        # deterministic, so re-evaluation is redundant, and we may sometimes have redundant evaluations because we have too many envs).
        # Otherwise, we count the number of evaluations (e.g. when evaluating on a single fixed world).
        if idx_counter:
            assert len(world_stats) == len(idxs)
            world_id = len(world_stats)
        else:
            world_id += len(new_world_stats)

    logbook_stats.update({
        'elapsed': timer() - start_time,
    })

        # [fitnesses[k].append(v) for k, v in trial_fitnesses.items()]
    # fitnesses = {k: ([np.mean([vi[0][fi] for vi in v]) for fi in range(len(v[0][0]))],
    #         [np.mean([vi[1][mi] for vi in v]) for mi in range(len(v[0][1]))]) for k, v in fitnesses.items()}

    return last_rl_stats, world_stats, logbook_stats