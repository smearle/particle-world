from pdb import set_trace as TT

import numpy as np
import ray
from timeit import default_timer as timer
import torch as th

from rllib_utils.utils import get_env_world_heuristics


def rllib_evaluate_worlds(trainer, worlds, start_time=None, net_itr=None, idx_counter=None, evaluate_only=False, 
    quality_diversity=False, oracle_policy=False, calc_world_heuristics=False, calc_agent_stats=False):
    """
    Simulate play on a set of worlds, returning statistics corresponding to players/generators, using rllib's
    train/evaluate functions.
    :param trainer:
    :param worlds:
    :param idx_counter:
    :param evaluate_only: If True, we are not training, just evaluating some trained players/generators. (Normally,
    during training, we also evaluate at regular intervals. This won't happen here.) If True, we do not collect stats
    about generator fitness.
    :param quality_diversity: Whether we are running a QD experiment, in which case we'll return measures corresponding
    to fitnesses of distinct populations, and an objective corresponding to fitness of an additional "protagonist"
    population. Otherwise, return placeholder measures, and an objective corresponding to a contrastive measure of
    population fitnesses.
    :return:
    """
    if start_time is None:
        start_time = timer()
    idxs = np.random.permutation(list(worlds.keys()))
    if evaluate_only:
        workers = trainer.evaluation_workers
    else:
        workers = trainer.workers
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
            lambda worker: worker.foreach_env(lambda env: env.set_worlds(worlds=worlds, idx_counter=idx_counter)))

        # If using oracle, manually load the world
        if oracle_policy:
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
                    lambda env: {env.world_key: ((flood_model.get_solution_length(th.Tensor(env.world).unsqueeze(0)),), (0,0))}))
            rl_stats.append([])

        else:
            # Train/evaluate
            if evaluate_only:
                stats = trainer.evaluate()
            else:
                stats = trainer.train()
            # print(pretty_print(stats))
            rl_stats.append(stats)

            # Collect stats for generator
            new_world_stats = workers.foreach_worker(
                lambda worker: worker.foreach_env(
                    lambda env: env.get_world_stats(evaluate=evaluate_only, quality_diversity=quality_diversity)))
        # assert(len(rl_stats) == 1)  # TODO: bring back this assertion except when we're re-evaluating world archive after training
        last_rl_stats = rl_stats[-1]
        logbook_stats = {'iteration': net_itr}
        stat_keys = ['mean', 'min', 'max']  # , 'std]  # Would need to compute std manually
        # if i == 0:
        if calc_world_heuristics:
            env_world_heuristics = get_env_world_heuristics(trainer)
            logbook_stats.update({f'{k}Path': env_world_heuristics[f'{k}_path_length'] for k in stat_keys})
        if calc_agent_stats and not evaluate_only:
            logbook_stats.update({
                f'{k}Rew': last_rl_stats[f'episode_reward_{k}'] for k in stat_keys})
        if 'evaluation' in last_rl_stats:
            logbook_stats.update({
                f'{k}EvalRew': last_rl_stats['evaluation'][f'episode_reward_{k}'] for k in stat_keys})
        # logbook.record(**logbook_stats)
        # print(logbook.stream)

        # list with entries (world_key, (fit_vals, bc_vals), policy_rewards)
        new_world_stats = [fit for worker_fits in new_world_stats for fit in worker_fits]

        new_fits = {}
        for stat_lst in new_world_stats:
            for stat_tpl in stat_lst:
                world_key = stat_tpl[0]
                assert world_key not in new_fits
                new_fits[world_key] = stat_tpl[1:]

        # Ensure we have not evaluated any world twice
        for k in new_fits:
            assert k not in world_stats

        world_stats.update(new_fits)

        # If we've mapped envs to specific worlds, then we count the number of unique worlds evaluated (assuming worlds are
        # deterministic, so re-evaluation is redundant, and we may sometimes have redundant evaluations because we have too many envs).
        # Otherwise, we count the number of evaluations (e.g. when evaluating on a single fixed world).
        if idx_counter:
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