import math
from pdb import set_trace as TT
import random
import numpy as np
from timeit import default_timer as timer

import ray

from evo.utils import get_archive_world_heuristics, qdpy_save_archive
from rl.eval_worlds import rllib_evaluate_worlds
from rl.trainer import train_players
from utils import update_individuals


def phase_switch_callback(net_itr, gen_itr, play_itr, trainer, archive, toolbox, logbook, idx_counter,
                          stale_generators, stats, cfg):
    """Callback function which checks if we should run a phase of player training.

    Launches player training if we have run the required number of generations of world-evolution, or, if
    gen_phase_len = -1, if the generators are "stale" (i.e., no new generators have been added to the archive for a
    given number of steps).
    """
    # Run a round of player training, either at fixed intervals (every gen_phase_len generations)
#   if args.objective_function == "min_solvable":
#       max_possible_generator_fitness = n_sim_steps - 1 / n_pop
#       optimal_generators = logbook.select("avg")[-1] >= max_possible_generator_fitness - 1e-3
#   else:
    optimal_generators = False

    # No player training if we're using the optimal player-policies
    if cfg.oracle_policy:
        return net_itr

    if gen_itr > 0 and (cfg.gen_phase_len != -1 and gen_itr % cfg.gen_phase_len == 0 or stale_generators or optimal_generators):
        qdpy_save_archive(container=archive, gen_itr=gen_itr, net_itr=net_itr, play_itr=play_itr, logbook=logbook,
                          save_dir=cfg.save_dir)
        training_worlds = sorted(archive, key=lambda i: i.fitness.values[0], reverse=True)
        if cfg.quality_diversity:
            # Eliminate impossible worlds
            training_worlds = [t for t in training_worlds if not t.features == [0, 0]]

            # In case all worlds are impossible, do more rounds of evolution until some worlds are feasible.
            if len(training_worlds) == 0:
                return net_itr

            training_worlds *= math.ceil(cfg.n_rllib_envs / len(training_worlds))

        mean_path_length = get_archive_world_heuristics(archive, trainer)
        logbook_stats = {}
        stat_keys = ['mean', 'min', 'max']  # , 'std]
        logbook_stats.update({f'{k}Path': mean_path_length[f'{k}_path_length'] for k in stat_keys})
        net_itr = train_players(net_itr=net_itr, play_phase_len=cfg.play_phase_len, trainer=trainer,
                                worlds=training_worlds, idx_counter=idx_counter, cfg=cfg, logbook=logbook)
        # else:
        #     if itr % play_phase_len:
        # pass
        start_time = timer()

        # Pop individuals from the container for re-evaluation.
        # Randomly select individuals to pop, without replacement
        invalid_inds = random.sample(archive, k=min(cfg.n_rllib_envs, len(archive)))
        # invalid_ind = [ind for ind in container]

        # TODO: we have an n-time lookup in discard. We should be getting the indices of the individuals directly,
        #  then discarding by index using ``discard_by_index``.
        [archive.discard(ind) for ind in invalid_inds]
        # container.clear_all()

        # After player training, the reckoning: re-evaluate all worlds against updated player policies.
        net_itr -= 1
        if cfg.rllib_eval:
            rl_stats, world_stats, logbook_stats_from_eval = rllib_evaluate_worlds(
                net_itr=net_itr, trainer=trainer, worlds={i: ind for i, ind in enumerate(invalid_inds)},
                idx_counter=idx_counter, start_time=start_time, cfg=cfg)
            logbook_stats.update(logbook_stats_from_eval)

        else:
            world_stats = toolbox.map(toolbox.evaluate, invalid_inds)

        # NOTE: Here we are assuming that we've only evaluated each world once. If we have duplicate stats for a given world, we will overwrite all but one instance of statistics 
        #  resulting from playthrough in this world.

        # Set fitness & feature attributes of the individuals we have just evaluated.
        update_individuals(invalid_inds, world_stats)

        # Add re-evaluated individuals back to the given their new stats.
        nb_updated = archive.update(invalid_inds, issue_warning=True)
        if nb_updated == 0:
            raise ValueError(
                "No individual could be added back to the QD container when re-evaluating after player training.")

        record = stats.compile(archive) if stats else {}
        logbook_stats.update({
            'iteration': net_itr, 'containerSize': archive.size_str(), 'evals': len(invalid_inds), 'nbUpdated': nb_updated,
            'elapsed': timer() - start_time, **record,
        })
        logbook.record(**logbook_stats)
        print(logbook.stream)
        net_itr += 1
    return net_itr


def iteration_callback(iteration, net_itr, play_itr, trainer, toolbox, staleness_counter, batch,
                        container, logbook, stats, cfg):
    """A callback function to be passed to the qdpy algorithm.

    TODO: remove need for net_itr and play_itr to be mutable. Replace qdpy functions/classes that are standing in
        our way.
    :param iteration: The current iteration number according to qdpy. We should remove this!
    :param staleness_counter: A mutable counter that counts how long it has been since new elite generators were added
        to the archive.
    :param batch: The current batch of individuals. QDPY passes this to us. We don't use it.
    """
    net_itr_lst = net_itr
    play_itr_lst = play_itr
    assert len(net_itr_lst) == 1 == len(play_itr_lst)
    net_itr = net_itr_lst[0]
    play_itr = play_itr_lst[0]

    gen_itr = iteration
    idx_counter = ray.get_actor('idx_counter')
    if net_itr % cfg.qdpy_save_interval == 0:
        qdpy_save_archive(container=container, play_itr=play_itr, gen_itr=gen_itr, net_itr=net_itr, logbook=logbook, save_dir=cfg.save_dir)
    time_until_stale = 10
    no_update = np.array(logbook.select('nbUpdated')[-1:]) == 0
    if no_update:
        staleness_counter[0] += 1
    else:
        staleness_counter[0] = 0
    stale = staleness_counter[0] >= time_until_stale
    if stale:
        staleness_counter[0] = 0
    net_itr = phase_switch_callback(net_itr=net_itr, gen_itr=gen_itr, play_itr=play_itr,
                            trainer=trainer, archive=container,
                            toolbox=toolbox, logbook=logbook, idx_counter=idx_counter, stale_generators=stale, stats=stats, cfg=cfg)
    net_itr_lst[0] = net_itr
    play_itr_lst[0] = play_itr
    return net_itr