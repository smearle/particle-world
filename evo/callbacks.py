import math
from pdb import set_trace as TT
import random
import sys
import numpy as np
from timeit import default_timer as timer

import ray

from evo.utils import compute_archive_world_heuristics, save
from rl.eval_worlds import evaluate_worlds
from rl.trainer import train_players
from utils import update_individuals
from visualize import visualize_archive


# def phase_switch_callback(net_itr, gen_itr, play_itr, trainer, archive, toolbox, logbook, idx_counter,
#                           stale_generators, stats, cfg):
#     """Callback function which checks if we should run a phase of player training.
# 
#     Launches player training if we have run the required number of generations of world-evolution, or, if
#     gen_phase_len = -1, if the generators are "stale" (i.e., no new generators have been added to the archive for a
#     given number of steps).
#     """
#     # Run a round of player training, either at fixed intervals (every gen_phase_len generations)
# #   if cfg.objective_function == "min_solvable":
# #       max_possible_generator_fitness = n_sim_steps - 1 / n_pop
# #       optimal_generators = logbook.select("avg")[-1] >= max_possible_generator_fitness - 1e-3
# #   else:
#     optimal_generators = False
# 
#     # No player training if we're using the optimal player-policies
#     if cfg.oracle_policy:
#         return net_itr
# 
#     # Launch player training if appropriate
#     if gen_itr > 0 and (cfg.gen_phase_len != -1 and gen_itr % cfg.gen_phase_len == 0 or stale_generators or optimal_generators):
#         save(archive=archive, gen_itr=gen_itr, net_itr=net_itr, play_itr=play_itr, logbook=logbook,
#                           save_dir=cfg.save_dir)
# 
#         mean_path_length = get_archive_world_heuristics(archive, trainer)
#         logbook_stats = {}
#         stat_keys = ['mean', 'min', 'max']  # , 'std]
#         logbook_stats.update({f'{k}Path': mean_path_length[f'{k}_path_length'] for k in stat_keys})
# 
#         if cfg.gen_adversarial_worlds:
#             # Then generator evolution has stagnated, so we are done.
#             env = trainer.workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.env))
#             visualize_archive(cfg, env, archive)
#             print('Done generating adversarial worlds.')
#             sys.exit()
# 
#         training_worlds = sorted(archive, key=lambda i: i.fitness.values[0], reverse=True)
#         if cfg.quality_diversity:
#             # Eliminate impossible worlds
#             training_worlds = [t for t in training_worlds if not t.features == [0, 0]]
# 
#             # In case all worlds are impossible, do more rounds of evolution until some worlds are feasible.
#             if len(training_worlds) == 0:
#                 return net_itr
# 
#             training_worlds *= math.ceil(cfg.world_batch_size / len(training_worlds))
# 
#         net_itr = train_players(net_itr=net_itr, play_phase_len=cfg.play_phase_len, trainer=trainer,
#                                 worlds=training_worlds, idx_counter=idx_counter, cfg=cfg, logbook=logbook)
#         # else:
#         #     if itr % play_phase_len:
#         # pass
#         start_time = timer()
# 
#         # Pop individuals from the container for re-evaluation.
#         # Randomly select individuals to pop, without replacement
#         invalid_inds = random.sample(archive, k=min(cfg.world_batch_size, len(archive)))
#         # invalid_ind = [ind for ind in container]
# 
#         # TODO: we have an n-time lookup in discard. We should be getting the indices of the individuals directly,
#         #  then discarding by index using ``discard_by_index``.
#         [archive.discard(ind) for ind in invalid_inds]
#         # container.clear_all()
# 
#         # After player training, the reckoning: re-evaluate all worlds against updated player policies.
#         net_itr -= 1
#         print(f"{len(invalid_inds)} up for re-evaluation.")
#         rl_stats, world_stats, logbook_stats_from_eval = evaluate_worlds(
#             net_itr=net_itr, trainer=trainer, worlds={i: ind for i, ind in enumerate(invalid_inds)},
#             idx_counter=idx_counter, start_time=start_time, cfg=cfg)
#         logbook_stats.update(logbook_stats_from_eval)
# 
#         # NOTE: Here we are assuming that we've only evaluated each world once. If we have duplicate stats for a given world, we will overwrite all but one instance of statistics 
#         #  resulting from playthrough in this world.
# 
#         # Set fitness & feature attributes of the individuals we have just evaluated.
#         update_individuals(invalid_inds, world_stats)
# 
#         # Add re-evaluated individuals back to the given their new stats.
#         nb_updated = archive.update(invalid_inds, issue_warning=True)
#         if nb_updated == 0:
#             raise ValueError(
#                 "No individual could be added back to the QD container when re-evaluating after player training.")
# 
#         record = stats.compile(archive) if stats else {}
#         logbook_stats.update({
#             'iteration': net_itr, 'containerSize': archive.size_str(), 'evals': len(invalid_inds), 'nbUpdated': nb_updated,
#             'elapsed': timer() - start_time, **record,
#         })
#         logbook.record(**logbook_stats)
#         print(logbook.stream)
#         net_itr += 1
#     return net_itr
