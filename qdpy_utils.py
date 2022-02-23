import os
import deap.tools
import deap.algorithms
import numpy as np
from timeit import default_timer as timer
import pickle
import copy
from pdb import set_trace as TT

from qdpy.phenotype import *
from qdpy.containers import *

from rllib_utils import IdxCounter, rllib_evaluate_worlds
from utils import update_individuals


def qdRLlibEval(init_batch, toolbox, container, batch_size, niter, 
                rllib_trainer, rllib_eval: bool, quality_diversity: bool, oracle_policy: bool, net_itr: int, gen_itr: int, play_itr: int,
                cxpb: float=0.0, mutpb:float=1.0, stats=None, logbook=None,
                halloffame=None, verbose=False, show_warnings=True, start_time=None, iteration_callback=None):
    """Simple QD algorithm using DEAP, modified to evaluate generated worlds inside an RLlib trainer object.
    :param rllib_trainer: RLlib trainer object.
    :param rllib_eval: #TODO
    :param quality_diversity: If this is False, we are reducing this to a vanilla evolutionary strategy.
    :param init_batch: Sequence of individuals used as initial batch.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param batch_size: The number of individuals in a batch.
    :param niter: The number of iterations.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param show_warnings: Whether or not to show warnings and errors. Useful to check if some individuals were out-of-bounds.
    :param start_time: Starting time of the illumination process, or None to take the current time.
    :param iteration_callback: Optional callback funtion called when a new batch is generated. The callback function parameters are (iteration, batch, container, logbook).
    :param gen_itr: The current generator iteration if reloading.
    :param net_itr: The current net iteration if reloading (sum of generator and player iterations).
    :returns: The final batch
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    if start_time == None:
        start_time = timer()
    # The co-learning loop will always start here, with at least a single round of world-generation
    idx_counter = ray.get_actor("idx_counter")
    if logbook is None:
        assert net_itr == gen_itr == 0
        logbook = deap.tools.Logbook()
    rllib_trainer.workers.local_worker().set_policies_to_train([])
    # TODO: use "chapters" to hierarchicalize generator fitness, agent reward, and path length stats?
    logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) \
        + ["meanRew", "meanEvalRew", "meanPath"] + ["elapsed"]

    if len(init_batch) == 0:
        raise ValueError("``init_batch`` must not be empty.")

    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]

    # Evaluate all individuals
    invalid_ind = init_batch

    if rllib_eval:
        rllib_stats, qd_stats, logbook_stats = rllib_evaluate_worlds(
            trainer=rllib_trainer, worlds={i: ind for i, ind in enumerate(init_batch)}, idx_counter=idx_counter,
            quality_diversity=quality_diversity, oracle_policy=oracle_policy, net_itr=net_itr,
            start_time=start_time, evaluate_only=False)
        # assert len(rllib_stats) == 1

    else:
        qd_stats = toolbox.map(toolbox.evaluate, invalid_ind)

    update_individuals(invalid_ind, qd_stats)

    if len(invalid_ind) == 0:
        raise ValueError("No valid individual found !")

    # Update halloffame
    if halloffame is not None:
        halloffame.update(init_batch)

    # Store batch in container
    nb_updated = container.update(init_batch, issue_warning=show_warnings)
    # if nb_updated == 0:
    #     raise ValueError("No individual could be added to the container !")

    # Compile stats and update logs
    record = stats.compile(container) if stats else {}
    logbook_stats.update({
        'iteration': net_itr, 'containerSize': container.size_str(), 'evals': len(invalid_ind), 'nbUpdated': nb_updated,
        'elapsed': timer() - start_time, **record,
    })
    # logbook.record(iteration=net_itr, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated, elapsed=timer()-start_time, **record) #, meanAgentReward=rllib_stats["episode_reward_mean"], maxAgentReward=rllib_stats["episode_reward_max"], minAgentReward=rllib_stats["episode_reward_min"])
    logbook.record(**logbook_stats)
    if verbose:
        print(logbook.stream)
    net_itr += 1
    # Call callback function
    old_net_itr = net_itr
    net_itr = [net_itr]
    play_itr = [play_itr]
    if iteration_callback != None:
        iteration_callback(net_itr=net_itr, iteration=gen_itr, play_itr=play_itr, batch=init_batch, container=container, 
                           logbook=logbook, stats=stats)
    new_net_itr = net_itr[0]
    play_itr = play_itr[0] + (new_net_itr - old_net_itr)
    net_itr = new_net_itr

    done = False

    # Begin the generational process
    while not done:
        start_time = timer()
        # Select the next batch individuals
        batch = toolbox.select(container, batch_size)

        ## Vary the pool of individuals
        offspring = deap.algorithms.varAnd(batch, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        if rllib_eval:
            rllib_stats, qd_stats, logbook_stats = rllib_evaluate_worlds(net_itr=net_itr, evaluate_only=False,
                trainer=rllib_trainer, worlds={i: ind for i, ind in enumerate(invalid_ind)}, idx_counter=idx_counter,
                oracle_policy=oracle_policy, start_time=start_time, quality_diversity=quality_diversity)

        else:
            qd_stats = toolbox.map(toolbox.evaluate, invalid_ind)

        update_individuals(invalid_ind, qd_stats)

        # Replace the current population by the offspring
        nb_updated = container.update(offspring, issue_warning=show_warnings)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(container)

        # Append the current generation statistics to the logbook
        record = stats.compile(container) if stats else {}
        logbook_stats.update({
            'iteration': net_itr, 'containerSize': container.size_str(), 'evals': len(invalid_ind), 'nbUpdated': nb_updated,
            'elapsed': timer() - start_time, **record,
        })
        # logbook.record(iteration=net_itr, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated, elapsed=timer()-start_time , **record) #, meanAgentReward=rllib_stats["episode_reward_mean"], maxAgentReward=rllib_stats["episode_reward_max"], minAgentReward=rllib_stats["episode_reward_min"])
        logbook.record(**logbook_stats)
        if verbose:
            print(logbook.stream)
        net_itr += 1
        # Call callback function
        old_net_itr = net_itr
        net_itr = [net_itr]
        play_itr = [play_itr]
        if iteration_callback != None:
            iteration_callback(net_itr=net_itr, iteration=gen_itr, play_itr=play_itr, batch=batch, container=container, 
            logbook=logbook, stats=stats)
        # TODO: properly increment play_itr
        new_net_itr = net_itr[0]
        play_itr = play_itr[0] + (new_net_itr - old_net_itr)
        net_itr = new_net_itr
        print('play_itr', play_itr)
        gen_itr += 1
        done = play_itr >= niter

    return batch, logbook


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

