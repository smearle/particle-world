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


def rllib_evaluate_worlds(trainer, worlds):
    idxs = np.random.permutation(list(worlds.keys()))
    workers = trainer.workers

    # workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.set_world(worlds={
    #     idxs[0]: worlds[idxs.pop(0)]
    # })))
    world_id = 0
    fitnesses = {}
    all_stats = []

    while world_id == 0 or world_id < len(worlds) - 1:
        envs = []
        for (wrk_i, worker) in enumerate([workers.local_worker()] + workers.remote_workers()):
            if wrk_i == 0:  # only ever 1 local worker
                envs += worker.foreach_env(lambda env: env)
            else:
                # world_i = idxs[wrk_i % len(idxs)]
                # worker.foreach_env.remote(lambda env: env.set_world(worlds={world_i: worlds[world_i]}))
                envs += ray.get(worker.foreach_env.remote(lambda env: env))
        if len(worlds) < len(envs):
            idxs = idxs * len(envs)

        [env.set_world(worlds={i: worlds[i]}) for i, env in zip(idxs[world_id:], envs)]
        world_id += len(envs)


        stats = trainer.train()
        all_stats.append(stats)
        # new_fitnesses = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.get_fitness()))
        # new_fitnesses = [fit for worker_fits in new_fitnesses for fit in worker_fits]
        # [fitnesses.update(nf) for nf in new_fitnesses]


        for (wrk_i, worker) in enumerate([workers.local_worker()] + workers.remote_workers()):
            world_i = idxs[wrk_i % len(idxs)]
            if wrk_i == 0:  # only ever 1 local worker
                new_fitnesses = worker.foreach_env(lambda env: env.get_fitness())
            else:
                new_fitnesses = worker.foreach_env.remote(lambda env: env.get_fitness())
            for nf in new_fitnesses:
                fitnesses.update(nf)

    return all_stats, fitnesses


def qdRLlibEval(rllib_trainer, init_batch, toolbox, container, batch_size, niter, cxpb = 0.0, mutpb = 1.0, stats = None, halloffame = None, verbose = False, show_warnings = False, start_time = None, iteration_callback = None):
    """The simplest QD algorithm using DEAP, modified to evaluate generated worlds inside an RLlib trainer object.
    :param rllib_trainer: RLlib trainer object.
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
    :returns: The final batch
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    TODO
    """
    rllib_trainer.workers.local_worker().set_policies_to_train([])
    if start_time == None:
        start_time = timer()
    logbook = deap.tools.Logbook()
    logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + ["elapsed"]

    if len(init_batch) == 0:
        raise ValueError("``init_batch`` must not be empty.")

    # Evaluate the individuals with an invalid fitness
    # invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
    invalid_ind = init_batch
    rllib_stats, fitnesses = rllib_evaluate_worlds(rllib_trainer, {i: ind for i, ind in enumerate(init_batch)})
    fitnesses = [fitnesses[k] for k in range(len(fitnesses))]

    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0]
        ind.features = fit[1]

    if len(invalid_ind) == 0:
        raise ValueError("No valid individual found !")

    # Update halloffame
    if halloffame is not None:
        halloffame.update(init_batch)

    # Store batch in container
    nb_updated = container.update(init_batch, issue_warning=show_warnings)
    if nb_updated == 0:
        raise ValueError("No individual could be added to the container !")

    # Compile stats and update logs
    record = stats.compile(container) if stats else {}
    logbook.record(iteration=0, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated, elapsed=timer()-start_time, **record)
    if verbose:
        print(logbook.stream)
    # Call callback function
    if iteration_callback != None:
        iteration_callback(0, init_batch, container, logbook)

    # Begin the generational process
    for i in range(1, niter + 1):
        start_time = timer()
        # Select the next batch individuals
        batch = toolbox.select(container, batch_size)

        ## Vary the pool of individuals
        offspring = deap.algorithms.varAnd(batch, toolbox, cxpb, mutpb)
        #offspring = []
        #for o in batch:
        #    newO = toolbox.clone(o)
        #    ind, = toolbox.mutate(newO)
        #    del ind.fitness.values
        #    offspring.append(ind)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        rllib_fitnesses = rllib_evaluate_worlds(rllib_trainer, {i: ind for i, ind in enumerate(invalid_ind)})
        fitnesses = [fitnesses[k] for k in range(len(fitnesses))]

        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.features = fit[1]

        # Replace the current population by the offspring
        nb_updated = container.update(offspring, issue_warning=show_warnings)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(container)

        # Append the current generation statistics to the logbook
        record = stats.compile(container) if stats else {}
        logbook.record(iteration=i, containerSize=container.size_str(), evals=len(invalid_ind), nbUpdated=nb_updated, elapsed=timer()-start_time, **record)
        if verbose:
            print(logbook.stream)
        # Call callback function
        if iteration_callback != None:
            iteration_callback(i, batch, container, logbook)

    return batch, logbook


