import argparse
import os
import pickle
import random
import time
from functools import partial
from pdb import set_trace as TT

import matplotlib.pyplot as plt
import numpy as np
import pygame
from deap import base
from deap import creator
from deap import tools
from qdpy import containers
import ray
from qdpy.algorithms.deap import DEAPQDAlgorithm
from qdpy.base import ParallelismManager
from qdpy.plots import plotGridSubplots
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from tqdm import tqdm

from env import ParticleSwarmEnv, ParticleMazeEnv, ParticleGym, ParticleGymRLlib
from generator import TileFlipFixedGenerator, SinCPPNGenerator, Rastrigin, Hill
from rllib_utils import init_particle_trainer
from swarm import eval_fit, NN, RLlibNN
from utils import infer, visualize, infer_elites, qdpy_eval, simulate, save

seed = None
ndim = 2
n_pop = 5
width = 100
pg_delay = 50
n_nca_steps = 10
n_sim_steps = 100
pg_width = 500
pg_scale = pg_width / width
# swarm_type = MemorySwarm
n_policies = 3


# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
fitness_weight = -1.0
creator.create("FitnessMin", base.Fitness, weights=(-fitness_weight,))
creator.create("Individual", list, fitness=creator.FitnessMin, features=list)

generator_phase = False  # Do we start by evolving generators, or training players?
gen_phase_len = 100
play_phase_len = 1000


def train_players(n_itr, trainer, landscapes):
    trainer.workers.foreach_worker(
        lambda worker: worker.foreach_env(lambda env: env.set_landscape(landscapes[0])))
    for i in range(n_itr):
        stats = trainer.train()
        keys = ['episode_reward_max', 'episode_reward_mean']
        print('\n'.join([f'Training iteration {i}'] + [f'{k}: {stats[k]}' for k in keys]))
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    # Also, in case you have trained a model outside of ray/RLlib and have created
    # an h5-file with weight values in it, e.g.
    # my_keras_model_trained_outside_rllib.save_weights("model.h5")
    # (see: https://keras.io/models/about-keras-models/)

    trainer.load_checkpoint()
    # ... you can load the h5-weights into your Trainer's Policy's ModelV2
    # (tf or torch) by doing:
    trainer.import_model("my_weights.h5")
    # NOTE: In order for this to work, your (custom) model needs to implement
    # the `import_from_h5` method.
    # See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
    # for detailed examples for tf- and torch trainers/models.


def callback(itr, player_trainer, landscapes):
    # if generator_phase:
    if itr % gen_phase_len:
        train_players(n_itr=play_phase_len, trainer=player_trainer, landscapes=landscapes)
    # else:
    #     if itr % play_phase_len:
            # pass


def run_qdpy():

    def iteration_callback(itr, batch, container, logbook):
        if itr % save_interval == 0:
            with open('learn.pickle', 'wb') as handle:
                dict = {
                    'policies': env.swarms,
                }
                pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_interval = 100
    if load:
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin, features=list)
        fname = f'latest-{args.loadIteration}' if args.loadIteration is not None else 'latest-0'
        with open(f"runs/{fname}.p", "rb") as f:
            data = pickle.load(f)
        with open('learn.pickle', 'rb') as f:
            supp_data = pickle.load(f)
            policies = supp_data['policies']
        env.set_policies(policies)
        grid = data['container']
        curr_iter = data['current_iteration']
        if enjoy:
            elites = sorted(grid, key=lambda ind: ind.fitness, reverse=True)
            elites = [np.array(i) for i in elites]
            infer_elites(env, generator, elites, pg_width, pg_delay)
    else:
        curr_iter = 0
    # Algorithm parameters
    dimension = len(initial_weights)  # The dimension of the target problem (i.e. genomes size)
    assert (dimension >= 2)
    assert (nb_features >= 1)
    bins_per_dim = int(pow(args.maxTotalBins, 1. / nb_features))

    init_batch_size = 10  # The number of evaluations of the initial batch ('batch' = population)
    batch_size = 10  # The number of evaluations in each subsequent batch
    nb_iterations = total_itrs - curr_iter  # The number of iterations (i.e. times where a new batch is evaluated)
    if generator_cls == TileFlipFixedGenerator:
        mutation_pb = 0.003  # The probability of mutating each value of a genome
    elif generator_cls == SinCPPNGenerator:
        mutation_pb = 0.03
    else:
        mutation_pb = 0.1
    eta = 20.0  # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
    max_items_per_bin = 1 if args.maxTotalBins != 1 else 100  # The number of items in each bin of the grid
    ind_domain = (0., 1.)  # The domain (min/max values) of the individual genomes
    # fitness_domain = [(0., 1.)]                # The domain (min/max values) of the fitness
    fitness_domain = [(-np.inf, np.inf)]
    verbose = True
    show_warnings = False  # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
    log_base_path = args.outputDir if args.outputDir is not None else "."

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")

    # Create Toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, ind_domain[0], ind_domain[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, dimension)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("evaluate", illumination_rastrigin_normalised, nb_features = nb_features)
    toolbox.register("evaluate", qdpy_eval, env, generator)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=ind_domain[0], up=ind_domain[1], eta=eta,
                     indpb=mutation_pb)
    toolbox.register("select", tools.selRandom)  # MAP-Elites = random selection on a grid container
    # toolbox.register("select", tools.selBest) # But you can also use all DEAP selection functions instead to create your own QD-algorithm

    # Create a dict storing all relevant infos
    results_infos = {}
    results_infos['dimension'] = dimension
    results_infos['ind_domain'] = ind_domain
    results_infos['features_domain'] = features_domain
    results_infos['fitness_domain'] = fitness_domain
    results_infos['nb_bins'] = nb_bins
    results_infos['init_batch_size'] = init_batch_size
    results_infos['nb_iterations'] = nb_iterations
    results_infos['batch_size'] = batch_size
    results_infos['mutation_pb'] = mutation_pb
    results_infos['eta'] = eta
    if not load:
        # Create container
        grid = containers.Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain,
                               fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)
    with ParallelismManager(args.parallelismType, toolbox=toolbox) as pMgr:
        # Create a QD algorithm
        algo = DEAPQDAlgorithm(pMgr.toolbox, grid, init_batch_siz=init_batch_size,
                               batch_size=batch_size, niter=nb_iterations,
                               verbose=verbose, show_warnings=show_warnings,
                               results_infos=results_infos, log_base_path=log_base_path, save_period=save_interval,
                               iteration_filename='latest-{}.p',
                               iteration_callback_fn=iteration_callback,
        )
        # Run the illumination process !
        algo.run(init_batch_size=init_batch_size)
    # Print results info
    print(f"Total elapsed: {algo.total_elapsed}\n")
    print(grid.summary())
    # print("Best ever fitness: ", container.best_fitness)
    # print("Best ever ind: ", container.best)
    # print("%s filled bins in the grid" % (grid.size_str()))
    ##print("Solutions found for bins: ", grid.solutions)
    # print("Performances grid: ", grid.fitness)
    # print("Features grid: ", grid.features)

    # Create plot of the performance grid
    plot_path = os.path.join(log_base_path, "performancesGrid.pdf")
    plotGridSubplots(grid.quality_array[..., 0], plot_path, plt.get_cmap("nipy_spectral_r"), features_domain,
                     fitness_domain[0], nbTicks=None)
    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
    print("All results are available in the '%s' pickle file." % algo.final_filename)

def run_pyribs():
    if load:
        with open('learn.pickle', 'rb') as handle:
            dict = pickle.load(handle)
            archive = dict['archive']
            emitters = dict['emitters']
            optimizer = dict['optimizer']
            stats = dict['stats']
            policies = dict['policies']
        visualize(archive)
        if enjoy:
            env.set_policies(policies)
            infer(env, generator, archive, pg_width, pg_delay)
    else:
        stats = {
            'n_iter': 0,
            'obj_max': [],
            'obj_mean': [],
        }
        archive = GridArchive(
            dims=nb_bins,
            # dims=[100,100],
            ranges=features_domain,
            seed=seed,
        )
        seeds = ([None] * n_emitters
                 if seed is None else [seed + i for i in range(n_emitters)])
        if args.maxTotalBins == 1:
            n_opt_emitters = len(seeds)
            n_imp_emitters = 0
        else:
            n_opt_emitters = 0
            n_imp_emitters = len(seeds)
        emitters = [
                       # OptimizingEmitter(
                       ImprovementEmitter(
                           archive,
                           initial_weights.flatten(),
                           sigma0=.1,
                           batch_size=batch_size,
                           seed=s,
                       ) for s in seeds[:n_imp_emitters]] + \
                   [
                       OptimizingEmitter(
                           # ImprovementEmitter(
                           archive,
                           initial_weights.flatten(),
                           sigma0=1,
                           batch_size=batch_size,
                           seed=s,
                       ) for s in seeds[n_imp_emitters:]
                   ]
        optimizer = Optimizer(archive, emitters)

    start_time = time.time()
    if multi_proc:
        from ray.util.multiprocessing import Pool

        # Setup for parallel processing.
        pool = Pool()
        generators = [generator_cls(width=width) for _ in range(batch_size * n_emitters)]
        # generators = [generator for _ in range(batch_size * n_emitters)]
        envs = [ParticleGym(width=width, n_pop=n_pop, n_policies=n_policies) for _ in
                range(batch_size * n_emitters)]
        # envs = [env for _ in range(batch_size * n_emitters)]
    for itr in tqdm(range(1, total_itrs + 1)):
        sols = optimizer.ask()
        objs = []
        bcs = []
        if multi_proc:
            sim = partial(simulate, render=False, n_steps=n_sim_steps, n_eps=1)
            ret = pool.starmap(sim, zip(generators, sols, envs))
            objs, bcs = zip(*ret)
        else:
            for i, sol in enumerate(sols):
                obj, bc = simulate(generator, sol, env, n_steps=n_sim_steps, n_eps=1, screen=None, pg_delay=pg_delay, pg_scale=pg_scale)
                objs.append(obj)
                bcs.append(bc)
        optimizer.tell(objs, bcs)
        if itr % 1 == 0:
            elapsed_time = time.time() - start_time
            print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
            print(f"  - Archive Size: {len(archive)}")
            print(f"  - Max Score: {archive.stats.obj_max}")
            print(f"  - Mean Score: {archive.stats.obj_mean}")
        if itr % save_interval == 0:
            save(archive, optimizer, emitters, stats, policies=env.swarms)
            visualize(archive)
        stats['n_iter'] += 1
        stats['obj_max'].append(archive.stats.obj_max)
        stats['obj_mean'].append(archive.stats.obj_mean)
    visualize(archive)
    # infer(archive)


if __name__ == '__main__':
    # generator_cls = Hill
    generator_cls = Rastrigin
    # generator_cls = TileFlipFixedGenerator
    # generator_cls = NCAGenerator
    # generator_cls = SinCPPNGenerator
    generator = generator_cls(width=width)
    # env = ParticleSwarmEnv(width=width, n_policies=n_policies, n_pop=n_pop)
    # env = ParticleMazeEnv(width=width, n_policies=n_policies, n_pop=n_pop)
    env = ParticleGym(width=width, n_policies=n_policies, n_pop=n_pop, max_steps=n_sim_steps, pg_width=pg_width)

    initial_weights = generator.get_init_weights()

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-e', '--enjoy', action='store_true')
    # parser.add_argument('-seq', '--sequential', help='not parallel', action='store_true')
    parser.add_argument('--maxTotalBins', type=int, default=10000, help="Maximum number of bins in the grid")
    parser.add_argument('-p', '--parallelismType', type=str, default='None',
                        help="Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default='./runs', help="Path of the output log files")
    parser.add_argument('-li', '--loadIteration', default=None)
    parser.add_argument('-a', '--algo', default='cmame')
    parser.add_argument('-exp', '--experimentName', default='test')
    args = parser.parse_args()
    load = args.load
    enjoy = args.enjoy
    total_itrs = 50000
    multi_proc = args.parallelismType != 'None'
    n_emitters = 5
    batch_size = 30
    nb_features = 2  # The number of features to take into account in the container
    bins_per_dim = int(pow(args.maxTotalBins, 1. / nb_features))
    nb_bins = (
                  bins_per_dim,) * nb_features  # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with $maxTotalBins^(1/nb_features)$ bins each
    save_interval = 10
    features_domain = [(0., 1.)] * nb_features  # The domain (min/max values) of the features

    particle_trainer = init_particle_trainer(env)

    if generator_phase:
        if args.algo == 'me':
            run_qdpy()
        elif args.algo == 'cmame':
            run_pyribs()
        elif args.algo == 'ppo':
            # TODO: train generators?
            pass
    else:
        # TODO: evolve players?
        train_players(1000, trainer=particle_trainer, landscapes=[generator.landscape])
        # train_players(play_phase_len, particle_trainer)
