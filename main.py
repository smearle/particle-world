import argparse
import copy
import math
import os
import pickle
import random
from re import A
import sys
from functools import partial
from pdb import set_trace as TT
from time import sleep
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import ray
from deap import base
from deap import creator
from deap import tools
from minerl.herobraine.env_specs.obtain_specs import ObtainDiamond
from qdpy import containers
from qdpy.algorithms.deap import DEAPQDAlgorithm
# from qdpy.base import ParallelismManager
from qdpy.plots import plotGridSubplots
from timeit import default_timer as timer
from tqdm import tqdm
from args import init_parser

from envs import DirectedMazeEnv, MazeEnvForNCAgents, ParticleMazeEnv, TouchStone, eval_mazes, full_obs_test_mazes, \
    ghost_action_test_maze
from envs.wrappers import make_env
from generators.representations import TileFlipGenerator2D, TileFlipGenerator3D, SinCPPNGenerator, CPPN, Rastrigin, Hill
from qdpy_utils.utils import qdpy_save_archive
from qdpy_utils.evolve import qdRLlibEval
from qdpy_utils.callbacks import iteration_callback
from qdpy_utils.individuals import TileFlipIndividual2D, NCAIndividual, TileFlipIndividual3D
from ray.tune.registry import register_env
from rllib_utils.trainer import init_particle_trainer, train_players, toggle_exploration
from rllib_utils.eval_worlds import rllib_evaluate_worlds
from rllib_utils.utils import IdxCounter
from envs.maze.swarm import DirectedMazeSwarm, NeuralSwarm, MazeSwarm
from utils import compile_train_stats, get_experiment_name, qdpy_eval, update_individuals, load_config
from visualize import visualize_train_stats

seed = None
ndim = 2

width = 15
pg_delay = 50
n_nca_steps = 10
# n_sim_steps = 100
pg_width = 500
pg_scale = pg_width / width
# swarm_type = MemorySwarm

generator_phase = True  # Do we start by evolving generators, or training players?

# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
fitness_weight = -1.0
creator.create("FitnessMin", base.Fitness, weights=(-fitness_weight,))
creator.create("Individual", list, fitness=creator.FitnessMin, features=list)

fitness_domain = [(-np.inf, np.inf)]


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    # TODO: do this better?
    cfg = args
    cfg.qdpy_save_interval = 100
    cfg.archive_size = 100
    cfg.translated_observations = True

    # Whether to use rllib trainer to perform evaluations of evolved worlds.
    cfg.rllib_eval = True

    # This NCA-type model is meant to be used with full observations.
    if args.model == 'flood':
        assert args.fully_observable

    if args.load_config is not None:
        args = load_config(args, args.load_config)
    if args.oracle_policy:
        gen_phase_len = -1
        play_phase_len = 0
        n_pop = 1
    else:
        gen_phase_len = args.gen_phase_len
        play_phase_len = args.play_phase_len
        n_pop = 5
    n_policies = args.n_policies


    # Set the generator/individual, environment, and model class based on command-line arguments.
    generator_class = None

    env_is_minerl = False

    if args.environment_class == 'ParticleMazeEnv':
        n_envs_per_worker = 6

        # set the generator
        if args.generator_class == 'TileFlipIndividual':
            generator_class = TileFlipIndividual2D
        else: raise NotImplementedError

#       # set the environment, if specific to player-model
#       if args.model == 'nca':
#           environment_class = MazeEnvForNCAgents
#       else:

#       if args.fully_observable:
#           swarm_cls = MazeSwarm
#           environment_class = ParticleMazeEnv
#       else:
#           swarm_cls = DirectedMazeSwarm
#           environment_class = DirectedMazeEnv


#       # set the environment based on observation type
        if args.rotated_observations:
            swarm_cls = DirectedMazeSwarm
            environment_class = DirectedMazeEnv
        else:
            swarm_cls = MazeSwarm
            environment_class = ParticleMazeEnv

        env_config = {'width': width, 'swarm_cls': swarm_cls, 'n_policies': n_policies, 'n_pop': n_pop,
            'pg_width': pg_width, 'evaluate': args.evaluate, 
            'fully_observable': args.fully_observable, 'field_of_view': args.field_of_view, 'num_eval_envs': 1, 
            'target_reward': args.target_reward, 'rotated_observations': args.rotated_observations, 
            'translated_observations': args.translated_observations}

    elif args.environment_class == 'TouchStone':
        env_is_minerl = True
        n_envs_per_worker = 1

        if args.generator_class == 'TileFlipIndividual':
            generator_class = TileFlipIndividual3D
        else: raise NotImplementedError

        environment_class = TouchStone
        touchstone = TouchStone()

        # DEBUG (comparison with built-in minerl environment)
        # environment_class = ObtainDiamond
        # touchstone = ObtainDiamond(dense=True)

        touchstone.register()  # Need to register since this is a custom env.
        env_config = {}

    else:
        raise Exception(f"Unrecognized environment class: {args.environment_class}")

    env_config.update({'environment_class': environment_class, 'args': args})

    if (args.enjoy or args.evaluate) and args.render:
        n_rllib_envs = 1
    else:
        n_rllib_envs = args.n_rllib_workers * n_envs_per_worker if args.n_rllib_workers > 1 \
            else (1 if env_is_minerl else 12)

    args.n_rllib_envs = n_rllib_envs
    register_env('world_evolution_env', make_env)

    # Dummy env, to get various parameters defined inside the env, or for debugging.
    env = make_env(env_config)

    n_sim_steps = env.max_episode_steps
    unique_chans = env.unique_chans
    experiment_name = get_experiment_name(args)
    load = args.load
    total_play_itrs = 50000
    multi_proc = args.parallelismType != 'None'
    rllib_save_interval = 10
    save_dir = os.path.join(args.outputDir, experiment_name)
    cfg.save_dir = save_dir
    cfg.env_is_minerl = env_is_minerl
    cfg.log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']

    idx_counter = IdxCounter.options(name='idx_counter', max_concurrency=1).remote()


#   ### DEBUGGING THE ENVIRONMENT ###
#   if environment_class == ParticleMazeEnv:
#       env.set_worlds(eval_mazes)
#       obs = env.reset()
#       for i in range(1000):
#           env.render()
#           obs, _, _, _ = env.step({ak: env.action_spaces[0].sample() for ak in obs})

#   # elif environment_class == ObtainDiamond:
#   elif environment_class == TouchStone:
#       init_world = TileFlipIndividual3D(env.task.width-2, env.n_chan, unique_chans=env.unique_chans).discrete
#       # env.set_worlds({'world_0': init_world})
#       obs = env.reset()
#       done = False
#       for i in range(6000):
#           env.render()
#           action = env.action_space.sample()
#           sleep(0.5)
#           # action = env.action_space.nooop()
#           print(i, done)
#           if done:
#               TT()
#           obs, rew, done, info = env.step(action)

    # If we're doing world evolution, set this up.
    if not args.fixed_worlds:
        if args.quality_diversity:
            # If using QD, objective score is determined by the fitness of an additional policy.
            assert n_policies > 1
            # Fitness function is fixed for QD experiments.
            assert args.objective_function is None
            max_total_bins = 169  # so that we're guaranteed to have at least 12 non-impossible worlds (along the diagonal)
        else:
            # If not running a QD experiment, we must specify an objective function.
            assert args.objective_function is not None
            if args.objective_function == "contrastive":
                assert n_policies > 1
            max_total_bins = 1

        # generator = generator_class(width=width, n_chan=env.n_chan, unique_chans=unique_chans)

        # Don't need to know the dimension here since we'll simply call an individual's "mutate" method
        # initial_weights = generator.get_init_weights()
    
        n_emitters = 5
        batch_size = 30

        if args.quality_diversity:
            # Define grid in terms of fitness of all policies (save 1, for fitness)
            nb_features = n_policies - 1
        else:
            nb_features = 2  # The number of features to take into account in the container
        bins_per_dim = int(pow(max_total_bins, 1. / nb_features))
        nb_bins = (bins_per_dim,) * nb_features  # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with $max_total_bins^(1/nb_features)$ bins each

        # Specific to maze env: since each agent could be on the goal for at most, e.g. 99 steps given 100 max steps
        features_domain = [(0, env.max_episode_steps - 1)] * nb_features  # The domain (min/max values) of the features

        # If doing QD (i.e. there is more than one bin), then we have 1 individual per bin. Otherwise, we have 1 bin,
        # and all the individuals in the archive are in this bin.
        max_items_per_bin = 1 if max_total_bins != 1 else cfg.archive_size  # The number of items in each bin of the grid

    trainer = None if args.load and args.visualize else \
        init_particle_trainer(env, idx_counter=idx_counter, env_config=env_config, cfg=cfg)

    # env.set_policies([particle_trainer.get_policy(f'policy_{i}') for i in range(n_policies)], particle_trainer.config)
    # env.set_trainer(particle_trainer)

    if args.fixed_worlds:
        # train_worlds = {0: generator.landscape}
        # training_worlds = full_obs_test_mazes
        training_worlds = ghost_action_test_maze

    if args.load:
        if not args.fixed_worlds:
            fname = 'latest-0'
            # fname = f'latest-0' if args.loadIteration is not None else 'latest-0'
            with open(os.path.join(save_dir, f"{fname}.p"), "rb") as f:
                data = pickle.load(f)
            # with open(f'runs/{args.experimentName}/learn.pickle', 'rb') as f:
            #     supp_data = pickle.load(f)
            #     policies = supp_data['policies']
            # env.set_policies(policies)
            grid = data['container']
            gen_itr = data['gen_itr']
            play_itr = data['play_itr']
            net_itr = data['net_itr']
            logbook = data['logbook']

            # Produce plots and visualizations
            if args.visualize:

                # visualize current worlds
                gg = sorted(grid, key=lambda i: i.features)
                world_im_width = width * 10

                # if doing QD, render a grid of 1 world per cell in archive
                if args.quality_diversity:
                    nb_bins = grid.shape
                    world_im_width = width * 10
                    im_grid = np.zeros((world_im_width * nb_bins[0], world_im_width * nb_bins[1], 3))
                    for g in gg:
                        i, j = grid.index_grid(g.features)
                        env.set_world(g.discrete)
                        env.reset()
                        im = env.render(mode='rgb', pg_width=world_im_width, render_player=True)
                        im_grid[i * world_im_width: (i + 1) * world_im_width, j * world_im_width: (j + 1) * world_im_width] = im

                # otherwise, render a grid of elite levels
                else:
                    gg = sorted(gg, key=lambda ind: ind.fitness[0], reverse=True)
                    fits = [g.fitness[0] for g in gg]
                    max_fit = max(fits)
                    min_fit = min(fits)
                    assert nb_bins == (1, 1) == grid.shape
                    max_items_per_bin = len(grid)
                    n_world_width = math.ceil(math.sqrt(max_items_per_bin))
                    im_grid = np.zeros((world_im_width * n_world_width, world_im_width * n_world_width, 3))
                    for gi, g in enumerate(gg):
                        i, j = gi // n_world_width, gi % n_world_width
                        env.set_world(g.discrete)
                        env.reset()
                        im = env.render(mode='rgb', pg_width=world_im_width, render_player=True)
                        im_grid[j * world_im_width: (j + 1) * world_im_width, i * world_im_width: (i + 1) * world_im_width] = im

                        # To visualize ranking of fitnesses
                        im_grid[j * world_im_width: j * world_im_width + 7, int((i + 0.5) * world_im_width): int((i + 0.5) * world_im_width) + 7] = int(255 * (g.fitness[0] - min_fit) / (max_fit - min_fit))

                im_grid = im_grid.transpose(1, 0, 2)
                # im_grid = np.flip(im_grid, 0)
                # im_grid = np.flip(im_grid, 1)
                im_grid = Image.fromarray(im_grid.astype(np.uint8))
                im_grid.save(os.path.join(save_dir, "level_grid.png"))

                visualize_train_stats(save_dir, logbook, quality_diversity=args.quality_diversity)
                compile_train_stats(save_dir, logbook, net_itr, gen_itr, play_itr, quality_diversity=args.quality_diversity)

                if args.quality_diversity:
                    # save fitness qd grid
                    plot_path = os.path.join(save_dir, "performancesGrid.png")
                    plotGridSubplots(grid.quality_array[..., 0], plot_path, plt.get_cmap("magma"), features_domain,
                                    fitness_domain[0], nbTicks=None)
                    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
                sys.exit()

        if isinstance(env.swarms[0], NeuralSwarm) and cfg.rllib_eval:
            # if args.loadIteration == -1:
            with open(os.path.join(save_dir, 'model_checkpoint_path.txt'), 'r') as f:
                model_checkpoint_path = f.read()
            # else:
                # model_checkpoint_path = os.path.join(save_dir, f'checkpoint_{args.loadIteration:06d}/checkpoint-{args.loadIteration}')
            trainer.load_checkpoint(model_checkpoint_path)
            print(f'Loaded model checkopint at {model_checkpoint_path}')


        # Render and observe
        if args.enjoy:

            # TODO: support multiple fixed worlds
            if args.fixed_worlds:
                trainer.workers.local_worker().set_policies_to_train([])
                rllib_evaluate_worlds(trainer=trainer, worlds=training_worlds, calc_world_heuristics=False, 
                                      fixed_worlds=args.fixed_worlds, render=args.render)
                # particle_trainer.evaluation_workers.foreach_worker(
                #     lambda worker: worker.foreach_env(lambda env: env.set_landscape(generator.world)))
                # particle_trainer.evaluate()
                sys.exit()

            # We'll look at each world independently in our single env
            elites = sorted(grid, key=lambda ind: ind.fitness, reverse=True)
            worlds = [i for i in elites]
            if args.evaluate:
                worlds = eval_mazes
            for i, elite in enumerate(worlds):
                ret = rllib_evaluate_worlds(trainer=trainer, worlds={i: elite}, idx_counter=idx_counter,
                                            evaluate_only=True, calc_world_heuristics=False, render=args.render)
            sys.exit()

        # Evaluate
        if args.evaluate:
            worlds = eval_mazes
            for i in range(10):
                rllib_stats = trainer.evaluate()
                qd_stats = trainer.evaluation_workers.foreach_worker(lambda worker: worker.foreach_env(
                    lambda env: env.get_world_stats(evaluate=True, quality_diversity=args.quality_diversity)))
                qd_stats = [qds for worker_stats in qd_stats for qds in worker_stats]
                # rllib_stats, qd_stats, logbook_stats = rllib_evaluate_worlds(trainer=particle_trainer, worlds=worlds, idx_counter=idx_counter,
                                            # evaluate_only=True)
            
            # TODO: save (per-world) stats
            raise NotImplementedError
            sys.exit()

    # Train
    else:
        if not args.fixed_worlds:
            # Initialize these if not reloading
            gen_itr = 0
            play_itr = 0
            net_itr = 0
            # Create container
            grid = containers.Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain,
                                fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)
        logbook = None

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if args.fixed_worlds:
        train_players(0, 1000, trainer=trainer, worlds=training_worlds,
                      # TODO: initialize logbook even if not evolving worlds
                      logbook=None, cfg=cfg)
        sys.exit()

    # Algorithm parameters
    # dimension = len(initial_weights)  # The dimension of the target problem (i.e. genomes size)
    # assert (dimension >= 2)
    assert (nb_features >= 1)

    init_batch_size = cfg.n_rllib_envs  # The number of evaluations of the initial batch ('batch' = population)
    batch_size = cfg.n_rllib_envs  # The number of evaluations in each subsequent batch
    nb_iterations = total_play_itrs - play_itr  # The number of iterations (i.e. times where a new batch is evaluated)

    # Set the probability of mutating each value of a genome
#   if generator_cls == TileFlipGenerator:
#       mutation_pb = 0.03
#   elif generator_cls == SinCPPNGenerator:
#       mutation_pb = 0.03
#   else:
#       mutation_pb = 0.1
    eta = 20.0  # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
    ind_domain = (0, env.n_chan)  # The domain (min/max values) of the individual genomes
    # fitness_domain = [(0., 1.)]                # The domain (min/max values) of the fitness
    verbose = True
    show_warnings = False  # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
    log_base_path = args.outputDir if args.outputDir is not None else "."

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")

    # Create Toolbox
    toolbox = base.Toolbox()
    # toolbox.register("attr_float", random.uniform, ind_domain[0], ind_domain[1])
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, dimension)
    toolbox.register("individual", generator_class, width=env.width-2, n_chan=env.n_chan, save_gen_sequence=args.render,
                     unique_chans=env.unique_chans)
    # toolbox.register("individual", CPPNIndividual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("evaluate", illumination_rastrigin_normalised, nb_features = nb_features)
    # toolbox.register("evaluate", qdpy_eval, env, generator)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=ind_domain[0], up=ind_domain[1], eta=eta,
                    #  indpb=mutation_pb)
    toolbox.register("mutate", lambda individual: individual.mutate())
    toolbox.register("select", tools.selRandom)  # MAP-Elites = random selection on a grid container
    # toolbox.register("select", tools.selBest) # But you can also use all DEAP selection functions instead to create your own QD-algorithm

    # Create a dict storing all relevant infos
    results_infos = {}
    # results_infos['dimension'] = dimension
    results_infos['ind_domain'] = ind_domain
    results_infos['features_domain'] = features_domain
    results_infos['fitness_domain'] = fitness_domain
    results_infos['nb_bins'] = nb_bins
    results_infos['init_batch_size'] = init_batch_size
    results_infos['nb_iterations'] = nb_iterations
    results_infos['batch_size'] = batch_size
    # results_infos['mutation_pb'] = mutation_pb
    results_infos['eta'] = eta

    # Turn off exploration before evolving. Henceforth toggled before/after player training.
    # toggle_exploration(particle_trainer, explore=False, n_policies=n_policies)

    # with ParallelismManager(args.parallelismType, toolbox=toolbox) as pMgr:
    qd_algo = partial(qdRLlibEval, rllib_trainer=trainer, cfg=cfg, net_itr=net_itr,
                      gen_itr=gen_itr, logbook=logbook, play_itr=play_itr)
    # The staleness counter will be incremented whenever a generation of evolution does not result in any update to
    # the archive. (Crucially, it is mutable.)
    staleness_counter = [0]
    callback = partial(iteration_callback, toolbox=toolbox, cfg=cfg,
                        staleness_counter=staleness_counter, trainer=trainer)
    # Create a QD algorithm
    algo = DEAPQDAlgorithm(toolbox, grid, init_batch_siz=init_batch_size,
                            batch_size=batch_size, niter=nb_iterations,
                            verbose=verbose, show_warnings=show_warnings,
                            results_infos=results_infos, log_base_path=log_base_path, save_period=None,
                            iteration_filename=f'{experiment_name}' + '/latest-{}.p',
                            iteration_callback_fn=callback,
                            ea_fn=qd_algo,
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
