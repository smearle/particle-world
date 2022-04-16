import json
import math
import os
import pickle
import random
from re import A
import shutil
import sys
from functools import partial
from pdb import set_trace as TT
from time import sleep
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from deap import base
from deap import creator
from deap import tools
from qdpy import containers
from qdpy.algorithms.deap import DEAPQDAlgorithm
# from qdpy.base import ParallelismManager
from qdpy.plots import plotGridSubplots
from timeit import default_timer as timer
from tqdm import tqdm
from args import init_parser

from envs import DirectedMazeEnv, MazeEnvForNCAgents, ParticleMazeEnv, eval_mazes, full_obs_test_mazes, \
    ghost_action_test_maze
from envs.wrappers import make_env
from generators.representations import TileFlipGenerator2D, TileFlipGenerator3D, SinCPPNGenerator, CPPN, Rastrigin, Hill
from evo.utils import compute_archive_world_heuristics, save
from evo.evolve import WorldEvolver
from evo.individuals import TileFlipIndividual2D, NCAIndividual, TileFlipIndividual3D, clone
from rl.trainer import init_trainer, sync_player_policies, toggle_train_player, train_players, toggle_exploration
from rl.eval_worlds import evaluate_worlds
from rl.utils import IdxCounter, set_worlds
from envs.maze.swarm import DirectedMazeSwarm, NeuralSwarm, MazeSwarm
from utils import compile_train_stats, get_experiment_name, load_config, log
from visualize import visualize_train_stats, visualize_archive

seed = None
ndim = 2

generator_phase = True  # Do we start by evolving generators, or training players?

# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
fitness_weight = -1.0
creator.create("FitnessMin", base.Fitness, weights=(-fitness_weight,))
creator.create("Individual", list, fitness=creator.FitnessMin, features=list)


def get_done_gen_phase(world_evolver, gen_itr, cfg):

    if gen_itr == 0:
        return False

    if cfg.gen_phase_len != -1 and gen_itr % cfg.gen_phase_len == 0:
        return True

    if world_evolver.stale_generators:
        world_evolver.reset_staleness()
        return True

#   if world_evolver.optimal_generators:
#       return True


def get_done_play_phase(play_itr, cfg):
    return play_itr % cfg.play_phase_len == 0


if __name__ == '__main__':
    parser = init_parser()
    cfg = parser.parse_args()

    if cfg.load_config is not None:
        cfg = load_config(cfg, cfg.load_config)

    # TODO: ClArgsConfig class.
    if cfg.model == 'flood':
        cfg.rotated_observations = False
        cfg.translated_observations = False
    else:
        cfg.translated_observations = True
    cfg.fitness_domain = [(-np.inf, np.inf)]
    cfg.width = 15
    pg_delay = 50  # Delay for rendering in pygame (ms?). Probably not needed!
    n_nca_steps = 10
    # n_sim_steps = 100
    pg_width = 500
    pg_scale = pg_width / cfg.width
    cfg.save_interval = 100
    cfg.archive_size = 1024 if not cfg.quality_diversity else 2500
    # cfg.log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']

    # Number of episodes for player training = n_rllib_envs / n_rllib_workers = n_envs_per_worker (since we use local
    # worker for training simulation so as not to waste CPUs).
    cfg.n_envs_per_worker = 40
    cfg.n_rllib_envs = cfg.n_rllib_workers * cfg.n_envs_per_worker  # Note that this effectively sets n_envs_per_worker to 40.
    # cfg.n_rllib_envs = 400
    cfg.n_eps_on_train = cfg.n_rllib_envs
    cfg.world_batch_size = cfg.n_eps_on_train 

    # Whether to run rounds of player-training and generator-evolution in parallel.
    cfg.parallel_gen_play = True

    n_workers = (1 if cfg.n_rllib_workers == 0 else cfg.n_rllib_workers)

    # We must have the same number of envs per worker, and want to meet our target number of envs exactly.
    assert cfg.n_rllib_envs % n_workers == 0, \
        f"n_rllib_envs ({cfg.n_rllib_envs}) must be divisible by n_workers ({n_workers})"

    # We don't want any wasted episodes when we call rllib_evaluate_worlds() to evaluate worlds.
    assert cfg.world_batch_size % cfg.n_eps_on_train == 0, \
        f"world_batch_size ({cfg.world_batch_size}) must be divisible by n_eps_on_train ({cfg.n_eps_on_train})"

    # We don't want any wasted episodes when we call train() to evaluate worlds.
    assert cfg.n_eps_on_train % cfg.n_rllib_envs == 0, \
        f"n_eps_on_train ({cfg.n_eps_on_train}) must be divisible by n_rllib_envs ({cfg.n_rllib_envs})"

    # Whether to use rllib trainer to perform evaluations of evolved worlds.
    cfg.rllib_eval = True

    # This NCA-type model is meant to be used with full observations.
    if cfg.model == 'flood':
        assert cfg.fully_observable

    if cfg.oracle_policy:
        gen_phase_len = -1
        play_phase_len = 0
        n_pop = 1
    else:
        gen_phase_len = cfg.gen_phase_len
        play_phase_len = cfg.play_phase_len
        n_pop = 5
    n_policies = cfg.n_policies

    # Set the generator/individual, environment, and model class based on command-line arguments.
    generator_class = None

    env_is_minerl = False

    if cfg.environment_class == 'ParticleMazeEnv':
        # n_envs_per_worker = 6

        # set the generator
        if cfg.generator_class == 'TileFlipIndividual':
            generator_class = TileFlipIndividual2D

        elif cfg.generator_class == 'NCA':
            generator_class = NCAIndividual

        else:
            raise NotImplementedError

#       # set the environment, if specific to player-model
#       if cfg.model == 'nca':
#           environment_class = MazeEnvForNCAgents

        # set the environment based on observation type
        if cfg.rotated_observations:
            swarm_cls = DirectedMazeSwarm
            environment_class = DirectedMazeEnv
        else:
            swarm_cls = MazeSwarm
            environment_class = ParticleMazeEnv

        env_config = {'width': cfg.width, 'swarm_cls': swarm_cls, 'n_policies': n_policies, 'n_pop': n_pop,
                      'pg_width': pg_width, 'evaluate': cfg.evaluate,
                      'fully_observable': cfg.fully_observable, 'field_of_view': cfg.field_of_view, 'num_eval_envs': 1,
                      'target_reward': cfg.target_reward, 'rotated_observations': cfg.rotated_observations,
                      'translated_observations': cfg.translated_observations}

    elif cfg.environment_class == 'TouchStone':
        env_is_minerl = True
        # from minerl.herobraine.env_specs.obtain_specs import ObtainDiamond
        from envs.minecraft.touchstone import TouchStone

        # n_envs_per_worker = 1

        if cfg.generator_class == 'TileFlipIndividual':
            generator_class = TileFlipIndividual3D
        else:
            raise NotImplementedError

        environment_class = TouchStone
        touchstone = TouchStone()

        # DEBUG (comparison with built-in minerl environment)
        # environment_class = ObtainDiamond
        # touchstone = ObtainDiamond(dense=True)

        touchstone.register()  # Need to register since this is a custom env.
        env_config = {}

    else:
        raise Exception(f"Unrecognized environment class: {cfg.environment_class}")

    env_config.update({'environment_class': environment_class, 'cfg': cfg})

    if (cfg.evaluate or cfg.enjoy) and cfg.render:
        cfg.n_rllib_envs = 1
    else:
        # cfg.n_rllib_envs = cfg.n_rllib_workers * n_envs_per_worker if cfg.n_rllib_workers > 1 \
            # else (1 if env_is_minerl else n_envs_per_worker)

        # NOTE: this is also the number of episodes we will train players on.
        assert cfg.n_envs_per_worker == cfg.n_rllib_envs // n_workers

    register_env('world_evolution_env', make_env)

    experiment_name = get_experiment_name(cfg)
    save_dir = os.path.join(cfg.outputDir, experiment_name)
    cfg.save_dir = save_dir
    cfg.env_is_minerl = env_is_minerl

    # Dummy env, to get various parameters defined inside the env, or for debugging.
    env = make_env(env_config)

    n_sim_steps = env.max_episode_steps
    unique_chans = env.unique_chans

    cfg.total_play_itrs = 50000
    multi_proc = cfg.parallelismType != 'None'
    rllib_save_interval = 10

    idx_counter = IdxCounter.options(name='idx_counter', max_concurrency=1).remote()

    #   ### DEBUGGING THE ENVIRONMENT ###
    #   if environment_class == ParticleMazeEnv:
    #       env.queue_worlds(eval_mazes)
    #       obs = env.reset()
    #       for i in range(1000):
    #           env.render()
    #           obs, _, _, _ = env.step({ak: env.action_spaces[0].sample() for ak in obs})

    #   # elif environment_class == ObtainDiamond:
    #   elif environment_class == TouchStone:
    #       init_world = TileFlipIndividual3D(env.task.width-2, env.n_chan, unique_chans=env.unique_chans).discrete
    #       # env.queue_worlds({'world_0': init_world})
    #       obs = env.reset()
    #       done = False
    #       for i in range(6000):
    #           env.render()
    #           action = env.action_space.sample()
    #           sleep(0.5)
    #           # action = env.action_space.nooop()
    #           print(i, done)
    #           if done:
    #               # Enter debug console to look around.
    #               pass
    #           obs, rew, done, info = env.step(action)

    # If we're doing world evolution, set this up.
    if not cfg.fixed_worlds:
        if cfg.quality_diversity:
            # If using QD, objective score is determined by the fitness of an additional policy.
            assert n_policies > 1
            # Fitness function is fixed for QD experiments.
            assert cfg.objective_function is None

            # TODO: We might want this to be a bit bigger than the intended archive size, assuming QD will have trouble 
            #  filling certain bins.
            max_total_bins = cfg.archive_size

        else:
            # If not running a QD experiment, we must specify an objective function.
            assert cfg.objective_function is not None
            if cfg.objective_function == "contrastive":
                assert n_policies > 1
            max_total_bins = 1

        # generator = generator_class(width=width, n_chan=env.n_chan, unique_chans=unique_chans)

        # Don't need to know the dimension here since we'll simply call an individual's "mutate" method
        # initial_weights = generator.get_init_weights()

        if cfg.quality_diversity:
            # Define grid in terms of fitness of all policies (save 1, for fitness)
            nb_features = n_policies - 1
        else:
            nb_features = 2  # The number of features to take into account in the container
        bins_per_dim = int(pow(max_total_bins, 1. / nb_features))

        # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with 
        # $max_total_bins^(1/nb_features)$ bins each
        nb_bins = (bins_per_dim,) * nb_features 

        # The domain (min/max values) of the features. Assume we are using mean policy rewards as diversity measures.
        features_domain = [(env.min_reward, env.max_reward)] * nb_features  

        # If doing QD (i.e. there is more than one bin), then we have 1 individual per bin. Otherwise, we have 1 bin,
        # and all the individuals in the archive are in this bin.
        max_items_per_bin = 1 if max_total_bins != 1 else cfg.archive_size  # The number of items in each bin of the grid

    # TODO: serializing these trainers doesn't work, so we need to use a single trainer, either by co-opting some eval
    #   workers for evo-eval, or creating a separate WorkerSet for this purpose.
#   if not cfg.parallel_gen_play:
    trainer = None if cfg.load and cfg.visualize else \
        init_trainer(env, idx_counter=idx_counter, env_config=env_config, cfg=cfg)
#   else:
#       gen_trainer = init_trainer(env, idx_counter=idx_counter, env_config=env_config, cfg=cfg, gen_only=True)
#       play_trainer = init_trainer(env, idx_counter=idx_counter, env_config=env_config, cfg=cfg, play_only=True)
#       # For enjoy/eval.
#       trainer = gen_trainer

    # env.set_policies([particle_trainer.get_policy(f'policy_{i}') for i in range(n_policies)], particle_trainer.config)
    # env.set_trainer(particle_trainer)

    # If training on fixed worlds, select the desired training set.
    if cfg.fixed_worlds:
        # train_worlds = {0: generator.landscape}
        training_worlds = full_obs_test_mazes
        # training_worlds = ghost_action_test_maze

    # If doing co-learning, do any setup that is necessary regardless of whether we're reloading or starting anew.
    else:
        # Default stats to be performed on worlds in the archive.
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        # TODO: use similar logic to compute path-length stats on worlds? (But only once per world-generation phase.)

    if cfg.load:
        # Load the world archive, logbook, and iteration counters.
        fname = 'latest-0.p'
        loadfile_name = os.path.join(save_dir, fname)

        if not os.path.isfile(loadfile_name):
            raise Exception(f'{loadfile_name} is not a file, cannot load.')

        with open(os.path.join(loadfile_name), "rb") as f:
            data = pickle.load(f)

        archive = data['archive']
        gen_itr = data['gen_itr']
        play_itr = data['play_itr']
        net_itr = data['net_itr']
        logbook = data['logbook']

        # Produce plots and visualizations of evolved worlds.
        if not cfg.fixed_worlds:
            if cfg.visualize:

                visualize_train_stats(cfg.save_dir, logbook, quality_diversity=cfg.quality_diversity)
                compile_train_stats(save_dir, logbook, net_itr, gen_itr, play_itr,
                                    quality_diversity=cfg.quality_diversity)
                visualize_archive(cfg, env, archive)
                if cfg.quality_diversity:
                                # save fitness qd grid
                    plot_path = os.path.join(cfg.save_dir, "performancesGrid.png")
                    plotGridSubplots(archive.quality_array[..., 0], plot_path, plt.get_cmap("magma"), features_domain,
                                                cfg.fitness_domain[0], nbTicks=None)
                    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
                print('Done visualizing, goodbye!')
                sys.exit()

        if isinstance(env.swarms[0], NeuralSwarm) and cfg.rllib_eval:
            # if cfg.loadIteration == -1:
            with open(os.path.join(save_dir, 'model_checkpoint_path.txt'), 'r') as f:
                model_checkpoint_path = f.read()
            # else:
            # model_checkpoint_path = os.path.join(save_dir, f'checkpoint_{cfg.loadIteration:06d}/checkpoint-{cfg.loadIteration}')
            trainer.load_checkpoint(model_checkpoint_path)
            print(f'Loaded model checkpoint at {model_checkpoint_path}')

        # Render and observe
        if cfg.enjoy:

            # TODO: support multiple fixed worlds
            # We'll look at each world independently in our single env
            if cfg.fixed_worlds:
                worlds = list(full_obs_test_mazes.values())
#               evaluate_worlds(trainer=trainer, worlds=training_worlds,
#                                     fixed_worlds=cfg.fixed_worlds, render=cfg.render)
                # particle_trainer.evaluation_workers.foreach_worker(
                #     lambda worker: worker.foreach_env(lambda env: env.set_landscape(generator.world)))
                # particle_trainer.evaluate()

            else:
                elites = sorted(archive, key=lambda ind: ind.fitness, reverse=True)
                worlds = [i for i in elites]
                # FIXME: Hack: avoid skipping world 0. Something about the way eval calls reset at step 0 of episode 0?
                worlds = [worlds[0]] + worlds

            for i, elite in enumerate(worlds):
                print(f"Evaluating world {i}")
                set_worlds({i: elite}, trainer.evaluation_workers, idx_counter, cfg)
                trainer.evaluate()
#               ret = evaluate_worlds(trainer=trainer, worlds={i: elite}, idx_counter=idx_counter,
#                                           evaluate_only=True, cfg=cfg)
            print('Done enjoying, goodbye!')
            sys.exit()

        # Evaluate
        if cfg.evaluate:
            worlds = eval_mazes
            net_world_stats = {eval_maze: {f'policy_{i}': {'pct_wins': [], 'mean_rewards': []} \
                                           for i in range(n_policies)} for eval_maze in worlds}

            # How many trials on each world? Kind of, minus garbage resets, and assuming evaluation_num_episodes == len(eval_worlds).
            for _ in range(10):
                rllib_stats = trainer.evaluate()
                qd_stats = trainer.evaluation_workers.foreach_worker(lambda worker: worker.foreach_env(
                    lambda env: env.get_world_stats(evaluate=True, quality_diversity=cfg.quality_diversity)))

                # Flattening the list of lists of stats (outer lists are per-worker, inner lists are per-environment).
                qd_stats = [qds for worker_stats in qd_stats for qds in worker_stats]

                # rllib_stats, qd_stats, logbook_stats = rllib_evaluate_worlds(trainer=particle_trainer, worlds=worlds, idx_counter=idx_counter,
                # evaluate_only=True)

                for env_stats in qd_stats:
                    for world_stats in env_stats:
                        #                       if world_stats['n_steps'] != env.max_episode_steps:
                        #                           # There will be one additional stats dict (the last one in the list), that was created on the last reset.
                        #                           # We will ignore it
                        #                           assert world_stats['n_steps'] == 0
                        #                           continue

                        world_key = world_stats['world_key']

                        for j in range(n_policies):
                            policy_key = f'policy_{j}'
                            net_world_stats[world_key][policy_key]['mean_rewards'].append(
                                world_stats[policy_key]['mean_reward'])
                            net_world_stats[world_key][policy_key]['pct_wins'].append(
                                world_stats[policy_key]['pct_win'])

            eval_stats_fname = os.path.join(save_dir, 'eval_stats.json')
            with open(os.path.join(save_dir, 'eval_stats.json'), 'w') as f:
                json.dump(net_world_stats, f, indent=4)
            # We're done evaluating. Exit now so we don't start training.
            print(f"\nEvaluation stats saved to '{eval_stats_fname}'. Exiting.")
            sys.exit()

    # If not loading, set up world-evolution stuff.
    else:
        # Initialize these counters if not reloading.
        gen_itr = 0
        play_itr = 0
        net_itr = 0
        
        # If we're not loading, and not overwriting, and the relevant ``save_dir`` exists, then raise Exception.
        if not cfg.overwrite:
            if os.path.exists(save_dir):
                # FIXME: even when we are running new experiment, the directoy already exists at this point. Why?
                # Ahhh it's TBXLogger? Can just skip this check?
                raise Exception(f"The save directory '{save_dir}' already exists. Use --overwrite to overwrite it.")

#       # Remove the save directory if it exists and we are overwriting.
#       else:
#           print(f"Overwriting save directory '{save_dir}'.")
#           shutil.rmtree(save_dir)

#       # Create the new save directory.
#       os.mkdir(save_dir)

        # If not loading, do some initial setup for world evolution if applicable.
        if not cfg.fixed_worlds:
            # Create empty container.
            archive = containers.Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=cfg.fitness_domain,
                                   fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)

            # TODO: use this when ``fixed_worlds``...?
            logbook = tools.Logbook()
            # TODO: use "chapters" to hierarchicalize generator fitness, agent reward, and path length stats?
            # NOTE: [avg, std, min, max] match the headers in deap.DEAPQDAlgorithm._init_stats. Could do this more cleanly.
            logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + stats.fields + ["meanRew", \
                "meanEvalRew", "meanPath", "maxPath", "elapsed"]

    # Perform setup that must occur whether reloading or not, but which will need to occur after reloading if reloading.
    if not cfg.fixed_worlds:
        # Evolutionary algorithm parameters
        assert (nb_features >= 1)

        nb_iterations = cfg.total_play_itrs - play_itr  # The number of iterations (i.e. times where a new batch is evaluated)

        eta = 20.0  # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
        ind_domain = (0, env.n_chan)  # The domain (min/max values) of the individual genomes
        verbose = True
        show_warnings = False  # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
        log_base_path = cfg.outputDir if cfg.outputDir is not None else "."
        # Create Toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", generator_class, width=env.width - 2, n_chan=env.n_chan,
                        save_gen_sequence=cfg.render,
                        unique_chans=env.unique_chans)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("clone", clone)
        toolbox.register("mutate", lambda individual: individual.mutate())
        toolbox.register("select", tools.selRandom)  # MAP-Elites = random selection on a grid container
        # toolbox.register("select", tools.selBest) # But you can also use all DEAP selection functions instead to create your own QD-algorithm

        # Create a dict storing all relevant infos
        results_infos = {'ind_domain': ind_domain, 'features_domain': features_domain, 'fitness_domain': cfg.fitness_domain,
                        'nb_bins': nb_bins, 'init_batch_size': cfg.world_batch_size, 'nb_iterations': nb_iterations,
                        'batch_size': cfg.world_batch_size, 'eta': eta}

        # Create a QD algorithm
        world_evolver = WorldEvolver(toolbox=toolbox, container=archive, init_batch_siz=cfg.world_batch_size,
                            batch_size=cfg.world_batch_size, niter=nb_iterations, stats=stats,
                            verbose=verbose, show_warnings=show_warnings,
                            results_infos=results_infos, log_base_path=log_base_path, save_period=None,
                            iteration_filename=f'{experiment_name}' + '/latest-{}.p',
                            trainer=trainer, idx_counter=idx_counter, cfg=cfg, 
                            )
        world_evolver.run_setup(init_batch_size=cfg.world_batch_size)

    else:
        world_evolver = None
        logbook = tools.Logbook()
        # TODO: use "chapters" to hierarchicalize generator fitness, agent reward, and path length stats?
        # NOTE: [avg, std, min, max] match the headers in deap.DEAPQDAlgorithm._init_stats. Could do this more cleanly.
        logbook.header = ["iteration", "meanRew", "meanEvalRew", "elapsed"]

#   if cfg.fixed_worlds:
#       for i in range(1000):
#           logbook_stats = train_players(trainer=trainer, worlds=training_worlds,
#                       # TODO: initialize logbook even if not evolving worlds
#                       cfg=cfg)
#       print('Done training, goodbye!')
#       sys.exit()

    if cfg.gen_adversarial_worlds:
        cfg.gen_phase_len = -1
        new_grid = containers.Grid(shape=(1, 1), max_items_per_bin=100, fitness_domain=cfg.fitness_domain,
                                   fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)
        archive = new_grid

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")

    # The outer co-learning loop
    if cfg.parallel_gen_play:
        toggle_train_player(trainer, train_player=True, cfg=cfg)
        # TODO: remove this function and initialize most of these objects in the trainer setup function.
        trainer.set_attrs(world_evolver, idx_counter, logbook, cfg, net_itr, gen_itr, play_itr)
        for _ in range(cfg.total_play_itrs):
            trainer.train()
        sys.exit()

    toggle_train_player(trainer, train_player=False, cfg=cfg)
    done = False
    while not done:

        # Run environment evolution
        done_gen_phase = False
        while not done_gen_phase:
            logbook_stats = world_evolver.evolve()
            gen_itr += 1
            done_gen_phase = get_done_gen_phase(world_evolver, gen_itr, cfg)

            if done_gen_phase:
                save(archive=archive, gen_itr=gen_itr, net_itr=net_itr, play_itr=play_itr, logbook=logbook,
                                save_dir=cfg.save_dir)
                mean_path_length = compute_archive_world_heuristics(archive=archive, trainer=trainer)
                stat_keys = ['mean', 'min', 'max']  # , 'std]
                logbook_stats.update({f'{k}Path': mean_path_length[f'{k}_path_length'] for k in stat_keys})

            elif gen_itr % cfg.save_interval == 0:
                save(archive=archive, play_itr=play_itr, gen_itr=gen_itr, net_itr=net_itr, logbook=logbook, save_dir=cfg.save_dir)

            log(logbook, logbook_stats, net_itr)
            net_itr += 1

        if cfg.gen_adversarial_worlds:
            # Then generator evolution has stagnated, so we are done.
            visualize_archive(cfg, env, archive)
            print('Done generating adversarial worlds.')
            sys.exit()

        # Run player training
        # TODO: account for player staleness/optimality?
        done_play_phase = False
        training_worlds = sorted(archive, key=lambda i: i.fitness.values[0], reverse=True)

        if cfg.quality_diversity:
            # Eliminate impossible worlds
            training_worlds = [t for t in training_worlds if not t.features == [0, 0]]

            # In case all worlds are impossible, do more rounds of evolution until some worlds are feasible.
            if len(training_worlds) == 0:
                done_play_phase = True

        # Use duplicate worlds if we don't have enough to match the number of rllib environments.
        training_worlds *= math.ceil(cfg.world_batch_size / len(training_worlds))

        while not done_play_phase:
            toggle_train_player(trainer, train_player=True, cfg=cfg)
            logbook_stats = train_players(training_worlds, trainer, cfg, idx_counter)
            log(logbook, logbook_stats, net_itr)
            play_itr += 1
            done_play_phase = get_done_play_phase(play_itr, cfg)
            done = play_itr >= cfg.total_play_itrs
            net_itr += 1

#       # Now that we've trained the player, update the generator-trainer before the next round of generator evolution.
#       sync_player_policies(gen_trainer, play_trainer, cfg)

        # Only doing this in case gen_trainer = play_trainer (i.e. not parallel). Can remove this once parallel evo/train
        # is implemented in separate loop (presumably).
        toggle_train_player(trainer, train_player=False, cfg=cfg)

        logbook_stats = world_evolver.reevaluate_elites()
        log(logbook, logbook_stats, net_itr)

    # Tie off some loose ends once co-learning completes.
    if world_evolver.final_filename != None and world_evolver.final_filename != "":
        world_evolver.save(os.path.join(world_evolver.log_base_path, world_evolver.final_filename))
    total_elapsed = timer() - world_evolver.start_time

    # Print results info
    print(f"Total elapsed: {world_evolver.total_elapsed}\n")
    print(archive.summary())
    # print("Best ever fitness: ", container.best_fitness)
    # print("Best ever ind: ", container.best)
    # print("%s filled bins in the grid" % (grid.size_str()))
    ##print("Solutions found for bins: ", grid.solutions)
    # print("Performances grid: ", grid.fitness)
    # print("Features grid: ", grid.features)

    # Create plot of the performance grid
    plot_path = os.path.join(log_base_path, "performancesGrid.pdf")
    plotGridSubplots(archive.quality_array[..., 0], plot_path, plt.get_cmap("nipy_spectral_r"), features_domain,
                     cfg.fitness_domain[0], nbTicks=None)
    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
    print("All results are available in the '%s' pickle file." % world_evolver.final_filename)
