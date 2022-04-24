import gc
import json
import math
import os
import pickle
import random
import shutil
import sys
from functools import partial
from pdb import set_trace as TT

import matplotlib.pyplot as plt
import numpy as np
import psutil
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

from envs import DirectedMazeEnv, MazeEnvForNCAgents, ParticleMazeEnv, eval_mazes, cross8_test_mazes, \
    ghost_action_test_maze, corridor_test_mazes, h_test_mazes, s_test_mazes
from envs.wrappers import make_env
from evo.players import Player
from generators.representations import TileFlipGenerator2D, TileFlipGenerator3D, SinCPPNGenerator, CPPN, Rastrigin, Hill
from evo.utils import compute_archive_world_heuristics, save
from evo.evolve import PlayerEvolver, WorldEvolver
from evo.individuals import TileFlipIndividual2D, NCAIndividual, TileFlipIndividual3D, clone
# from rl.trainer import init_trainer, sync_player_policies, toggle_train_player, train_players, toggle_exploration
from rl.trainer import init_trainer, toggle_train_player
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


@ray.remote
def simulate_remote(players, worlds, env, cfg, render_env=False):
    """
    Simulate a player in a separate process.
    :param player: The player to simulate.
    :param env: The environment to simulate in.
    :param cfg: The configuration.
    :return: The result of the simulation.
    """
    ret = simulate(players, worlds, env, cfg, render_env=render_env)
    # print(f"Remote simulation finished with {ret}")

    return ret

def simulate(players, worlds, env, cfg, render_env=False):
    """
    Simulate a player in a separate process.
    :param player: The player to simulate.
    :param env: The environment to simulate in.
    :param cfg: The configuration.
    :return: The result of the simulation.
    """
    env.set_player_policies(players=players)
    if not isinstance(list(worlds.values())[0], np.ndarray):
        # print(f'Simulating on world: {worlds.keys()}')
        env.queue_worlds(worlds={k: ind.discrete for k, ind in worlds.items()}, load_now=False)
    else:
        env.queue_worlds(worlds=worlds, load_now=False)
    ret = env.simulate(render_env=render_env)
    # print(f"Simulation finished with {ret}")

    return ret


def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


if __name__ == '__main__':
    parser = init_parser()
    cfg = parser.parse_args()

    if cfg.load_config is not None:
        cfg = load_config(cfg, cfg.load_config)

    # TODO: ClArgsConfig class.
    if cfg.model == 'flood':
        cfg.rotated_observations = False
        cfg.translated_observations = False

    cfg.single_thread = False
    cfg.n_elite_worlds = 1
    cfg.n_elite_players = 1
    cfg.fitness_domain = [(-np.inf, np.inf)]
    cfg.width = 15
    pg_delay = 50  # Delay for rendering in pygame (ms?). Probably not needed!
    n_nca_steps = 10
    # n_sim_steps = 100
    pg_width = 500
    pg_scale = pg_width / cfg.width
    cfg.save_interval = 100
    if cfg.evolve_players:
        cfg.archive_size = 100 if not cfg.quality_diversity else 256
    else:
        cfg.archive_size = 100 if not cfg.quality_diversity else 256
    # cfg.log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']

    n_evo_workers = (1 if cfg.n_evo_workers == 0 else cfg.n_evo_workers)

    # Number of episodes for player training = n_rllib_envs / n_rllib_workers = n_envs_per_worker (since we use local
    # worker for training simulation so as not to waste CPUs).
    # cfg.n_envs_per_worker = 40
    cfg.n_rllib_envs = n_evo_workers * cfg.n_envs_per_worker  # Note that this effectively sets n_envs_per_worker to 40.
    # cfg.n_rllib_envs = 400
    cfg.n_eps_on_train = cfg.n_rllib_envs
    cfg.world_batch_size = cfg.n_eps_on_train 

    # Whether to run rounds of player-training and generator-evolution in parallel.
    cfg.parallel_gen_play = True

    # # We must have the same number of envs per worker, and want to meet our target number of envs exactly.
    # assert cfg.n_rllib_envs % n_evo_workers == 0, \
    #     f"n_rllib_envs ({cfg.n_rllib_envs}) must be divisible by n_workers ({n_evo_workers})"

    # # We don't want any wasted episodes when we call rllib_evaluate_worlds() to evaluate worlds.
    # assert cfg.world_batch_size % cfg.n_eps_on_train == 0, \
    #     f"world_batch_size ({cfg.world_batch_size}) must be divisible by n_eps_on_train ({cfg.n_eps_on_train})"

    # # We don't want any wasted episodes when we call train() to evaluate worlds.
    # assert cfg.n_eps_on_train % cfg.n_rllib_envs == 0, \
    #     f"n_eps_on_train ({cfg.n_eps_on_train}) must be divisible by n_rllib_envs ({cfg.n_rllib_envs})"

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
                      'translated_observations': cfg.translated_observations, "training_world": True}

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
        assert cfg.n_envs_per_worker == cfg.n_rllib_envs // n_evo_workers

    register_env('world_evolution_env', make_env)

    experiment_name = get_experiment_name(cfg)
    save_dir = os.path.join(cfg.outputDir, experiment_name)
    cfg.save_dir = save_dir
    cfg.env_is_minerl = env_is_minerl

    if not cfg.load:
        # If we're not loading, and not overwriting, and the relevant ``save_dir`` exists, then raise Exception.
        if not cfg.overwrite:
            if os.path.exists(save_dir):
                # FIXME: even when we are running new experiment, the directoy already exists at this point. Why?
                # Ahhh it's TBXLogger? Can just skip this check?
                raise Exception(f"The save directory '{save_dir}' already exists. Use --overwrite to overwrite it.")

        # Remove the save directory if it exists and we are overwriting.
        else:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                print(f"Overwriting save directory '{save_dir}'.")
            else:
                print(f"No directory to overwrite, creating a new one: '{save_dir}'.")

        # Create the new save directory.
        os.mkdir(save_dir)


    # Dummy env, to get various parameters defined inside the env, or for debugging.
    env = make_env(env_config)

    # The callable objective function.
    cfg._obj_fn = env.objective_function

    n_sim_steps = env.max_episode_steps
    unique_chans = env.unique_chans

    cfg.total_play_itrs = 50000
    multi_proc = cfg.parallelismType != 'None'
    rllib_save_interval = 10

    world_idx_counter = IdxCounter.options(name='idx_counter', max_concurrency=1).remote()
    player_idx_counter = IdxCounter.options(name='play_idx_counter', max_concurrency=1).remote()

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

    if cfg.evolve_players:
        nb_bins_play = (1, 1)
        fitness_domain_play = [(env.min_reward, env.max_reward * 100)]
        features_domain_play = [(-1, 1), (-1, 1)]  # placeholder
        max_items_per_bin_play = 100
        fitness_weight_play = 1

    trainer = None if cfg.load and cfg.visualize or cfg.evolve_players else \
        init_trainer(env, idx_counter=world_idx_counter, env_config=env_config, cfg=cfg)

    # If training on fixed worlds, select the desired training set.
    if cfg.fixed_worlds:
        if cfg.evolve_players:
            # training_worlds = h_test_mazes
            training_worlds = s_test_mazes
        else:
            training_worlds = trainer.world_archive
        world_archive = training_worlds

    # If doing world/player, do any setup that is necessary regardless of whether we're reloading or starting anew.
    if not cfg.fixed_worlds or cfg.evolve_players:
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

        world_archive = data['world_archive']
        player_archive = data['player_archive']
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
                visualize_archive(cfg, env, world_archive)
                if cfg.quality_diversity:
                                # save fitness qd grid
                    plot_path = os.path.join(cfg.save_dir, "performancesGrid.png")
                    plotGridSubplots(world_archive.quality_array[..., 0], plot_path, plt.get_cmap("magma"), features_domain,
                                                cfg.fitness_domain[0], nbTicks=None)
                    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
                print('Done visualizing, goodbye!')
                sys.exit()

        if isinstance(env.swarms[0], NeuralSwarm) and cfg.rllib_eval:
            # if cfg.loadIteration == -1:
            if not cfg.evolve_players:
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
                worlds = list(training_worlds.values())
#               evaluate_worlds(trainer=trainer, worlds=training_worlds,
#                                     fixed_worlds=cfg.fixed_worlds, render=cfg.render)
                # particle_trainer.evaluation_workers.foreach_worker(
                #     lambda worker: worker.foreach_env(lambda env: env.set_landscape(generator.world)))
                # particle_trainer.evaluate()

            else:
                elites = sorted(world_archive, key=lambda ind: ind.fitness, reverse=True)
                worlds = [i for i in elites]
                # FIXME: Hack: avoid skipping world 0. Something about the way eval calls reset at step 0 of episode 0?
                worlds = [worlds[0]] + worlds

            if cfg.evolve_players:
                results = [simulate({0: sorted(player_archive, key=lambda ind: ind.fitness.values[0], reverse=True)[0]}, 
                                    worlds={wk: world}, env=env, cfg=cfg, render_env=True) for wk, world in enumerate(worlds)]

            else:
                for i, elite in enumerate(worlds):
                    print(f"Evaluating world {i}")
                    set_worlds({i: elite}, trainer.evaluation_workers, world_idx_counter, cfg)
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

    # If not loading, set up world/player-evolution stuff.
    else:
        # Initialize these counters if not reloading.
        gen_itr = 0
        play_itr = 0
        net_itr = 0

        logbook = tools.Logbook()
        logbook.header = ["iteration"]
        
        # If not loading, do some initial setup for world evolution if applicable.
        if not cfg.fixed_worlds:
            # Create empty container.
            world_archive = containers.Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=cfg.fitness_domain,
                                   fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)

            # TODO: use "chapters" to hierarchicalize generator fitness, agent reward, and path length stats?
            # NOTE: [avg, std, min, max] match the headers in deap.DEAPQDAlgorithm._init_stats. Could do this more cleanly.
            logbook.header += ["containerSize", "evals", "nbUpdated"] + stats.fields + ["meanPath", "maxPath"]
        
        else:
            # TODO: use "chapters" to hierarchicalize generator fitness, agent reward, and path length stats?
            # NOTE: [avg, std, min, max] match the headers in deap.DEAPQDAlgorithm._init_stats. Could do this more cleanly.
            logbook.header += ["minRew", "meanRew", "maxRew", "pctWin", "meanEvalRew"]
        
        if cfg.evolve_players:
            player_archive = containers.Grid(shape=nb_bins_play, max_items_per_bin=max_items_per_bin_play, fitness_domain=fitness_domain_play,
                                   fitness_weight=fitness_weight_play, features_domain=features_domain_play, storage_type=list)
            logbook.header += ["containerSizePlay", "evalsPlay", "nbUpdatedPlay"] + [f"{sf}Play" for sf in stats.fields]

        else:
            player_archive = None
            logbook.header += ["meanRew", "meanEvalRew"]

        logbook.header += ["fps", "elapsed"]

    # This is a placeholder when passed to Evolvers --> DeapQD algorithms.
    nb_iterations = cfg.total_play_itrs - play_itr  # The number of iterations (i.e. times where a new batch is evaluated)

    # Perform setup that must occur whether reloading or not, but which will need to occur after reloading if reloading.
    if not cfg.fixed_worlds:
        # Evolutionary algorithm parameters
        assert (nb_features >= 1)

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
        world_evolver = WorldEvolver(toolbox=toolbox, container=world_archive, init_batch_siz=cfg.world_batch_size,
                            batch_size=cfg.world_batch_size, niter=nb_iterations, stats=stats,
                            verbose=verbose, show_warnings=show_warnings,
                            results_infos=results_infos, log_base_path=log_base_path, save_period=None,
                            iteration_filename=f'{experiment_name}' + '/latest-{}.p',
                            curr_itr=net_itr, idx_counter=world_idx_counter, cfg=cfg, 
                            )
        world_evolver.run_setup(init_batch_size=cfg.world_batch_size)

    else:
        world_evolver = None

    if cfg.evolve_players:
        # Evolutionary algorithm parameters for players.
        nb_features = 2
        ind_domain = (0, env.n_chan)  # The domain (min/max values) of the individual genomes
        verbose = True
        show_warnings = False  # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
        log_base_path = cfg.outputDir if cfg.outputDir is not None else "."
        # Create Toolbox
        toolbox = base.Toolbox()
        player_class = Player
        toolbox.register("individual", player_class, obs_width=env.width - 2, obs_n_chan=env.n_chan)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("clone", clone)
        toolbox.register("mutate", lambda individual: individual.mutate())
        toolbox.register("select", tools.selRandom)  # MAP-Elites = random selection on a grid container
        # toolbox.register("select", tools.selBest) # But you can also use all DEAP selection functions instead to create your own QD-algorithm

        # Create a dict storing all relevant infos
        results_infos = {'ind_domain': ind_domain, 'features_domain': features_domain_play, 'fitness_domain': cfg.fitness_domain,
                        'nb_bins': nb_bins_play, 'init_batch_size': cfg.world_batch_size,
                        'batch_size': cfg.world_batch_size}

        # Create a QD algorithm
        player_evolver = PlayerEvolver(toolbox=toolbox, container=player_archive, init_batch_size=cfg.world_batch_size,
                            batch_size=cfg.world_batch_size, niter=nb_iterations, stats=stats,
                            verbose=verbose, show_warnings=show_warnings,
                            results_infos=results_infos, log_base_path=log_base_path, save_period=None,
                            iteration_filename=f'{experiment_name}' + '/latest-players-{}.p',
                            curr_itr=net_itr, idx_counter=world_idx_counter, cfg=cfg, 
                            )
        player_evolver.run_setup(init_batch_size=cfg.world_batch_size)

    else:
        player_evolver=None


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
        world_archive = new_grid

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")

    # The outer co-learning loop
    # TODO: remove this function and initialize most of these objects in the trainer setup function.
    if trainer:
        toggle_train_player(trainer, train_player=True, cfg=cfg)
        trainer.set_attrs(world_evolver, world_idx_counter, logbook, cfg, net_itr, gen_itr, play_itr, player_evolver)

    for _ in range(cfg.total_play_itrs):
        if cfg.evolve_players:
            for _ in range(cfg.play_phase_len):
                player_evolve_start_time = timer()
                player_batch = player_evolver.ask(cfg.world_batch_size)

                if not cfg.fixed_worlds:
                    if gen_itr == 0:
                        world_batch = world_evolver.ask(cfg.world_batch_size)
                        elite_world_lst = [ind for ind in world_batch.values()]
                    else:
                        elite_world_lst = sorted(world_evolver.container, key=lambda ind: ind.fitness.values[0], reverse=True)
                    elite_world_lst = elite_world_lst[:cfg.n_elite_worlds]
                    elite_worlds = {k: ind for k, ind in enumerate(elite_world_lst)}
                    # print(f"Evaluating players on worlds with fitness: {[ind.fitness.values[0] for ind in elite_worlds.values()]}")
                    
                else:
                    elite_worlds = world_archive

                # Submit a bunch of simulations for player evaluation/evolution.
                if cfg.single_thread:
                    results = [simulate(players={pk: player}, worlds=elite_worlds, env=env, cfg=cfg) for pk, player in player_batch.items()]
                else:
                    futures = [simulate_remote.remote(players={pk: player}, worlds=elite_worlds, env=env, cfg=cfg, render_env=(pk==0 and cfg.render)) for pk, player in player_batch.items()]
                    # Get results of player simulations.
                    results = ray.get(futures)

                player_rews_lst, _ = [r[0] for r in results], [r[1] for r in results]

                player_rews = {}
                [player_rews.update(pr) for pr in player_rews_lst]
                logbook_stats = player_evolver.tell(player_batch, player_rews)
                frames = cfg.world_batch_size * env.max_episode_steps
                player_evolve_time = timer() - player_evolve_start_time
                logbook_stats.update({'elapsed': player_evolve_time, 'fps': frames / player_evolve_time})
                log(logbook, logbook_stats, net_itr)
                play_itr += 1
                net_itr += 1

            if not cfg.fixed_worlds:
                for _ in range(cfg.gen_phase_len):
                    world_evolve_start_time = timer()
                    world_batch = world_evolver.ask(cfg.world_batch_size)
                    if play_itr == 0:
                        elite_players_lst = [ind for ind in player_batch.values()]
                    else:
                        elite_players_lst = sorted(player_evolver.container, key=lambda ind: ind.fitness.values[0], reverse=True)
                    elite_players_lst = elite_players_lst[:cfg.n_elite_players]
                    elite_players = {k: ind for k, ind in enumerate(elite_players_lst)}
                    # print(f"Evaluating worlds on players with fitness: {[ind.fitness.values[0] for ind in elite_players.values()]}")

                    # Submit a bunch of simulations for world evaluation/evolution.
                    # print(f"Evaluating worlds: {world_batch.keys()}")
                    if cfg.single_thread:
                        results = [simulate(players=elite_players, worlds={wk: world}, env=env, cfg=cfg) for wk, world in world_batch.items()]
                    else:
                        futures = [simulate_remote.remote(players=elite_players, worlds={wk: world}, env=env, cfg=cfg, render_env=(wk==0 and cfg.render)) for wk, world in world_batch.items()]
                        results = ray.get(futures)
                    _, world_qd_stats_lst = [r[0] for r in results], [r[1] for r in results]
                    world_qd_stats = {s['world_key']: s['qd_stats'] for s_lst in world_qd_stats_lst for s in s_lst}
                    logbook_stats = world_evolver.tell(world_batch, world_qd_stats)
                    frames = cfg.world_batch_size * env.max_episode_steps
                    world_evolve_time = timer() - world_evolve_start_time
                    logbook_stats.update({'elapsed': world_evolve_time, 'fps': frames / world_evolve_time})
                    log(logbook, logbook_stats, net_itr)

                    gen_itr += 1
                    net_itr += 1

            if net_itr % 10 == 0:
                save(world_archive=world_archive, player_archive=player_archive, gen_itr=net_itr, 
                    net_itr=net_itr, play_itr=net_itr, logbook=logbook, save_dir=cfg.save_dir)


            auto_garbage_collect()

        else:
            trainer.train()

    sys.exit()

