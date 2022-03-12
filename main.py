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
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import ray
from deap import base
from deap import creator
from deap import tools
from qdpy import containers
from qdpy.algorithms.deap import DEAPQDAlgorithm
# from qdpy.base import ParallelismManager
from qdpy.plots import plotGridSubplots
from timeit import default_timer as timer
from tqdm import tqdm

from envs import DirectedMazeEnv, MazeEnvForNCAgents, ParticleMazeEnv, TouchStone, eval_mazes, full_obs_test_mazes
from envs.wrappers import make_env
from generator import TileFlipGenerator2D, TileFlipGenerator3D, SinCPPNGenerator, CPPN, Rastrigin, Hill
from qdpy_utils.utils import qdRLlibEval, qdpy_save_archive
from qdpy_utils.individuals import TileFlipIndividual2D, NCAIndividual, TileFlipIndividual3D
from ray.tune.registry import register_env
from rllib_utils.trainer import init_particle_trainer, train_players, toggle_exploration
from rllib_utils.eval_worlds import rllib_evaluate_worlds
from rllib_utils.utils import IdxCounter
from swarm import DirectedMazeSwarm, NeuralSwarm, MazeSwarm
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
rllib_eval = True

generator_phase = True  # Do we start by evolving generators, or training players?

# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
fitness_weight = -1.0
creator.create("FitnessMin", base.Fitness, weights=(-fitness_weight,))
creator.create("Individual", list, fitness=creator.FitnessMin, features=list)

fitness_domain = [(-np.inf, np.inf)]


def phase_switch_callback(net_itr, gen_itr, play_itr, player_trainer, container, toolbox, logbook, idx_counter, stale_generators, 
                          save_dir, quality_diversity, stats):
    # Run a round of player training, either at fixed intervals (every gen_phase_len generations)
#   if args.objective_function == "min_solvable":
#       max_possible_generator_fitness = n_sim_steps - 1 / n_pop
#       optimal_generators = logbook.select("avg")[-1] >= max_possible_generator_fitness - 1e-3
#   else:
    optimal_generators = False

    # No player training if we're using the optimal player-policies
    if args.oracle_policy:
        return net_itr

    if gen_itr > 0 and (gen_phase_len != -1 and gen_itr % gen_phase_len == 0 or stale_generators or optimal_generators):
        qdpy_save_archive(container=container, gen_itr=gen_itr, net_itr=net_itr, play_itr=play_itr, logbook=logbook, save_dir=save_dir)
        training_worlds = sorted(container, key=lambda i: i.fitness.values[0], reverse=True)
        if quality_diversity:
            # Eliminate impossible worlds
            training_worlds = [t for t in training_worlds if not t.features == [0, 0]]
            training_worlds *= math.ceil(num_rllib_envs / len(training_worlds))
        net_itr = train_players(net_itr=net_itr, play_phase_len=play_phase_len, trainer=player_trainer,
                      landscapes=training_worlds,
                      idx_counter=idx_counter, n_policies=n_policies, n_pop=n_pop, n_sim_steps=n_sim_steps, 
                      save_dir=save_dir, n_rllib_envs=num_rllib_envs, logbook=logbook, 
                      quality_diversity=quality_diversity, render=args.render)
        # else:
        #     if itr % play_phase_len:
        # pass
        start_time = timer()
        invalid_ind = [ind for ind in container]
        container.clear_all()

        # After player training, the reckoning: re-evaluate all worlds with new policies
        net_itr -= 1
        if rllib_eval:
            rl_stats, world_stats, logbook_stats = rllib_evaluate_worlds(
                net_itr=net_itr, trainer=player_trainer, worlds={i: ind for i, ind in enumerate(invalid_ind)}, 
                idx_counter=idx_counter, quality_diversity=quality_diversity, start_time=start_time,
                calc_world_heuristics=False, render=args.render)

        else:
            world_stats = toolbox.map(toolbox.evaluate, invalid_ind)

        world_stats = {k: ws[0] for k, ws in world_stats.items()}
        update_individuals(invalid_ind, world_stats)
        

        # Store batch in container
        nb_updated = container.update(invalid_ind, issue_warning=True)
        if nb_updated == 0:
            raise ValueError(
                "No individual could be added back to the QD container when re-evaluating after player training.")

        record = stats.compile(container) if stats else {}
        logbook_stats.update({
            'iteration': net_itr, 'containerSize': container.size_str(), 'evals': len(invalid_ind), 'nbUpdated': nb_updated,
            'elapsed': timer() - start_time, **record,
        })
        logbook.record(**logbook_stats)
        print(logbook.stream)
        net_itr += 1
    return net_itr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ec', '--environment_class', type=str, default="ParticleMazeEnv", help="Which environment to "
    "use (one of ParticleMazeEnv, TouchStone).")
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-en', '--enjoy', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="If reloading an experiment, produce plots, etc. "
                             "to visualize progress.")
    # parser.add_argument('-seq', '--sequential', help='not parallel', action='store_true')
    # parser.add_argument('--max_total_bins', type=int, default=1, help="Maximum number of bins in the grid")
    parser.add_argument('-p', '--parallelismType', type=str, default='None',
                        help="Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default='./runs', help="Path of the output log files")
    # parser.add_argument('-li', '--loadIteration', default=-1, type=int)
    parser.add_argument('-a', '--algo', default='me')
    parser.add_argument('-exp', '--exp_name', default='test')
    parser.add_argument('-fw', '--fixed_worlds', action="store_true", help="When true, train players on fixed worlds, "
                                                                           "skipping the world-generation phase.")
    parser.add_argument('-g', '--generator_class', type=str, default="TileFlipIndividual",
                        help="An evolvable representation of the environment (or environment-generator(?))."
                        )
    parser.add_argument('-r', '--render', action='store_true', help="Render the environment (even during training).")
    parser.add_argument('-np', '--num_proc', type=int, default=0, help="Number of RLlib workers. Each uses 1 CPU core.")
    parser.add_argument('-gpus', '--num_gpus', type=int, default=1, help="How many GPUs to use for training.")
    parser.add_argument('-ev', '--evaluate', action='store_true', help="Whether to evaluate trained agents/worlds and"
                                                                       "collect relevant stats.")
    parser.add_argument('-qd', '--quality_diversity', action='store_true',
                        help='Search for a grid of levels with dimensions (measures) given by the fitness of distinct '
                             'policies, and objective score given by the inverse fitness of an additional policy.')
    parser.add_argument('-obj', '--objective_function', type=str, default=None,
                        help='If not using quality diversity, the name of the fitness function that will compute world'
                             'fitness based on population-wise rewards.')
    parser.add_argument('-n_pol', '--n_policies', type=int, default=1, help="How many distinct policies to train.")
    parser.add_argument('-op', '--oracle_policy', action='store_true', help="Whether to use the oracle (optimal) policy, and"
                                                                      "thereby focus on validating generator-evolution.")
    parser.add_argument('-fo', '--fully_observable', action='store_true',
                        help="Whether to use a fully observable environment.")
    parser.add_argument('-gp', '--gen_phase_len', type=int, default=-1,
                        help="How many generations to evolve worlds (generator). If -1, run until convergence.")
    parser.add_argument('-pp', '--play_phase_len', type=int, default=1, 
                        help="How many iterations to train the player. If -1, run until convergence.")
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-fov', '--field_of_view', type=int, default=2, help='How far agents can see in each direction.')
    parser.add_argument('-lc', '--load_config', type=int, default=None, 
                        help="Load a dictionary of (automatically-generated) arguments. "
                        "NOTE: THIS OVERWRITES ALL OTHER ARGUMENTS AVAILABLE IN THE COMMAND LINE.")
    parser.add_argument('-tr', '--target_reward', type=int, default=0, 
                        help="Target reward the world should elicit from player if using the min_solvable objective "
                        "function.")
    parser.add_argument('-ro', '--rotated_observations', action='store_true',
                        help="Whether to use rotated observations. If so, the agent will have an action space consisting"
                        "of moving forward, staying put, and turning left or right. It will perceive its orientation as a "
                        "discrete space.")
    parser.add_argument('-to', '--translated_observations', action='store_true', help='Whether to use translated '
                        ' observations. This will always be the True when fully_observable is False.')
    args = parser.parse_args()

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
            'pg_width': pg_width, 'evaluate': args.evaluate, 'objective_function': args.objective_function, 
            'fully_observable': args.fully_observable, 'fov': args.field_of_view, 'num_eval_envs': 1, 
            'target_reward': args.target_reward, 'rotated_observations': args.rotated_observations, 
            'translated_observations': args.translated_observations}

    elif args.environment_class == 'TouchStone':
        n_envs_per_worker = 1

        if args.generator_class == 'TileFlipIndividual':
            generator_class = TileFlipIndividual3D
        else: raise NotImplementedError

        environment_class = TouchStone
        touchstone = TouchStone()
        touchstone.register()
        env_config = {}

    else:
        raise Exception(f"Unrecognized environment class: {args.environment_class}")

    env_config.update({'environment_class': environment_class})

    n_rllib_workers = args.num_proc

    if (args.enjoy or args.evaluate) and args.render:
        num_rllib_envs = 1
    else:
        num_rllib_envs = n_rllib_workers * n_envs_per_worker if n_rllib_workers > 1 else 1

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

    register_env('world_evolution_env', make_env)
 
    env = make_env(env_config)
    TT()

#   ### DEBUGGING THE ENVIRONMENT ###
#   if environment_class == ParticleMazeEnv:
#       env.set_worlds(eval_mazes)
#       obs = env.reset()
#       for i in range(1000):
#           env.render()
#           obs, _, _, _ = env.step({ak: env.action_spaces[0].sample() for ak in obs})

#   elif environment_class == TouchStone:
#       env.set_worlds({'world_0': TileFlipIndividual3D(7, env.n_chan, unique_chans=env.unique_chans).discrete})
#       obs = env.reset()
#       done = False
#       while not done:
#           env.render()
#           action = env.action_space.sample()
#           # action = env.action_space.nooop()
#           obs, rew, done, info = env.step(action)


    n_sim_steps = env.max_episode_steps
    unique_chans = env.unique_chans

    generator = generator_class(width=width, n_chan=env.n_chan, unique_chans=unique_chans)

    # Don't need to know the dimension here since we'll simply call an individual's "mutate" method
    # initial_weights = generator.get_init_weights()
    
    experiment_name = get_experiment_name(args)
    load = args.load
    total_play_itrs = 50000
    multi_proc = args.parallelismType != 'None'
    n_emitters = 5
    batch_size = 30

    if args.quality_diversity:
        # Define grid in terms of fitness of all policies (save 1, for fitness)
        nb_features = n_policies - 1
    else:
        nb_features = 2  # The number of features to take into account in the container
    bins_per_dim = int(pow(max_total_bins, 1. / nb_features))
    nb_bins = (bins_per_dim,) * nb_features  # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with $max_total_bins^(1/nb_features)$ bins each
    rllib_save_interval = 10

    # Specific to maze env: since each agent could be on the goal for at most, e.g. 99 steps given 100 max steps
    features_domain = [(0, env.max_episode_steps - 1)] * nb_features  # The domain (min/max values) of the features
    save_dir = os.path.join(args.outputDir, experiment_name)

    idx_counter = IdxCounter.options(name='idx_counter', max_concurrency=1).remote()

    particle_trainer = None if args.load and args.visualize else \
        init_particle_trainer(env, num_rllib_remote_workers=n_rllib_workers, n_rllib_envs=num_rllib_envs,
                                             enjoy=args.enjoy, render=args.render, save_dir=save_dir,
                                             num_gpus=args.num_gpus, evaluate=args.evaluate, idx_counter=idx_counter,
                                             oracle_policy=args.oracle_policy, fully_observable=args.fully_observable,
                                             model=args.model, env_config=env_config, fixed_worlds=args.fixed_worlds,
                                             rotated_observations=args.rotated_observations)

    # env.set_policies([particle_trainer.get_policy(f'policy_{i}') for i in range(n_policies)], particle_trainer.config)
    # env.set_trainer(particle_trainer)

    max_items_per_bin = 1 if max_total_bins != 1 else num_rllib_envs  # The number of items in each bin of the grid

    if args.fixed_worlds:
        # train_worlds = {0: generator.landscape}
        training_worlds = full_obs_test_mazes

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
                        im = env.render(mode='rgb', pg_width=world_im_width)
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
                        im = env.render(mode='rgb', pg_width=world_im_width)
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

        if isinstance(env.swarms[0], NeuralSwarm) and rllib_eval:
            # if args.loadIteration == -1:
            with open(os.path.join(save_dir, 'model_checkpoint_path.txt'), 'r') as f:
                model_checkpoint_path = f.read()
            # else:
                # model_checkpoint_path = os.path.join(save_dir, f'checkpoint_{args.loadIteration:06d}/checkpoint-{args.loadIteration}')
            particle_trainer.load_checkpoint(model_checkpoint_path)
            print(f'Loaded model checkopint at {model_checkpoint_path}')


        # Render and observe
        if args.enjoy:

            # TODO: support multiple fixed worlds
            if args.fixed_worlds:
                particle_trainer.workers.local_worker().set_policies_to_train([])
                rllib_evaluate_worlds(trainer=particle_trainer, worlds=training_worlds, calc_world_heuristics=False, 
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
                ret = rllib_evaluate_worlds(trainer=particle_trainer, worlds={i: elite}, idx_counter=idx_counter,
                                            evaluate_only=True, calc_world_heuristics=False, render=args.render)
            sys.exit()

        # Evaluate
        if args.evaluate:
            worlds = eval_mazes
            rllib_stats = particle_trainer.evaluate()
            qd_stats = particle_trainer.evaluation_workers.foreach_worker(lambda worker: worker.foreach_env(
                lambda env: env.get_world_stats(evaluate=True, quality_diversity=args.quality_diversity)))
            qd_stats = [qds for worker_stats in qd_stats for qds in worker_stats]
            # rllib_stats, qd_stats, logbook_stats = rllib_evaluate_worlds(trainer=particle_trainer, worlds=worlds, idx_counter=idx_counter,
                                        # evaluate_only=True)
            TT()
            sys.exit()

    # Train

    else:
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
        train_players(0, 1000, trainer=particle_trainer, landscapes=training_worlds, n_policies=n_policies,
                      n_rllib_envs=num_rllib_envs, save_dir=save_dir, n_pop=n_pop, n_sim_steps=n_sim_steps, 
                      quality_diversity=args.quality_diversity, fixed_worlds=args.fixed_worlds,
                      # TODO: initialize logbook even if not evolving worlds
                      logbook=None)
        sys.exit()

    def iteration_callback(iteration, net_itr, play_itr, toolbox, rllib_eval, staleness_counter, save_dir, batch, 
                           container, logbook, stats, oracle_policy=False, quality_diversity=False):
        net_itr_lst = net_itr
        play_itr_lst = play_itr
        assert len(net_itr_lst) == 1 == len(play_itr_lst)
        net_itr = net_itr_lst[0]
        play_itr = play_itr_lst[0]

        gen_itr = iteration
        idx_counter = ray.get_actor('idx_counter')
        if net_itr % qdpy_save_interval == 0:
            qdpy_save_archive(container=container, play_itr=play_itr, gen_itr=gen_itr, net_itr=net_itr, logbook=logbook, save_dir=save_dir)
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
                              player_trainer=particle_trainer, container=container, 
                              toolbox=toolbox, logbook=logbook, idx_counter=idx_counter, stale_generators=stale, 
                              save_dir=save_dir, quality_diversity=quality_diversity, stats=stats)
        net_itr_lst[0] = net_itr
        play_itr_lst[0] = play_itr
        return net_itr

    qdpy_save_interval = 100

    # Algorithm parameters
    # dimension = len(initial_weights)  # The dimension of the target problem (i.e. genomes size)
    # assert (dimension >= 2)
    assert (nb_features >= 1)

    init_batch_size = num_rllib_envs  # The number of evaluations of the initial batch ('batch' = population)
    batch_size = num_rllib_envs  # The number of evaluations in each subsequent batch
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
    toolbox.register("individual", generator_class, width=env.width-2, n_chan=env.n_chan, save_gen_sequence=args.render)
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
    results_infos['mutation_pb'] = mutation_pb
    results_infos['eta'] = eta

    # Turn off exploration before evolving. Henceforth toggled before/after player training.
    # toggle_exploration(particle_trainer, explore=False, n_policies=n_policies)

    # with ParallelismManager(args.parallelismType, toolbox=toolbox) as pMgr:
    qd_algo = partial(qdRLlibEval, rllib_trainer=particle_trainer, rllib_eval=rllib_eval, net_itr=net_itr,
                        quality_diversity=args.quality_diversity, oracle_policy=args.oracle_policy, gen_itr=gen_itr,
                        logbook=logbook, play_itr=play_itr, render=args.render)
    # The staleness counter will be incremented whenver a generation of evolution does not result in any update to
    # the archive. (Crucially, it is mutable.)
    staleness_counter = [0]
    callback = partial(iteration_callback, toolbox=toolbox, rllib_eval=rllib_eval, 
                        staleness_counter=staleness_counter, save_dir=save_dir,
                        quality_diversity=args.quality_diversity)
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
