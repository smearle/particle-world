import argparse
import os
import pickle
import random
import sys
import time
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
from qdpy.base import ParallelismManager
from qdpy.plots import plotGridSubplots
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer
from tqdm import tqdm

from env import ParticleGym, ParticleGymRLlib, ParticleMazeEnv, eval_mazes, eval_mazes_probdists
from generator import TileFlipGenerator, SinCPPNGenerator, CPPN, Rastrigin, Hill
from qdpy_utils import qdRLlibEval, qdpy_save_archive
from rllib_utils import init_particle_trainer, train_players, rllib_evaluate_worlds, IdxCounter, toggle_exploration
from swarm import NeuralSwarm, MazeSwarm
from utils import infer, infer_elites, qdpy_eval, simulate, save, discrete_to_onehot
from visualize import visualize_pyribs, plot_fitness_qdpy

seed = None
ndim = 2
n_pop = 1

# Base number of policies among which to differentiate. (If using QD to differentiate explicitly, we'll need 1 more,
# which we'll add later.)
width = 15
pg_delay = 50
n_nca_steps = 10
n_sim_steps = 100
pg_width = 500
pg_scale = pg_width / width
# swarm_type = MemorySwarm
rllib_eval = True

generator_phase = True  # Do we start by evolving generators, or training players?
gen_phase_len = -1
play_phase_len = -1

# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
fitness_weight = -1.0
creator.create("FitnessMin", base.Fitness, weights=(-fitness_weight,))
creator.create("Individual", list, fitness=creator.FitnessMin, features=list)

fitness_domain = [(-np.inf, np.inf)]


def phase_switch_callback(gen_itr, player_trainer, container, toolbox, idx_counter, stale, save_dir):
    # Run a round of player training, either at fixed intervals (every gen_phase_len generations)
    if gen_itr > 0 and (gen_phase_len != -1 and gen_itr % gen_phase_len == 0 or stale):
        qdpy_save_archive(container, gen_itr, save_dir)
        train_players(n_itr=play_phase_len, trainer=player_trainer, landscapes=container, idx_counter=idx_counter,
                      n_policies=n_policies, save_dir=save_dir, n_rllib_envs=n_rllib_envs)
        # else:
        #     if itr % play_phase_len:
        # pass
        invalid_ind = [ind for ind in container]
        container.clear_all()

        # After player training,
        if rllib_eval:
            rllib_stats, fitnesses = rllib_evaluate_worlds(player_trainer,
                                                           {i: ind for i, ind in enumerate(invalid_ind)}, idx_counter)
            fitnesses = [fitnesses[k] for k in range(len(fitnesses))]

        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.features = fit[1]

        # Store batch in container
        nb_updated = container.update(invalid_ind, issue_warning=True)
        if nb_updated == 0:
            raise ValueError(
                "No individual could be added back to the QD container when re-evaluating after player training.")

# class MazeIndividual(creator.Individual, M)

# TODO:
class CPPNIndividual(creator.Individual, CPPN):
    def __init__(self):
        CPPN.__init__(self, width)
        creator.Individual.__init__(self)


def run_qdpy():
    def iteration_callback(toolbox, rllib_eval, staleness_counter, save_dir, gen_itr, batch, container, logbook):
        """qdpy callback function, with custom args first, given with `partial` before the qdpy algo is instantiated."""
        idx_counter = ray.get_actor('idx_counter')
        if gen_itr % qdpy_save_interval == 0:
            pass
            # qdpy already saves the container, logbook, etc. regularly, so we can save any additional things here.
            # with open(f'runs/{args.experimentName}/learn.pickle', 'wb') as handle:
            #     dict = {
                    # 'logbook': logbook,
                    # 'policies': env.swarms,
                # }
                # pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        time_until_stale = 10
        no_update = np.array(logbook.select('nbUpdated')[-1:]) == 0
        if no_update:
            staleness_counter[0] += 1
        else:
            staleness_counter[0] = 0
        stale = staleness_counter[0] >= time_until_stale
        if stale:
            staleness_counter[0] = 0
        phase_switch_callback(gen_itr, player_trainer=particle_trainer, container=container, toolbox=toolbox,
                              idx_counter=idx_counter, stale=stale, save_dir=save_dir)

    qdpy_save_interval = 100
    if load:
        fname = f'latest-0'
        # fname = f'latest-0' if args.loadIteration is not None else 'latest-0'
        with open(f"runs/{args.experimentName}/{fname}.p", "rb") as f:
            data = pickle.load(f)
        # with open(f'runs/{args.experimentName}/learn.pickle', 'rb') as f:
        #     supp_data = pickle.load(f)
        #     policies = supp_data['policies']
        # env.set_policies(policies)
        grid = data['container']
        curr_iter = data['current_iteration']

        # Produce plots and visualizations
        if args.visualize:

            # render a grid of all worlds
            im_pg_width = width * 10
            im_grid = np.zeros((im_pg_width * nb_bins[0], im_pg_width * nb_bins[1], 3))
            print(im_grid.shape)
            gg = sorted(grid, key=lambda i: i.features)
            for g in gg:
                i, j = grid.index_grid(g.features)
                env.set_world(g)
                env.reset()
                im = env.render(mode='rgb', pg_width=im_pg_width)
                im_grid[i * im_pg_width: (i + 1) * im_pg_width, j * im_pg_width: (j + 1) * im_pg_width] = im

            # im_grid = im_grid.T
            im_grid = np.flip(im_grid, 0)
            im_grid = Image.fromarray(im_grid.astype(np.uint8))
            im_grid.save(os.path.join(save_dir, "level_grid.png"))

            logbook = data['logbook']
            plot_fitness_qdpy(save_dir, logbook, quality_diversity=args.quality_diversity)

            # save fitness qd grid
            plot_path = os.path.join(save_dir, "performancesGrid.png")
            plotGridSubplots(grid.quality_array[..., 0], plot_path, plt.get_cmap("magma"), features_domain,
                             fitness_domain[0], nbTicks=None)
            print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
            sys.exit()

        # Render and observe
        elif args.enjoy:
            # We'll look at each world independently in our single env
            elites = sorted(grid, key=lambda ind: ind.fitness, reverse=True)
            worlds = [np.array(i) for i in elites]
            if args.evaluate:
                worlds = eval_mazes_probdists
            for i, elite in enumerate(worlds):
                ret = rllib_evaluate_worlds(particle_trainer, worlds={i: elite}, idx_counter=idx_counter,
                                            evaluate_only=True)
            sys.exit()

        # Evaluate
        elif args.evaluate:

            # If rendering, we evaluate one world at a time so we end up rendering them all (as we only render one
            # environment at a time)
            if args.render:
                for i, z in enumerate(eval_mazes_probdists):
                    print(i)
                    worlds = {i: z}
                    ret = rllib_evaluate_worlds(particle_trainer, worlds=worlds, idx_counter=idx_counter,
                                                evaluate_only=True)

            # Otherwise, we evaluate worlds in parallel
            else:
                worlds = {i: z for i, z in enumerate(eval_mazes_probdists)}
                ret = rllib_evaluate_worlds(particle_trainer, worlds=worlds, idx_counter=idx_counter,
                                            evaluate_only=True)
            sys.exit()
    else:
        curr_iter = 0
    # Algorithm parameters
    dimension = len(initial_weights)  # The dimension of the target problem (i.e. genomes size)
    assert (dimension >= 2)
    assert (nb_features >= 1)
    bins_per_dim = int(pow(args.maxTotalBins, 1. / nb_features))

    init_batch_size = n_rllib_envs  # The number of evaluations of the initial batch ('batch' = population)
    batch_size = n_rllib_envs  # The number of evaluations in each subsequent batch
    nb_iterations = total_itrs - curr_iter  # The number of iterations (i.e. times where a new batch is evaluated)

    # Set the probability of mutating each value of a genome
    if generator_cls == TileFlipGenerator:
        mutation_pb = 0.3
    elif generator_cls == SinCPPNGenerator:
        mutation_pb = 0.03
    else:
        mutation_pb = 0.1
    eta = 20.0  # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
    max_items_per_bin = 1 if args.maxTotalBins != 1 else 100  # The number of items in each bin of the grid
    ind_domain = (0., 1.)  # The domain (min/max values) of the individual genomes
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
    toolbox.register("attr_float", random.uniform, ind_domain[0], ind_domain[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, dimension)
    # toolbox.register("individual", CPPNIndividual)
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

    # Turn off exploration before evolving. Henceforth toggled before/after player training.
    toggle_exploration(particle_trainer, explore=False, n_policies=n_policies)
    if not load:
        # Create container
        grid = containers.Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain,
                               fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)

    with ParallelismManager(args.parallelismType, toolbox=toolbox) as pMgr:
        qd_algo = partial(qdRLlibEval, particle_trainer, rllib_eval, args.quality_diversity)
        # The staleness counter will be incremented whenver a generation of evolution does not result in any update to
        # the archive. (Crucially, it is mutable.)
        staleness_counter = [0]
        callback = partial(iteration_callback, toolbox, rllib_eval, staleness_counter,
                           os.path.join('runs', args.experimentName))
        # Create a QD algorithm
        algo = DEAPQDAlgorithm(pMgr.toolbox, grid, init_batch_siz=init_batch_size,
                               batch_size=batch_size, niter=nb_iterations,
                               verbose=verbose, show_warnings=show_warnings,
                               results_infos=results_infos, log_base_path=log_base_path, save_period=qdpy_save_interval,
                               iteration_filename=f'{args.experimentName}' + '/latest-{}.p',
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


def run_pyribs():
    if load:
        with open('learn.pickle', 'rb') as handle:
            dict = pickle.load(handle)
            archive = dict['archive']
            emitters = dict['emitters']
            optimizer = dict['optimizer']
            stats = dict['stats']
            policies = dict['policies']
        visualize_pyribs(archive)
        if args.enjoy:
            # env.set_policies(policies)
            infer(env, generator, particle_trainer, archive, pg_width, pg_delay, rllib_eval)
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
        generators = [generator_cls(width=width, n_chan=env.n_chan) for _ in range(batch_size * n_emitters)]
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
                obj, bc = simulate(generator, particle_trainer, sol, env, n_steps=n_sim_steps, n_eps=1, screen=None,
                                   pg_delay=pg_delay, pg_scale=pg_scale)
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
            visualize_pyribs(archive)
        stats['n_iter'] += 1
        stats['obj_max'].append(archive.stats.obj_max)
        stats['obj_mean'].append(archive.stats.obj_mean)
    visualize_pyribs(archive)
    # infer(archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-en', '--enjoy', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="If reloading an experiment, produce plots, etc. "
                             "to visualize progress.")
    # parser.add_argument('-seq', '--sequential', help='not parallel', action='store_true')
    parser.add_argument('--maxTotalBins', type=int, default=1, help="Maximum number of bins in the grid")
    parser.add_argument('-p', '--parallelismType', type=str, default='None',
                        help="Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default='./runs', help="Path of the output log files")
    parser.add_argument('-li', '--loadIteration', default=-1, type=int)
    parser.add_argument('-a', '--algo', default='me')
    parser.add_argument('-exp', '--experimentName', default='test')
    parser.add_argument('-fw', '--fixed_worlds', action="store_true", help="When true, train players on fixed worlds, "
                                                                           "skipping the world-generation phase.")
    parser.add_argument('-g', '--generator_class', type=str, default="Rastrigin",
                        help="Which type of world-generator to use, whether a "
                             "fixed (set of) world(s), or a parameterizable "
                             "representation."
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
    args = parser.parse_args()
    n_policies = args.n_policies
    generator_cls = globals()[args.generator_class]
    n_rllib_workers = args.num_proc
    n_rllib_envs = n_rllib_workers * 6 if n_rllib_workers > 0 else 36

    if args.quality_diversity:
        # If using QD, objective score is determined by the fitness of an additional policy.
        assert n_policies > 1
        # Fitness function is fixed for QD experiments.
        assert args.objective_function is None
    else:
        # If not running a QD experiment, we must specify an objective function.
        assert args.objective_function is not None

    # swarm_cls = NeuralSwarm
    swarm_cls = MazeSwarm

    # env = ParticleSwarmEnv(width=width, n_policies=n_policies, n_pop=n_pop)
    env = ParticleMazeEnv(
    # env = ParticleGymRLlib(
        {'width': width, 'swarm_cls': swarm_cls, 'n_policies': n_policies, 'n_pop': n_pop, 'max_steps': n_sim_steps,
         'pg_width': pg_width, 'evaluate': args.evaluate, 'objective_function': args.objective_function})
    generator = generator_cls(width=width, n_chan=env.n_chan)

    initial_weights = generator.get_init_weights()

    save_dir = os.path.join(args.outputDir, args.experimentName)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    load = args.load
    total_itrs = 50000
    multi_proc = args.parallelismType != 'None'
    n_emitters = 5
    batch_size = 30

    if args.quality_diversity:
        # Define grid in terms of fitness of all policies (save 1, for fitness)
        nb_features = n_policies - 1
    else:
        nb_features = 2  # The number of features to take into account in the container
    bins_per_dim = int(pow(args.maxTotalBins, 1. / nb_features))
    nb_bins = (
                  bins_per_dim,) * nb_features  # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with $maxTotalBins^(1/nb_features)$ bins each
    save_interval = 10

    # Specific to maze env: since each agent could be on the goal for at most, e.g. 99 steps given 100 max steps
    features_domain = [(0, env.max_steps - 1)] * nb_features  # The domain (min/max values) of the features

    idx_counter = IdxCounter.options(name='idx_counter', max_concurrency=1).remote()
    particle_trainer = init_particle_trainer(env, num_rllib_workers=n_rllib_workers, n_rllib_envs=n_rllib_envs,
                                             enjoy=args.enjoy, render=args.render, save_dir=save_dir,
                                             num_gpus=args.num_gpus, evaluate=args.evaluate)

    # env.set_policies([particle_trainer.get_policy(f'policy_{i}') for i in range(n_policies)], particle_trainer.config)
    # env.set_trainer(particle_trainer)

    if args.load:
        if isinstance(env.swarms[0], NeuralSwarm) and rllib_eval:
            if args.loadIteration == -1:
                with open(os.path.join(save_dir, 'model_checkpoint_path.txt'), 'r') as f:
                    model_checkpoint_path = f.read()
            else:
                model_checkpoint_path = os.path.join(save_dir, f'checkpoint_{args.loadIteration:06d}/checkpoint-{args.loadIteration}')
            particle_trainer.load_checkpoint(model_checkpoint_path)

            # TODO: support multiple fixed worlds
            if args.enjoy and args.fixed_worlds:
                particle_trainer.workers.local_worker().set_policies_to_train([])
                rllib_evaluate_worlds(particle_trainer, {0: generator.landscape})
                # particle_trainer.evaluation_workers.foreach_worker(
                #     lambda worker: worker.foreach_env(lambda env: env.set_landscape(generator.world)))
                # particle_trainer.evaluate()
                sys.exit()

    if not args.fixed_worlds:
        if args.algo == 'me':
            run_qdpy()
        elif args.algo == 'cmame':
            run_pyribs()
        elif args.algo == 'ppo':
            # TODO: train generators?
            pass
    else:
        # TODO: evolve players?
        train_players(1000, trainer=particle_trainer, landscapes=[generator.world], n_policies=n_policies,
                      n_rllib_envs=n_rllib_envs, save_dir=save_dir)
        # train_players(play_phase_len, particle_trainer)
