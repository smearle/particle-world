import pickle
import tqdm
from timeit import default_timer as timer

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer

from visualize import visualize_pyribs

# Currently this file is not maintained. Working with qdpy only to start.


#def run_pyribs(
#    particle_trainer, env, generator, load, args, nb_bins, features_domain, seed, n_emitters, initial_weights, 
#    batch_size, multi_proc, generator_cls, width, n_policies, ...
#                ):
#    if load:
#        with open('learn.pickle', 'rb') as handle:
#            dict = pickle.load(handle)
#            archive = dict['archive']
#            emitters = dict['emitters']
#            optimizer = dict['optimizer']
#            stats = dict['stats']
#            policies = dict['policies']
#        visualize_pyribs(archive)
#        if args.enjoy:
#            # TODO: ...
#            pass
#            # env.set_policies(policies)
#            # infer(env, generator, particle_trainer, archive, pg_width, pg_delay, rllib_eval)
#    else:
#        stats = {
#            'n_iter': 0,
#            'obj_max': [],
#            'obj_mean': [],
#        }
#        archive = GridArchive(
#            dims=nb_bins,
#            # dims=[100,100],
#            ranges=features_domain,
#            seed=seed,
#        )
#        seeds = ([None] * n_emitters
#                 if seed is None else [seed + i for i in range(n_emitters)])
#        if max_total_bins == 1:
#            n_opt_emitters = len(seeds)
#            n_imp_emitters = 0
#        else:
#            n_opt_emitters = 0
#            n_imp_emitters = len(seeds)
#        emitters = [
#                       # OptimizingEmitter(
#                       ImprovementEmitter(
#                           archive,
#                           initial_weights.flatten(),
#                           sigma0=.1,
#                           batch_size=batch_size,
#                           seed=s,
#                       ) for s in seeds[:n_imp_emitters]] + \
#                   [
#                       OptimizingEmitter(
#                           # ImprovementEmitter(
#                           archive,
#                           initial_weights.flatten(),
#                           sigma0=1,
#                           batch_size=batch_size,
#                           seed=s,
#                       ) for s in seeds[n_imp_emitters:]
#                   ]
#        optimizer = Optimizer(archive, emitters)
#
#    start_time = timer()
#    if multi_proc:
#        from ray.util.multiprocessing import Pool
#
#        # Setup for parallel processing.
#        pool = Pool()
#        generators = [generator_cls(width=width, n_chan=env.n_chan) for _ in range(batch_size * n_emitters)]
#        # generators = [generator for _ in range(batch_size * n_emitters)]
#        envs = [ParticleGym(width=width, n_pop=n_pop, n_policies=n_policies) for _ in
#                range(batch_size * n_emitters)]
#        # envs = [env for _ in range(batch_size * n_emitters)]
#    for itr in tqdm(range(1, total_itrs + 1)):
#        sols = optimizer.ask()
#        objs = []
#        bcs = []
#        if multi_proc:
#            sim = partial(simulate, render=False, n_steps=n_sim_steps, n_eps=1)
#            ret = pool.starmap(sim, zip(generators, sols, envs))
#            objs, bcs = zip(*ret)
#        else:
#            for i, sol in enumerate(sols):
#                obj, bc = simulate(generator, particle_trainer, sol, env, n_steps=n_sim_steps, n_eps=1, screen=None,
#                                   pg_delay=pg_delay, pg_scale=pg_scale)
#                objs.append(obj)
#                bcs.append(bc)
#        optimizer.tell(objs, bcs)
#        if itr % 1 == 0:
#            elapsed_time = time.time() - start_time
#            print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
#            print(f"  - Archive Size: {len(archive)}")
#            print(f"  - Max Score: {archive.stats.obj_max}")
#            print(f"  - Mean Score: {archive.stats.obj_mean}")
#        if itr % rllib_save_interval == 0:
#            save(archive, optimizer, emitters, stats, policies=env.swarms)
#            visualize_pyribs(archive)
#        stats['n_iter'] += 1
#        stats['obj_max'].append(archive.stats.obj_max)
#        stats['obj_mean'].append(archive.stats.obj_mean)
#    visualize_pyribs(archive)
#    # infer(archive)

