import os
from pdb import set_trace as TT
import pickle

import numpy as np
import pygame
from matplotlib import pyplot as plt
from pygame.constants import KEYDOWN
from ribs.visualize import grid_archive_heatmap

from swarm import eval_fit


def nnb(ps):
    d_xy = ((ps[None, :, :] - ps[:, None, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d_xy, d_xy.max() + 1)
    nnbs = np.argmin(d_xy, axis=1)
    return nnbs


def emptiness(x):
    empt = x.mean()
    return empt


def symmetry(x):
    w, h = x.shape
    symm = np.abs(x - x.T).sum()
    # symmetry = np.abs(x[:w // 2].flip(0) - x[w // 2:]).sum() + \
    #            np.abs(x[:, :h // 2].flip(1) - x[:, h // 2:]).sum()
    # symmetry = np.abs(np.flip(x[:w // 2], 0) - x[w // 2:]).sum() + \
    #            np.abs(np.flip(x[:, :h // 2], 1) - x[:, h // 2:]).sum()
    return (1 - symm / (w * h)).item()


def fit_dist(pops, scape):
    '''n-1 distances in mean fitness, determining the ranking of n populations.'''
    assert len(pops) == 2
    inter_dist = [eval_fit(pi.ps, scape).mean() - eval_fit(pj.ps, scape).mean()
                  for j, pj in enumerate(pops) for pi in pops[j + 1:]]


# def animate_nca(generator):
#     #   cv2.namedWindow('NCA landscape generation')
#     generator.generate(render=True, pg_delay=pg_delay)


def simulate(generator, g_weights, env, n_steps=100, n_eps=1, render=False, screen=None, pg_delay=1, pg_scale=1):
    generator.set_weights(g_weights)
    all_fit_difs = np.empty((n_eps))
    all_bcs = np.empty((n_eps, 2))
    for i in range(n_eps):
        generator.generate(render=render, screen=screen, pg_delay=pg_delay)
        env.set_landscape(generator.landscape)
        env.reset()
        fit_difs, bcs = env.simulate(n_steps=n_steps, generator=generator, render=render, screen=screen,
                                     pg_delay=pg_delay, pg_scale=pg_scale)
        all_fit_difs[i] = fit_difs
        all_bcs[i] = bcs
    #   obj = -np.std(all_fit_difs)
    #   bcs = np.mean(all_fit_difs)
    obj = np.mean(all_fit_difs)
    # obj = symmetry(generator.landscape)
    # bcs = [symmetry(generator.landscape), emptiness(generator.landscape)]
    bcs = all_bcs.mean(0) / env.width
    return obj, bcs


def infer(env, generator, archive, pg_width, pg_delay, trainer, rllib_eval):
    df = archive.as_pandas()
    high_perf_sols = df.sort_values("objective", ascending=False)
    # elites = high_perf_sols.iloc[[0, len(high_perf_sols) // 2, -1]].iterelites()
    elites = high_perf_sols.iterelites()
    elites = [e.sol for e in elites]
    return infer_elites(env, generator, trainer, elites, pg_width, pg_delay, rllib_eval)


def infer_elites(env, generator, trainer, elites, pg_width, pg_delay, rllib_eval):
    n_eps = 1
    pygame.init()
    screen = pygame.display.set_mode([pg_width, pg_width])
    pg_scale = pg_width / env.width
    running = True
    while running:
        for world_idx, g_weights in enumerate(elites):
            generator.set_weights(g_weights)
            all_fit_difs = np.empty((n_eps))
            all_bcs = np.empty((n_eps, 2))
            for i in range(n_eps):
                generator.generate(render=True, screen=screen, pg_delay=pg_delay)
                env.set_world_eval(generator.landscape, world_idx)
                if rllib_eval:
                    obs = env.reset()
                    env.render(pg_delay=pg_delay)
                    done = False
                    while not done:
                        agent_obs = {}
                        agent_ids = obs.keys()
                        for k in agent_ids:
                            n_pol = k[0]
                            if n_pol not in agent_obs:
                                agent_obs[n_pol] = {k: obs[k]}
                            else:
                                agent_obs[n_pol].update({k: obs[k]})
                        actions = {}
                        for k in agent_obs:
                            actions.update(trainer.compute_actions(agent_obs[k], policy_id=f'policy_{k}', explore=False))
                        obs, rew, dones, info = env.step(actions)
                        env.render(pg_delay=pg_delay)
                        done = dones['__all__']
                    fitnesses = env.get_fitness()
                    assert len(fitnesses) == 1
                    obj, bcs = env.get_fitness()[world_idx]
                    assert len(obj) == 1
                    all_fit_difs[i] = obj[0]
                    all_bcs[i] = bcs
                else:
                    fit_difs, bcs = env.simulate(n_steps=env.max_steps, generator=generator, render=True, screen=screen,
                                                 pg_delay=pg_delay, pg_scale=pg_scale)
                    all_fit_difs[i] = fit_difs
                    all_bcs[i] = bcs
            #   obj = -np.std(all_fit_difs)
            #   bcs = np.mean(all_fit_difs)
            obj = np.mean(all_fit_difs)
            # obj = symmetry(generator.landscape)
            # bcs = [symmetry(generator.landscape), emptiness(generator.landscape)]
            bcs = all_bcs.mean(0) / env.width
            print(f'obj: {obj}, bcs: {bcs}')
            # Pause on the final frame for analysis/debugging
            pause = True
            while pause:
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        pause = False
    pygame.quit()


def _pg_anim(generator, trainer, model, env, pg_width, pg_delay):
    screen = pygame.display.set_mode([pg_width, pg_width])
    pg_scale = pg_width / env.width
    obj, bcs = simulate(generator, trainer, model, env, render=True, screen=screen, pg_delay=pg_delay, pg_scale=pg_scale)
    print(f'obj: {obj} \n bcs: {bcs}')


def save(archive, optimizer, emitters, stats, save_dir, policies=None):
    with open(os.path.join(save_dir, 'learn.pickle', 'wb')) as handle:
        dict = {
            'archive': archive,
            'optimizer': optimizer,
            'emitters': emitters,
            'stats': stats,
            'policies': policies,
        }
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def qdpy_eval(env, generator, weights):
    weights = np.array(weights)
    obj, bcs = simulate(generator, weights, env)
    return (obj,), bcs
