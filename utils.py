import json
import os
import pickle
from pdb import set_trace as TT
from pprint import PrettyPrinter
# from ribs.visualize import grid_archive_heatmap
from timeit import default_timer as timer

import numpy as np
import pygame
import ray
from matplotlib import pyplot as plt
from pygame.constants import KEYDOWN
import torch as th


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


# def animate_nca(generator):
#     #   cv2.namedWindow('NCA world generation')
#     generator.generate(render=True, pg_delay=pg_delay)


def simulate(generator, g_weights, env, n_steps=100, n_eps=1, render=False, screen=None, pg_delay=1, pg_scale=1):
    generator.set_weights(g_weights)
    all_fit_difs = np.empty((n_eps))
    all_bcs = np.empty((n_eps, 2))
    for i in range(n_eps):
        generator.generate(render=render, screen=screen, pg_delay=pg_delay)
        env.set_landscape(generator.world)
        env.reset()
        fit_difs, bcs = env.simulate(n_steps=n_steps, generator=generator, render=render, screen=screen,
                                     pg_delay=pg_delay, pg_scale=pg_scale)
        all_fit_difs[i] = fit_difs
        all_bcs[i] = bcs
    #   obj = -np.std(all_fit_difs)
    #   bcs = np.mean(all_fit_difs)
    obj = np.mean(all_fit_difs)
    # obj = symmetry(generator.world)
    # bcs = [symmetry(generator.world), emptiness(generator.world)]
    bcs = all_bcs.mean(0) / env.width
    return obj, bcs


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

def discrete_to_onehot(a, n_chan=None):
    n_chan = a.max() + 1 if not n_chan else n_chan
    return np.eye(n_chan)[a].transpose(2, 0, 1)


adj_coords_2d = np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
])


def get_solution(arr, passable=0, impassable=1, src=2, trg=3):
    width = arr.shape[0]
    assert width == arr.shape[1]
    srcs = np.argwhere(arr == src)
    assert srcs.shape[0] == 1
    src = tuple(srcs[0])
    frontier = [src]
    back_paths = {}
    visited = set({})
    while frontier:
        curr = frontier.pop(0)
        if arr[curr[0], curr[1]] == trg:
            path = []
            path.append(curr)
            while curr in back_paths:
                curr = back_paths[curr]
                path.append(curr)
            return path[::-1]
        visited.add(curr)
        adjs = [tuple((np.array(curr) + move) % width) for move in adj_coords_2d]
        for adj in adjs:
            if adj in visited or arr[adj[0], adj[1]] == impassable:
                continue
            frontier.append(adj)
            back_paths.update({adj: curr})
    return []

# start_time = timer()
# print(get_solution(np.array([
#     [1, 1, 1, 0, 1],
#     [1, 2, 0, 0, 1],
#     [1, 1, 1, 0, 1],
#     [1, 3, 0, 0, 1],
#     [1, 1, 1, 0, 1],
# ])))
# print(timer() - start_time)


def update_individuals(individuals, qd_stats):
    qd_stats = [qd_stats[k] for k in range(len(qd_stats))]
    # print(f"Updating individuals with new qd stats: {qd_stats}.")

    for ind, s in zip(individuals, qd_stats):
        ind.fitness.values = s[0]
        ind.features = s[1]
        if ind.playability_penalty > 0:
            ind.fitness.values = (- 10 - ind.playability_penalty, )
            ind.features = [0, 0]


def get_experiment_name(cfg):
    if cfg.fixed_worlds:
        exp_name = f'fixedWorlds_{cfg.n_policies}-pol'
    else:
        exp_name = f'{cfg.generator_class}_'
        if cfg.quality_diversity:
            exp_name += 'qd'
        else:
            exp_name += f'{cfg.objective_function}'
        exp_name += f'{cfg.n_policies}-pol_{cfg.gen_phase_len}-gen_{cfg.play_phase_len}-play'
    if cfg.fully_observable:
        exp_name += '_fullObs'
    if cfg.model is not None:
        exp_name += f'_mdl-{cfg.model}'
    exp_name += f'_{cfg.exp_name}'
    return exp_name


def load_config(args, config_file):
    config_file = os.path.join('configs', 'auto', f'{config_file}.json')
    with open(config_file, 'r') as f:
        new_args = json.load(f)
    pp = PrettyPrinter(indent=4)
    print(f'Loaded config:')
    pp.pprint(new_args)
    for k, v in new_args.items():
        assert hasattr(args, k), f"Attempting to write a config parameter, {k}, from run_batch.py which is not present in args.py"
        setattr(args, k, v)
    return args


def filter_nones(lst):
    return [l for l in lst if l is not None]


def compile_train_stats(save_dir, logbook, net_itr, gen_itr, play_itr, quality_diversity=False):
    stats = {
        'net_itr': net_itr,
        'gen_itr': gen_itr,
        'play_itr': play_itr,
        'meanRew': np.mean(filter_nones(logbook.select('meanRew'))[-10:]),
        'meanEvalRew': np.mean(filter_nones(logbook.select('meanEvalRew'))[-10:]),
        'meanPath': np.mean(filter_nones(logbook.select('meanPath'))[-10:]),
        'maxPath': np.mean(filter_nones(logbook.select('maxPath'))[-10:]),
        'meanFit': np.mean(filter_nones(logbook.select('avg'))[-10:]),
    }
    with open(os.path.join(save_dir, 'train_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
