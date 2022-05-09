import argparse
from collections import namedtuple
import json
import os
import numpy as np
from pdb import set_trace as TT

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from utils import get_experiment_name

EVAL_DIR = 'runs_eval'


def vis_cross_eval(exp_names):
    stats = []
    col_headers = None
    data_rows = []
    plot_names = set({})
    bar_names = []
    plot_vals = {}
    for exp_name in exp_names:
        exp_save_dir = os.path.join('runs', exp_name)
        if not os.path.isdir(exp_save_dir):
            print(f'No directory found for experiment {exp_name}. Skipping.')
            continue
        try:
            with open(os.path.join(exp_save_dir, 'train_stats.json'), 'r') as f:
                exp_train_stats = json.load(f)
                last_exp_train_stats = exp_train_stats
        except FileNotFoundError:
            # exp_train_stats = {k: None for k in last_exp_train_stats}
            exp_train_stats = {}

        try:
            with open(os.path.join(exp_save_dir, 'eval_stats.json'), 'r') as f:
                exp_eval_stats = json.load(f)
                last_exp_eval_stats = exp_eval_stats
        except FileNotFoundError:
            exp_eval_stats = {k: {k1: {k2: -0.01 for k2 in v1} for k1, v1 in v.items()} for k, v in last_exp_eval_stats.items()}

        # Flatten the dict of dict of dicts into a dict for eval stats.
        flat_eval_stats = {}
        for widx, world_key in enumerate(exp_eval_stats.keys()):
            for policy_key in exp_eval_stats[world_key]:
                bar_name = f'{exp_name} {policy_key}'
                if widx == 0:
                    bar_names.append(bar_name)
                for stat_key in exp_eval_stats[world_key][policy_key]:
                    stat_lst = exp_eval_stats[world_key][policy_key][stat_key]
                    mean_stat = np.mean(stat_lst)
                    std_stat = np.std(stat_lst)
                    flat_eval_stats[f"{world_key} {policy_key} {stat_key}"] = mean_stat
                    plot_name = f"{world_key} {stat_key}"
                    if plot_name not in plot_names:
                        plot_names.add(plot_name)
                        plot_vals[plot_name] = {}
                    plot_vals[plot_name][bar_name] = {'mean': mean_stat, 'std': std_stat}

        # Get mean performance over all worlds.
        plot_vals['all_worlds'] = {}
        [plot_vals['all_worlds'].update({
            bar_name: {
                'mean': np.mean([world_stats[bar_name]['mean'] \
                    for world_stats in plot_vals.values() if bar_name in world_stats]), 
                'std': np.mean([world_stats[bar_name]['std'] \
                    for world_stats in plot_vals.values() if bar_name in world_stats])}}) \
                        for bar_name in bar_names]

        # Assume there's no identical keys appearing in these two dictionaries.
        exp_stats = {**exp_train_stats, **flat_eval_stats}

        # If we have no column headers, create them based on the keys of this dictionary (assuming we have logged the 
        # same stats for all experiments).
        stats.append(exp_stats)
        if col_headers is None:
            col_headers = list(exp_stats.keys())

        data_rows.append([exp_stats[k] if k in exp_stats else None for k in col_headers])

    color_keys = ['qd', 'min_solvable', 'regret', 'contrastive', 'paired', 'fixedWorlds']
    color_names = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    legend_elements = [Patch(color=color_names[i], label=k) for i, k in enumerate(color_keys)]
    color_map = {k: color_names[i] for i, k in enumerate(color_keys)}
    colors = []

    for bn in bar_names:
        color_found = False
        for ck in color_keys:
            if ck in bn:
                colors.append(color_map[ck])
                color_found = True
                break
        if not color_found:
            raise Exception(f'No color found for {bn}.')

    # eval_keys = flat_eval_stats.keys()
    for plot_name in plot_vals:
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        # n_subplots = len(plot_keys)
        # n_subplots = 1
        vals = [plot_vals[plot_name][bar_name]['mean'] for bar_name in bar_names]
        errs = [plot_vals[plot_name][bar_name]['std'] for bar_name in bar_names]
        plt.barh(bar_names, vals, xerr=errs, color=colors)
        ax.set_title(plot_name)
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(f'{EVAL_DIR}/{plot_name}.png')
        
    df = pd.DataFrame(data_rows, columns=col_headers, index=exp_names)
    df.to_latex(os.path.join(EVAL_DIR, 'cross_eval.tex'))
    proj_dir = os.curdir
    os.system(f'cd {EVAL_DIR}; pdflatex tables.tex; cd {proj_dir}')
    