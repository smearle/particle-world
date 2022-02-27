import argparse
from collections import namedtuple
import json
import os

import pandas as pd
from utils import get_experiment_name


def vis_cross_eval(exp_configs):
    exp_names = []
    train_stats = []
    col_headers = None
    data_rows = []
    for exp_config in exp_configs:
        args = namedtuple('args', exp_config.keys())(**exp_config)
        # parser = argparse.ArgumentParser()
        # args = parser.parse_args()
        # [setattr(args, k, v) for k, v in exp_config.items()]
        exp_name = get_experiment_name(args)
        exp_save_dir = os.path.join('runs', exp_name)
        if not os.path.isdir(exp_save_dir):
            print(f'No directory found for experiment {exp_name}. Skipping.')
            continue
        exp_names.append(exp_name)
        with open(os.path.join(exp_save_dir, 'train_stats.json'), 'r') as f:
            exp_train_stats = json.load(f)
        train_stats.append(exp_train_stats)
        if col_headers is None:
            col_headers = list(exp_train_stats.keys())
        data_rows.append([exp_train_stats[k] for k in col_headers])
    df = pd.DataFrame(data_rows, columns=col_headers, index=exp_names)
    df.to_latex('cross_eval.tex')
    os.system('pdflatex tables.tex')
    