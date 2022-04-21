import os
import math
import sys

import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as TT
from PIL import Image

from qdpy.plots import plotGridSubplots
# from ribs.visualize import grid_archive_heatmap


# def visualize_pyribs(archive):
#     plt.figure(figsize=(8, 6))
#     grid_archive_heatmap(archive)
#     plt.gca().invert_yaxis()
#     # plt.xlabel("Symmetry")
#     # plt.ylabel("Emptiness")
#     plt.xlabel("pop1 mean x")
#     plt.ylabel("pop1 mean y")
#     plt.savefig("fitness.png")
#     plt.close()


def visualize_train_stats(save_dir, logbook, quality_diversity=True):
    gen = logbook.select("iteration")
    fit_mins = remove_nones(logbook.select("min"))
    fit_avgs = remove_nones(logbook.select("avg"))
    fit_stds = remove_nones(logbook.select("std"))
    fit_maxs = remove_nones(logbook.select("max"))

    fig, ax1 = plt.subplots()
    #  line1_err = ax1.errorbar(gen, fit_avgs, fit_stds[:,0], color='green', mfc='green', mec='green', linestyle="-",
    #                           label="Average Fitness",
    #                           ms=20, mew=4,
    #                           alpha=min(0.9, 100 / len(gen)),
    #                           # alpha=0.9,
    #                           )
    # line0 = ax1.plot(gen, fit_mins, "b--")
    line1 = ax1.plot(gen, fit_avgs, 'b-', label='Mean Fitness')
    # line2 = ax1.plot(gen, fit_maxs, "b--")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    # FIXME: figure out from logbook if we've got all-1 bin sizes so we don't plot size
    # if not np.all(self.config.ME_BIN_SIZES == 1):
    if quality_diversity:
        # plot the size of the archive
        containerSize_avgs = remove_nones(logbook.select('containerSize'))
        for tl in ax1.get_yticklabels():
            tl.set_color("b")
        ax2 = ax1.twinx()
        line3 = ax2.plot(gen, containerSize_avgs, "r-", label="Archive Size")
        ax2.set_ylabel("Size", color="r")
        # ax2_ticks = ax2.get_yticklabels()
        start, end = ax2.get_ylim()
        ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
        for tl in ax2.get_yticklabels():
            tl.set_color("r")
        lns = line1 + line3
    else:
        lns = line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'fitness.png'))

    fig, ax1 = plt.subplots()
    path_means = remove_nones(logbook.select("meanPath"))
    path_maxs = remove_nones(logbook.select("maxPath"))
    line0 = ax1.plot(gen, path_means, "b-", label="Mean Path Length")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Path Length")
    lns = line0
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'path_length.png'))

    fig, ax1 = plt.subplots()
    mean_rews = remove_nones(logbook.select("meanRew"))
    min_rews = remove_nones(logbook.select("minRew"))
    max_rews = remove_nones(logbook.select("maxRew"))
    # line1 = ax2.plot(gen, min_rews, "r--")
    line2 = ax1.plot(gen, mean_rews, "r-", label="Mean Reward")
    # line3 = ax2.plot(gen, max_rews, "r--")
    ax1.set_ylabel("Agent Reward")
    # ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
    lns = line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'agent_reward.png'))

    fig, ax1 = plt.subplots()
    mean_eval_rews = remove_nones(logbook.select("meanEvalRew"))
    min_eval_rews = remove_nones(logbook.select("minEvalRew"))
    max_eval_rews = remove_nones(logbook.select("maxEvalRew"))
    # line1 = ax2.plot(gen, min_eval_rews, "r--")
    line2 = ax1.plot(gen, smooth(mean_eval_rews, 100), "r-", label="Mean Eval Reward")
    # line3 = ax2.plot(gen, max_eval_rews, "r--")
    ax1.set_ylabel("Agent Evaluation Reward")
    # ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
    lns = line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'agent_eval_reward.png'))


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def remove_nones(l):
    iv = 0
    nl = []
    for li in l:
        if li is not None:
            nl.append(li)
            iv = li
        else:
            nl.append(iv)
    return nl


def visualize_archive(cfg, env, grid):
    # visualize current worlds
    gg = sorted(grid, key=lambda i: i.features)
    world_im_width = cfg.width * 10

                # if doing QD, render a grid of 1 world per cell in archive
    if cfg.quality_diversity:
        nb_bins = grid.shape
        world_im_width = cfg.width * 10
        im_grid = np.zeros((world_im_width * nb_bins[0], world_im_width * nb_bins[1], 3))
        for g in gg:
            i, j = grid.index_grid(g.features)
            env.set_world(g.discrete)
            env.reset()
            im = env.render(mode='rgb', pg_width=world_im_width, render_player=True)
            im_grid[i * world_im_width: (i + 1) * world_im_width,
                        j * world_im_width: (j + 1) * world_im_width] = im

                # otherwise, render a grid of elite levels
    else:
        gg = sorted(gg, key=lambda ind: ind.fitness[0], reverse=True)
        fits = [g.fitness[0] for g in gg]
        max_fit = max(fits)
        min_fit = min(fits)
        # assert nb_bins == (1, 1) == grid.shape
        max_items_per_bin = len(grid)
        n_world_width = math.ceil(math.sqrt(max_items_per_bin))
        im_grid = np.zeros((world_im_width * n_world_width, world_im_width * n_world_width, 3))
        for gi, g in enumerate(gg):
            i, j = gi // n_world_width, gi % n_world_width
            env.queue_worlds({0: g.discrete})
            env.reset()
            im = env.render(mode='rgb', pg_width=world_im_width, render_player=True)
            im_grid[j * world_im_width: (j + 1) * world_im_width,
                        i * world_im_width: (i + 1) * world_im_width] = im

                        # To visualize ranking of fitnesses
            im_grid[j * world_im_width: j * world_im_width + 7,
                        int((i + 0.5) * world_im_width): int((i + 0.5) * world_im_width) + 7] = int(
                            255 * (g.fitness[0] - min_fit) / (max_fit - min_fit))

    im_grid = im_grid.transpose(1, 0, 2)
                # im_grid = np.flip(im_grid, 0)
                # im_grid = np.flip(im_grid, 1)
    im_grid = Image.fromarray(im_grid.astype(np.uint8))
    im_grid.save(os.path.join(cfg.save_dir, "level_grid.png"))