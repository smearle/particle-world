import os
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as TT
from ribs.visualize import grid_archive_heatmap


def visualize_pyribs(archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive)
    plt.gca().invert_yaxis()
    # plt.xlabel("Symmetry")
    # plt.ylabel("Emptiness")
    plt.xlabel("pop1 mean x")
    plt.ylabel("pop1 mean y")
    plt.savefig("fitness.png")
    plt.close()

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
    line2 = ax1.plot(gen, mean_eval_rews, "r-", label="Mean Eval Reward")
    # line3 = ax2.plot(gen, max_eval_rews, "r--")
    ax1.set_ylabel("Agent Evaluation Reward")
    # ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
    lns = line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'agent_eval_reward.png'))

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