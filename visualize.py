import os
import numpy as np
from matplotlib import pyplot as plt
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

def plot_fitness_qdpy(save_dir, logbook, quality_diversity=True):
    gen = logbook.select("iteration")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")
    fit_stds = logbook.select("std")
    fit_maxs = logbook.select("max")

    fig, ax1 = plt.subplots()
    line0 = ax1.plot(gen, fit_mins, "b--")
    #  line1_err = ax1.errorbar(gen, fit_avgs, fit_stds[:,0], color='green', mfc='green', mec='green', linestyle="-",
    #                           label="Average Fitness",
    #                           ms=20, mew=4,
    #                           alpha=min(0.9, 100 / len(gen)),
    #                           # alpha=0.9,
    #                           )
    line1 = ax1.plot(gen, fit_avgs, 'b-', label='Average Fitness')
    line2 = ax1.plot(gen, fit_maxs, "b--")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    # FIXME: figure out from logbook if we've got all-1 bin sizes so we don't plot size
    # if not np.all(self.config.ME_BIN_SIZES == 1):
    if quality_diversity:
        # plot the size of the archive
        containerSize_avgs = logbook.select('containerSize')
        for tl in ax1.get_yticklabels():
            tl.set_color("b")
        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, containerSize_avgs, "r-", label="Archive Size")
        ax2.set_ylabel("Size", color="r")
        # ax2_ticks = ax2.get_yticklabels()
        start, end = ax2.get_ylim()
        ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
        for tl in ax2.get_yticklabels():
            tl.set_color("r")
        lns = line1 + line2
    else:
        lns = line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'fitness.png'))

