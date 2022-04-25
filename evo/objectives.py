import numpy as np
from pdb import set_trace as TT

from envs.maze.swarm import toroidal_distance

def regret_fitness(regret_loss):
    return regret_loss


def eval_fit(ps, scape):
    ps = np.floor(ps).astype(int)
    scores = scape[ps[0, ...], ps[1, ...]]
    return scores


def contrastive_pop(ps, width):
    # dists = np.sqrt(((ps[0][None, ...] - ps[1][:, None, ...]) ** 2).sum(-1))
    # reward distance between population types
    # tor_dists = np.sqrt(((toroidal_distance(ps[0][None,...], ps[1][:, None, ...]))**2).sum(-1))
    # FIXME: this is wrong
    inter_dist_means = [(toroidal_distance(pi, pj, width) ** 2).sum(-1).mean()
                        for i, pi in enumerate(ps)
                        for j, pj in enumerate(ps) if i != j]
    # penalize intra-pop distance (want clustered populations)
    intra_dist_means = [(toroidal_distance(p, p, width) ** 2).sum(-1).mean() for p in ps]
    if len(intra_dist_means) == 0:
        intra_dist_means = [0]
    n_intra = len(ps)
    assert len(inter_dist_means) == n_intra * (n_intra - 1)
    fit = np.mean(inter_dist_means) - np.mean(intra_dist_means)
    assert fit is not None
    return fit


def contrastive_fitness(rews):
    rews = [np.array(r) for r in rews]
    inter_dist_means = [np.abs(rew[None, ...] - fj[:, None, ...]).mean() for i, rew in enumerate(rews) for j, fj in
                        enumerate(rews) if i != j]

    # If only one agent per population, do not calculate intra-population distance.
    if rews[0].shape[0] == 1:
        inter_dist_mean = np.mean(inter_dist_means)
        return inter_dist_mean

    intra_dist_means = [np.abs(fi[None, ...] - fi[:, None, ...]).sum() / (fi.shape[0] * (fi.shape[0] - 1)) for fi in
                        rews]
    obj = np.mean(inter_dist_means) - np.mean(intra_dist_means)
    return obj


# # Deprecated. We would be accepting fitnesses as an argument, in theory.
# def fit_dist(pops, scape):
#     '''n-1 distances in mean fitness, determining the ranking of n populations.'''
#     assert len(pops) == 2
#     inter_dist = [eval_fit(pi.ps, scape).mean() - eval_fit(pj.ps, scape).mean()
#                   for j, pj in enumerate(pops) for pi in pops[j + 1:]]


def min_solvable_fitness(rews, max_rew, trg_rew=0):
    """ A fitness function rewarding levels that result in the least non-zero reward.
    :param rews: a list of lists of rewards achieved by distinct agents from distinct populations (length 1 or greater)
    :param max_rew: An uppser bound on the maximum reward achievable by a given population. Note this should not 
        actually be achievable by the agent, or impossible episodes will rank equal to extremely easy ones. Though the
        former is worse in principle.
    :return: a fitness value
    """
    assert len(rews) >= 1
    rews = np.array(rews)
    rews = np.mean(rews, axis=1)  # get mean per-population rewards
    if np.all(rews == 0):
        # return 0
        return - max_rew
    else:
        # return max_rew - np.mean(rews)
        return - np.abs(np.mean(rews) - trg_rew)


def contrastive_fitness(rews):
    rews = [np.array(r) for r in rews]
    inter_dist_means = [np.abs(rew[None, ...] - fj[:, None, ...]).mean() for i, rew in enumerate(rews) for j, fj in
                        enumerate(rews) if i != j]

    # If only one agent per population, do not calculate intra-population distance.
    if rews[0].shape[0] == 1:
        inter_dist_mean = np.mean(inter_dist_means)
        return inter_dist_mean

    intra_dist_means = [np.abs(fi[None, ...] - fi[:, None, ...]).sum() / (fi.shape[0] * (fi.shape[0] - 1)) for fi in
                        rews]
    obj = np.mean(inter_dist_means) - np.mean(intra_dist_means)

    return obj


def paired_fitness(rews):
    """A PAIRED-type objective. Seeks to maximize the advantage of the first policy over all others."""
    rews = np.array([np.array(r) for r in rews])
    rew_a = np.mean(rews[0])
    rew_b = np.mean(rews[1:])
    obj = rew_a - rew_b

    return obj