import math
from pdb import set_trace as TT

import numpy as np
import torch as th
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import TensorType
from torch import nn


def gen_policy(i, observation_space, action_space, field_of_view):
    config = {
        "model": {
            "custom_model_config": {
                "field_of_view": field_of_view,
            }
        }
    }
    return PolicySpec(config=config, observation_space=observation_space, action_space=action_space)

th.set_printoptions(profile='full')


class Swarm(object):
    def __init__(self, n_pop, field_of_view=1, trg_scape_val=1.0):
        assert field_of_view > 0
        self.ps = None
        self.vs = None
        self.field_of_view = field_of_view
        self.n_pop = n_pop
        self.trg_scape_val = trg_scape_val

    def reset(self, scape, n_pop):
        # self.world = scape
        self.n_pop = n_pop
        self.ps, self.vs = init_ps(self.world_width, n_pop)

    def get_full_observations(self, scape, flatten=False):
        # Add a 4th channel for player positions
        scape = np.vstack((scape, np.zeros_like(scape)[0:1]))
        obs = np.tile(scape[None,...], (self.n_pop, 1, 1, 1))

        # NOTE: don't change this, the environment assumes the last (4th) channel is for player positions
        obs[np.arange(self.n_pop), -1, self.ps[:, 0].astype(int), self.ps[:, 1].astype(int)] = 1
        assert obs[:, -1, :, :].sum() == self.n_pop

        return obs.transpose(0, 2, 3, 1)

    def get_observations(self, scape, flatten=True, ps=None, padding_mode='wrap', surplus_padding=0):
        """
        Return a batch of local observations corresponding to particles in the swarm, of size (n_pop, patch_w, patch_w),
        where patch_w is a square patch allowing the agent to see field_of_view (field of vision)-many tiles in each direction.
        :param scape: The one-hot encoded and/or discrete world observed by the agents, of size (n_chan, width,
        width)
        :param flatten: If true, return obs of size (n_pop, patch_w ** 2), i.e. for processing by a dense layer.
        :return:
        """
        ps = self.ps if ps is None else ps
        field_of_view = self.field_of_view
        patch_w = int(field_of_view * 2 + 1)
        # Add new dimensions for patch_w-width patches of the environment around each agent
        ps_int = np.floor(ps).astype(int)
        # weird edge case, is modulo broken?
        ps_int = np.where(ps_int == self.world_width, 0, ps_int)
        padding = field_of_view + surplus_padding
        constant_values = {'constant_values': 0} if padding_mode == 'constant' else {}
        # TODO: this is discretized right now. Maybe it should use eval_fit instead to take advantage of continuity?
        landscape = np.pad(scape, ((0, 0), (padding, padding), (padding, padding)), mode=padding_mode, **constant_values)
        # Padding makes this indexing a bit weird but these are patch_w-width neighborhoods, centered at provided positions
        nbs = [landscape[:, pxi + padding - field_of_view: pxi + 1 + padding + field_of_view, pyi + padding - field_of_view:pyi + 1 + padding + field_of_view] for pxi, pyi in
               zip(ps_int[:, 0], ps_int[:, 1])]
        nbs = np.stack(nbs)
        # (agents, channels, width, height)
        # nbs = nbs[:, None, ...]
        if flatten:
            nbs = np.reshape(nbs, (nbs.shape[0], nbs.shape[1], -1))
        return nbs.transpose(0, 2, 3, 1)

    def get_rewards(self, scape):
        ps = self.ps.astype(int)
        return 1 - np.abs(self.trg_scape_val - scape[:, ps[:, 0], ps[:, 1]])


def init_ps(world_width, n_pop, n_dim=2):
    # velocities between -1 and 1
    vls = np.zeros((n_pop, n_dim))
    if n_pop == 1:
        ps = np.array(
            [[world_width / 2, world_width / 2]]
        )
    else:
        # vls = (np.random.random(size=(n_pop, n_dim)) - 0.5) * 1
        # ps = np.random.random(size=(n_pop, n_dim)) * width
        x = np.linspace(0, 2 * np.pi, n_pop + 1)[:-1, None]
        ps = (np.hstack((np.sin(x), np.cos(x))))
        ps = ps * world_width / 4
        ps -= world_width / 2
        # ps = ps / np.sqrt((ps ** 2).sum(axis=-1))[:, None]
        # ps += width / 2
        ps = ps % world_width
    return ps.astype(np.float64), vls


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         m.weight.data.uniform_(-1.0, 1.0)
#         m.bias.data.fill_(0.01)
#     if isinstance(m, nn.Conv2d):
#         m.weight.data.uniform_(-1.0, 1.0)
#         m.bias.data.fill_(0.01)


class NN(nn.Module):
    def __init__(self, field_of_view):
        super().__init__()
        kernel_width = 2 * field_of_view + 1
        self.l1 = nn.Conv2d(1, 32, kernel_width, 1, padding=0)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(32, 6)  # double length of actual output because TorchDiagGaussian computes mean and std
        # https://github.com/ray-project/ray/issues/17934
        self.l4 = nn.Linear(32, 1)
        # self.apply(init_weights)

    def forward(self, x):
        x = th.relu(self.l1(x))
        x = x.view(x.shape[0], -1)
        # x = th.sigmoid(self.l2(x))
        act = th.sigmoid(self.l3(x))
        act = act * 2 - 1
        self.val = self.l4(x)
        hids = {}
        return act, hids


class RLlibNN(NN, TorchModelV2):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **customized_model_kwargs):
        ModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name, framework='torch')
        field_of_view = model_config["custom_model_config"]["field_of_view"]
        NN.__init__(self, field_of_view=field_of_view)
        self.tower_stats = {}

    def forward(self, input_dict, state_batches=None, seq_lens=None):
        obs = input_dict['obs']
        ret = super().forward(obs)
        return ret

    def value_function(self) -> TensorType:
        return self.val.squeeze(-1)


class NeuralSwarm(Swarm):
    """ Handles actions as might be output by a neural network, but assumes the neural network is defined outside the
    environment/swarm."""
    def __init__(self, world_width, n_pop: int, field_of_view: int = 4, trg_scape_val=1.0):
        super().__init__(n_pop, field_of_view, trg_scape_val=trg_scape_val)
        self.world_width = world_width

        # self.nn = NN(field_of_view=field_of_view)
        self.nn = None

    def set_nn(self, nn, policy_id, obs_space, act_space, trainer_config):
        # self.nn = type(nn)(obs_space, act_space, trainer_config['model'])
        self.nn.model = type(nn.model)(obs_space, act_space, nn.model.num_outputs, trainer_config['model'],
                                       f'policy_{policy_id}')
        # self.nn.set_state(copy.deepcopy(nn.get_state()))
        # self.nn.set_weights(copy.deepcopy(nn.get_weights()))
        # self.nn.model.load_state_dict(copy.copy(nn.model.state_dict()))
        # self.nn.__delattr__ = None
        # self.nn.model_gpu_towers = None
        # attrs = dir(self.nn)
        # for k in self.nn.__dict__:
        #
        # for attr in attrs:
        # at = getattr(self.nn, k)
        # print('at', k, at)
        # copy.deepcopy(at)
        return

    def update(self, scape, actions, obstacles):
        if actions is None:
            if self.nn is None:
                print("Generating random NN swarm policies inside environment, because no accelerations were supplied "
                      "when updating swarms.")
            nbs = self.get_observations(scape=scape)
            # actions, hid_state = self.nn({'obs': th.Tensor(nbs)})
            actions, hid_state = self.nn(th.Tensor(nbs))
            actions = actions.detach().numpy()
            actions = np.hstack((actions[:, :3].argmax(1)[..., None], actions[:, 3:].argmax(1)[..., None]))
        if obstacles is not None:
            # TODO: proper collision detection (started in standalone function update_with_collision)
            # For now, we assume agent can only encounter one new wall per step
            new_ps = (self.ps + actions).astype(np.uint8) % self.world_width
            colls = obstacles[new_ps[:, 0], new_ps[:, 1]]
            self.ps = np.where(colls[..., None] == 1, self.ps, new_ps)
        else:
            self.ps += actions
            # self.ps += self.vs
        self.ps = self.ps % self.world_width
        # momentum = 0.5
        # self.vs += (accelerations - 1) * momentum
        # self.vs /= 1 + 0.1 * momentum


class MazeSwarm(NeuralSwarm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_reward = 1.0

        # TODO:  use max_steps to make this exact.
        self.min_reward = 0.0 

    def get_rewards(self, scape, goal_idx, n_step, max_steps):
        """Calculates rewards for each agent in the swarm."""
        assert goal_idx is not None
        ps = self.ps.astype(int)
#       rewards = (scape[3, ps[:, 0], ps[:, 1]] == 1).astype(np.int)

        # All players currently on a goal tile will receive reward
        rewards = (scape[goal_idx, ps[:, 0], ps[:, 1]] == 1)

        # Reward decreases expenontially with time taken to reach goal.
        rewards = np.where(rewards, 1 - 0.9 * (n_step / max_steps), 0)

        return rewards

    def update(self, scape, actions, obstacles):
        super().update(scape, actions, obstacles=obstacles)


dirs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])


class DirectedMazeSwarm(MazeSwarm):

    def __init__(self, *args, **kwargs):
        self.directions = None
        super().__init__(*args, **kwargs)
    
    def reset(self, *args, **kwargs):
        self.directions = np.zeros(self.n_pop, dtype=int)

        # DEBUG rotation
        # self.directions[:] = 3

        super().reset(*args, **kwargs)

    def get_observations(self, scape, flatten=True, padding_mode='wrap', surplus_padding=0):
        # ps = self.ps + (dirs[self.directions] * self.field_of_view)
        ps = self.ps
        map_obs = super().get_observations(scape, flatten, ps=ps, padding_mode=padding_mode, surplus_padding=surplus_padding)

        # TODO: this would give separate map/direction observations, see TODO in DirectedMazeEnv
#       obs = [{'map': mo, 'direction': d} for mo, d in zip(map_obs, self.directions)]

        # For now we'll add a onehot-encoded direction to the map, beneath the player (should it occupy the whole layer?)
        direction_layers = np.zeros((*map_obs.shape[:-1], 4), dtype=map_obs.dtype)
        direction_layers[np.arange(self.n_pop), self.field_of_view, self.field_of_view, self.directions] = 1

        # lining up render with printout
        map_obs = np.flip(map_obs, axis=1)  # flip along x axis

        obs = np.concatenate((map_obs, direction_layers), axis=-1)

        # Rotate observation to match each agent's orientation (we've offset to make it look right when printing local 
        # observations and rendering the environment.)
        for i, d in enumerate(self.directions):

            obs[i] = np.rot90(obs[i], k=d, axes=(0, 1))

        # DEBUG: rotation
#       v_obs = obs.transpose(0, 3, 1, 2)
#       print(v_obs[0,0])
#       print()
#       print(v_obs[0,2])

        return obs

    def update(self, scape, actions, obstacles):
        """Update the positions of agents in the swarm.
        
        Args:
            scape (np.ndarray): The landscape of the environment.
            actions (np.ndarray): An array of size (n_pop, 1) containing discrete actions in range(4), corresponding to
                [turn right, turn left, move forward, stay still].
            obstacles (np.ndarray): An array with the same shape as ``scape``, containing 1 for obstacles and 0 for 
                passable tiles. Agents who attempt to move onto an obstacle will stay in their current position.
        """
        # NOTE: these accelerations actually . We now 
        # convert them to accelerations where appropriate.

        # DEBUG:
#       accelerations[:] = 3

        # get rid of this extra (normally (x, y)) dimension
        actions = actions[:, 0]

        # Turn right (-1 just a placeholder, will be overwritten in this function)
        self.directions = np.where(actions == 0, (self.directions - 1) % 4, self.directions)

        # Turn left
        self.directions = np.where(actions == 1, (self.directions + 1) % 4, self.directions)

        # Move forward, otherwise stay put. The resulting acceleration depends on the agent's current direction.
        accelerations = np.where(np.stack((actions, actions), axis=-1) == 2, dirs[self.directions], (0, 0))

        # The parent class accepts accelerations as actions.
        super().update(scape, actions=accelerations, obstacles=obstacles)


class GreedySwarm(Swarm):
    def __init__(self, world_width, n_pop: int, field_of_view: int = 4, trg_scape_val=1.0):
        super().__init__(n_pop)
        assert field_of_view > 0
        self.field_of_view = field_of_view
        self.world_width = world_width
        self.trg_scape_val = trg_scape_val

    def reset(self, scape):
        super().reset(scape)
        self.ps, self.vs = init_ps(self.world_width, self.n_pop)

    def update(self, scape, obstacles):
        # if obstacles is not None:
        #     update_pos_with_collision(self.ps, self.vs, obstacles)
        # else:
        #     self.ps += self.vs
        # self.ps = self.ps % self.world_width
        field_of_view = self.field_of_view
        patch_w = field_of_view * 2 + 1
        # Add new dimensions for patch_w-width patches of the environment around each agent
        ps_int = np.floor(self.ps).astype(int)
        # weird edge case, is modulo broken?
        ps_int = np.where(ps_int == self.world_width, 0, ps_int)
        # TODO: this is discretized right now. Maybe it should use eval_fit instead to take advantage of continuity?
        landscape = np.pad(scape, field_of_view, mode='wrap')
        # Padding makes this indexing a bit weird but these are patch_w-width neighborhoods
        nbs = [landscape[pxi:pxi + 1 + 2 * field_of_view, pyi:pyi + 1 + 2 * field_of_view] for pxi, pyi in zip(ps_int[:, 0], ps_int[:, 1])]
        nbs = np.stack(nbs)
        nbs[..., 1:-1, 1:-1] = -1  # observe only the periphery of our field of vision
        # nbs += np.random.normal(0, 0.1, (nbs.shape))
        flat_ds = np.abs(nbs.reshape(self.n_pop, -1) - self.trg_scape_val).argmin(axis=-1)
        ds = np.stack((flat_ds // patch_w, flat_ds % patch_w)).swapaxes(1, 0)
        ds = (ds - field_of_view) / field_of_view * 1.0

        self.ps += ds
        self.ps = self.ps % self.world_width

        # momentum = 0.5
        # self.vs += ds * momentum
        # self.vs /= 1 + 0.1 * momentum


class MemorySwarm(Swarm):
    def __init__(self, n_pop, xplor, xploit):
        super().__init__(n_pop)
        self.r_xplor = xplor
        self.r_xploit = xploit

    def reset(self, scape):
        super().reset(scape)
        self.ps, self.vs = init_ps(self.n_pop)
        self.i_vals = eval_fit(self.ps.swapaxes(1, 0), scape)
        self.i_bests = self.ps.copy()
        g_best_idx = np.argmax(self.i_vals)
        self.g_best = self.i_bests[g_best_idx]
        self.g_val = self.i_vals[g_best_idx]

    def update(self, scape):
        self.ps += self.vs
        self.ps = self.ps % self.world_width
        fits = eval_fit(self.ps.swapaxes(1, 0), scape)
        i_idxs = np.where(fits > self.i_vals)
        self.i_bests[i_idxs, :] = self.ps[i_idxs, :]
        self.i_vals[i_idxs] = fits[i_idxs]
        g_best_idx = np.argmax(self.i_vals)
        self.g_best = self.i_bests[g_best_idx]
        self.g_val = self.i_vals[g_best_idx]
        nst = np.random.random(self.vs.shape)
        dsr = np.random.random(self.vs.shape)
        self.vs += toroidal_distance(self.i_bests, self.ps, self.world_width) * nst * self.r_xplor
        self.vs += toroidal_distance(self.g_best, self.ps, self.world_width) * dsr * self.r_xploit
        self.vs /= 2
        # nnbs = nnb(self.ps)


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


def contrastive_fitness(fits):
    fits = [np.array(f) for f in fits]
    inter_dist_means = [np.abs(fi[None, ...] - fj[:, None, ...]).mean() for i, fi in enumerate(fits) for j, fj in
                        enumerate(fits) if i != j]

    # If only one agent per population, do not calculate intra-population distance.
    if fits[0].shape[0] == 1:
        inter_dist_mean = np.mean(inter_dist_means)
        return inter_dist_mean

    intra_dist_means = [np.abs(fi[None, ...] - fi[:, None, ...]).sum() / (fi.shape[0] * (fi.shape[0] - 1)) for fi in
                        fits]
    fit = np.mean(inter_dist_means) - np.mean(intra_dist_means)
    return fit


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


def toroidal_distance(a, b, width):
    dist = a[None, ...] - b[:, None, ...]
    shift = -np.sign(dist) * width
    dist = np.where(np.abs(dist) > width / 2, dist + shift, dist)
    return dist


def update_pos_with_collision(ps, vs, obstacles):
    new_ps = ps + vs
    # get all gridlines with which each line intersects
    xs = np.hstack(ps[:, 0], new_ps[:, 0])
    ys = np.hstack(ps[:, 1], new_ps[:, 1])
    x0s, x1s = np.min(xs, -1), np.max(xs, -1)
    y0s, y1s = np.min(ys, -1), np.max(ys, -1)
    gridlines = [(np.arange(x0, x1), np.arange(y0, y1)) for x0, x1, y0, y1 in zip(x0s, x1s, y0s, y1s)]
    for pi, line in enumerate(gridlines):
        grid_xs, grid_ys = line
        # get all points of intersection with grid
        grid_points = []
        for grid_x in grid_xs:
            a = (grid_x - ps[pi, 0]) / vs[ps, 0]
            grid_y = ps[pi, 0] + a * vs[ps, 1]
            grid_points.append((grid_x, grid_y))

    # For each particle, get the largest delta of the two directions
    max_step = vs[np.arange(vs.shape[0]), np.abs(vs).argmax(axis=-1)]
    # Ceiling of how many cells we will cross through in direction of greatest change
    n_steps = np.ceil(np.abs(max_step))
    step_size = 1 / n_steps
    # FIXME: lazily checking too many inter-cells here
    step_fracs = np.arange(1, n_steps.max() + 1) / n_steps.max()
    print(step_fracs.shape)
    inter_steps = np.tile(vs, len(step_fracs)) * np.repeat(step_fracs, 2)
    inter_cells = np.tile(ps, len(step_fracs)) + inter_steps
    # Taking floors to get grid coordinates
    inter_cells_int = inter_cells.astype(int)
    collisions = obstacles[inter_cells_int[:, 0], inter_cells_int[:, 1]]
    print(collisions.shape)

    return new_ps
