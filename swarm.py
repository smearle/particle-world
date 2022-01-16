import numpy as np
import torch as th
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
from torch import nn
from pdb import set_trace as TT


class Swarm(object):
    def __init__(self, n_pop, fov=1, trg_scape_val=1.0):
        assert fov > 0
        self.ps = None
        self.vs = None
        self.fov = fov
        self.n_pop = n_pop
        self.landscape = None
        self.trg_scape_val = trg_scape_val

    def reset(self, scape):
        self.landscape = scape
        self.ps, self.vs = init_ps(self.world_width, self.n_pop)

    def get_observations(self):
        fov = self.fov
        patch_w = fov * 2 + 1
        # Add new dimensions for patch_w-width patches of the environment around each agent
        ps_int = np.floor(self.ps).astype(int)
        # weird edge case, is modulo broken?
        ps_int = np.where(ps_int == self.world_width, 0, ps_int)
        # TODO: this is discretized right now. Maybe it should use eval_fit instead to take advantage of continuity?
        scape = np.pad(self.landscape, fov, mode='wrap')
        # Padding makes this indexing a bit weird but these are patch_w-width neighborhoods
        nbs = [scape[pxi:pxi + 1 + 2 * fov, pyi:pyi + 1 + 2 * fov] for pxi, pyi in zip(ps_int[:, 0], ps_int[:, 1])]
        nbs = np.stack(nbs)
        nbs = nbs[:, None, ...]
        nbs = np.reshape(nbs, (nbs.shape[0], nbs.shape[1], -1))
        return nbs

    def get_rewards(self):
        ps = self.ps.astype(int)
        return np.abs(self.trg_scape_val - self.landscape[ps[:, 0], ps[:, 1]])


def init_ps(world_width, npop, ndim=2):
    # velocities between -1 and 1
    vls = np.zeros((npop, ndim))
    # vls = (np.random.random(size=(npop, ndim)) - 0.5) * 1
    # ps = np.random.random(size=(npop, ndim)) * width
    x = np.linspace(0, 2 * np.pi + 0.1 * np.pi, npop)[:, None]
    ps = (np.hstack((np.sin(x), np.cos(x))))
    ps -= world_width / 2
    # ps = ps / np.sqrt((ps ** 2).sum(axis=-1))[:, None]
    ps = ps * world_width / 4
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
    def __init__(self, fov):
        super().__init__()
        kernel_width = 2 * fov + 1
        self.l1 = nn.Conv2d(1, 32, kernel_width, 1, padding=0)
        self.l2 = nn.Linear(32, 32)
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
        fov = model_config["custom_model_config"]["fov"]
        NN.__init__(self, fov=fov)
        self.tower_stats = {}

    def forward(self, input_dict, state_batches=None, seq_lens=None):
        obs = input_dict['obs']
        ret = super().forward(obs)
        return ret

    def value_function(self) -> TensorType:
        return self.val.squeeze(-1)


class NeuralSwarm(Swarm):
    def __init__(self, world_width, n_pop: int, fov: int = 4, trg_scape_val=1.0):
        super().__init__(n_pop, fov, trg_scape_val=trg_scape_val)
        self.world_width = world_width
        # self.nn = NN(fov=fov)
        self.nn = None

    def update(self, accelerations=None, obstacles=None):
        if accelerations is None:
            if self.nn is None:
                print("Generating random NN swarm policies inside environment, because no accelerations were supplied "
                      "when updating swarms.")
            nbs = self.get_observations(fov=self.fov)
            accelerations = self.nn(th.Tensor(nbs)).detach().numpy()
        if obstacles is not None:
            # TODO: collision detection
            pass
        else:
            self.ps += accelerations - 1
        self.ps = self.ps % self.world_width
        # momentum = 0.5
        # self.vs += accelerations * momentum
        # self.vs /= 1 + 0.1 * momentum


class GreedySwarm(Swarm):
    def __init__(self, world_width, n_pop: int, fov: int = 4, trg_scape_val=1.0):
        super().__init__(n_pop)
        assert fov > 0
        self.fov = fov
        self.world_width = world_width
        self.trg_scape_val = trg_scape_val

    def reset(self, scape):
        super().reset(scape)
        self.ps, self.vs = init_ps(self.world_width, self.n_pop)

    def update(self, obstacles=None):
        if obstacles is not None:
            new_ps = self.ps + self.vs
            # For each particle, get the largest delta of the two directions
            max_step = self.vs[np.arange(self.vs.shape[0]), np.abs(self.vs).argmax(axis=-1)]
            # Ceiling of how many cells we will cross through in direction of greatest change
            n_steps = np.ceil(np.abs(max_step))
            step_size = 1 / n_steps
            # FIXME: lazily checking too many inter-cells here
            step_fracs = np.arange(1, n_steps.max() + 1) / n_steps.max()
            print(step_fracs.shape)
            inter_steps = np.tile(self.vs, len(step_fracs)) * np.repeat(step_fracs, 2)
            inter_cells = np.tile(self.ps, len(step_fracs)) + inter_steps
            # Taking floors to get grid coordinates
            inter_cells_int = inter_cells.astype(int)
            collisions = obstacles[inter_cells_int[:, 0], inter_cells_int[:, 1]]
            print(collisions.shape)
            pass
        else:
            self.ps += self.vs
        self.ps = self.ps % self.world_width
        fov = self.fov
        patch_w = fov * 2 + 1
        # Add new dimensions for patch_w-width patches of the environment around each agent
        ps_int = np.floor(self.ps).astype(int)
        # weird edge case, is modulo broken?
        ps_int = np.where(ps_int == self.world_width, 0, ps_int)
        # TODO: this is discretized right now. Maybe it should use eval_fit instead to take advantage of continuity?
        scape = np.pad(self.landscape, fov, mode='wrap')
        # Padding makes this indexing a bit weird but these are patch_w-width neighborhoods
        nbs = [scape[pxi:pxi + 1 + 2 * fov, pyi:pyi + 1 + 2 * fov] for pxi, pyi in zip(ps_int[:, 0], ps_int[:, 1])]
        nbs = np.stack(nbs)
        nbs[..., 1:-1, 1:-1] = -1  # observe only the periphery of our field of vision
        # nbs += np.random.normal(0, 0.1, (nbs.shape))
        flat_ds = np.abs(nbs.reshape(self.n_pop, -1) - self.trg_scape_val).argmin(axis=-1)
        ds = np.stack((flat_ds // patch_w, flat_ds % patch_w)).swapaxes(1, 0)
        ds = (ds - fov) / fov * 1.0
        momentum = 0.5
        self.vs += ds * momentum
        self.vs /= 1 + 0.1 * momentum


class MemorySwarm(Swarm):
    def __init__(self, n_pop, xplor, xploit):
        super().__init__(n_pop)
        self.r_xplor = xplor
        self.r_xploit = xploit

    def reset(self, scape):
        super().reset(scape)
        self.ps, self.vs = init_ps(self.n_pop)
        self.i_vals = eval_fit(self.ps.swapaxes(1, 0), self.landscape)
        self.i_bests = self.ps.copy()
        g_best_idx = np.argmax(self.i_vals)
        self.g_best = self.i_bests[g_best_idx]
        self.g_val = self.i_vals[g_best_idx]

    def update(self):
        self.ps += self.vs
        self.ps = self.ps % self.world_width
        fits = eval_fit(self.ps.swapaxes(1, 0), self.landscape)
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


def eval_fit(ps, scape):
    ps = np.floor(ps).astype(int)
    scores = scape[ps[0, ...], ps[1, ...]]
    return scores


def contrastive_pop(ps, width):
    # dists = np.sqrt(((ps[0][None, ...] - ps[1][:, None, ...]) ** 2).sum(-1))
    # reward distance between population types
    # tor_dists = np.sqrt(((toroidal_distance(ps[0][None,...], ps[1][:, None, ...]))**2).sum(-1))
    inter_dist_means = [np.sqrt((toroidal_distance(pi, pj, width) ** 2).sum(-1)).mean()
                        for i, pi in enumerate(ps)
                        for j, pj in enumerate(ps) if i != j]
    # penalize intra-pop distance (want clustered populations)
    intra_dist_means = [np.sqrt((toroidal_distance(p, p, width) ** 2).sum(-1)).mean() for p in ps]
    n_intra = len(ps)
    assert len(inter_dist_means) == n_intra * (n_intra - 1)
    return np.mean(inter_dist_means) - np.mean(intra_dist_means)


def toroidal_distance(a, b, width):
    dist = a[None, ...] - b[:, None, ...]
    shift = -np.sign(dist) * width
    dist = np.where(np.abs(dist) > width / 2, dist + shift, dist)
    return dist


