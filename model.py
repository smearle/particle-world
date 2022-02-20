import random
from typing import Dict, List

import cv2
import gym
import numpy as np
import pygame
import torch as th
from pdb import set_trace as TT
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelConfigDict, ModelWeights
from torch import Tensor, TensorType, nn

from env import ParticleMazeEnv, eval_mazes


th.set_printoptions(profile='full')

# indices of weights capturing adjacency in 3x3 kernel (left to right, top to bottom)
adjs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]

adjs_to_acts = {adj: i for i, adj in enumerate(adjs)}
RENDER = False

class FloodFill(nn.Module):
    def __init__(self, empty_chan=0, wall_chan=1, src_chan=3, trg_chan=2):
        super(FloodFill, self).__init__()
        self.trg_chan = 4
        self.n_in_chans = n_in_chans = 4
        n_hid_chans = n_in_chans + 2
        self.conv_0 = nn.Conv2d(n_in_chans, n_hid_chans, 1, 1, padding=0, bias=False)
        self.conv_1 = nn.Conv2d(n_hid_chans, n_hid_chans, 3, 1, padding=1, padding_mode='circular', bias=False)
        with th.no_grad():
            # input: (empty, wall, src, trg)
            # weight: (out_chan, in_chan, w, h)

            # this convolution copies the input (empty, wall, src, trg) to the hidden layer...
            self.conv_0.weight = nn.Parameter(th.zeros_like(self.conv_0.weight), requires_grad=False)
            for i in range(n_in_chans):
                self.conv_0.weight[i, i, 0, 0] = 1

            # ... and copies the source to a flood tile
            self.flood_chan = flood_chan = n_in_chans
            self.conv_0.weight[flood_chan, src_chan, 0, 0] = 1

            # this convolution handles the flood
            self.conv_1.weight = nn.Parameter(th.zeros_like(self.conv_1.weight), requires_grad=False)

            # the first n_in_chans channels will hold the actual map (via additive skip connections)

            # the next channel will contain the (binary) flood, with activation flowing from flooded tiles...
            for adj in adjs:
                self.conv_1.weight[flood_chan, flood_chan, adj[0], adj[1]] = 1.

            # ...but stopping at walls.
            self.conv_1.weight[flood_chan, wall_chan, 1, 1] = -6.

            # the next channel will contain the age of the flood
            self.age_chan = age_chan = flood_chan + 1
            self.conv_1.weight[age_chan, flood_chan, 1, 1] = 1.
            self.conv_1.weight[age_chan, age_chan, 1, 1] = 1.
                

    def forward(self, input):
        n_batches = input.shape[0]
        with th.no_grad():
            agent_pos = (input.shape[2] // 2, input.shape[3] // 2)
            x = self.conv_0(input)
            batch_dones = self.get_dones(x, agent_pos)
            i = 0
            while not batch_dones.all() and i < 129:
                x = self.flood(input, x)
                if RENDER:
                    im = x[0, self.flood_chan].cpu().numpy()
                    im = im / im.max()
                    im = np.expand_dims(np.vstack(im), axis=0)
                    im = im.transpose(1, 2, 0)
                    # im = cv2.resize(im, (600, 600), interpolation=None)
                    cv2.imshow("FloodFill", im)
                    cv2.waitKey(1)

                batch_dones = self.get_dones(x, agent_pos)
                i += 1
            diag_neighb = x[:, self.age_chan, agent_pos[0] - 1: agent_pos[0] + 2, agent_pos[1] - 1: agent_pos[1] + 2]
            neighb = th.zeros_like(diag_neighb)
            for adj in adjs:
                neighb[:, adj[0], adj[1]] = diag_neighb[:, adj[0], adj[1]]
            next_pos = neighb.reshape(n_batches, -1).argmax(dim=1)
            next_pos = th.cat(((next_pos % neighb.shape[1]).view(n_batches, -1), (next_pos // neighb.shape[2]).view(n_batches, -1)), dim=1)
            next_pos = next_pos.cpu().numpy()

            # If no path found, stay put
            next_pos = np.where(next_pos == (0, 0), (1, 1), next_pos)

            act = [adjs_to_acts[tuple(pos)] for pos in next_pos]
        return act

    def get_dones(self, x, agent_pos):
        batch_dones = x[:, self.age_chan, agent_pos[0], agent_pos[1]] > 0.1
        return batch_dones

    def flood(self, input, x):
        x = self.conv_1(x)
        x[:, self.flood_chan] = th.clamp(x[:, self.flood_chan], 0., 1.)
        x[:, :self.n_in_chans] += input
        return x



class OraclePolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FloodFill()

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs):
        self.n_batches = obs_batch.shape[0]
        act = self.model(th.Tensor(obs_batch))
        return act, \
               [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(self,
                                actions,
                                obs_batch,
                                state_batches=None,
                                prev_action_batch=None,
                                prev_reward_batch=None):
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass


class FloodModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, 
                model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space=obs_space, action_space=action_space, num_outputs=num_outputs,
                                             model_config=model_config, name=name)
        nn.Module.__init__(self)
        self.n_hid_chans = n_hid_chans = 8
        self.n_in_chans = n_in_chans = obs_space.shape[0]
        self.conv_0 = nn.Conv2d(n_in_chans, n_hid_chans, 1, 1, padding=0, bias=True)
        self.conv_1 = nn.Conv2d(n_hid_chans, n_hid_chans, 3, 1, padding=1, padding_mode='circular', bias=True)
        self.act_dense = nn.Linear(3 * 3, num_outputs)
        self.val_dense = nn.Linear(3 * 3, 1)
        self.hid_neighb = None

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType):
        input = input_dict['obs']
        n_batches = input.shape[0]
        agent_pos = (input.shape[2] // 2, input.shape[3] // 2)
        x = self.conv_0(input)
        # batch_dones = self.get_dones(x, agent_pos)
        # while not batch_dones.all():
        for i in range(100):
            x = self.conv_1(x)
            x[:self.n_hid_chans//2] = th.sigmoid(x[:self.n_hid_chans//2])
            x[self.n_hid_chans//2:] = th.relu(x[self.n_hid_chans//2:])
            x[:, :self.n_in_chans] += input
        self.hid_neighb = neighb = x[:, -1, agent_pos[0] - 1: agent_pos[0] + 2, agent_pos[1] - 1: agent_pos[1] + 2]
        act = self.act_dense(neighb.reshape(n_batches, -1))
        # Return model output and RNN hidden state (not used)
        return act, []

    def value_function(self):
        val = self.val_dense(self.hid_neighb.reshape(self.hid_neighb.shape[0], -1))
        val = val.reshape(val.shape[0])
        return val



if __name__ == '__main__':
    n_policies = 1
    n_pop = 1
    width = 15
    n_sim_steps = 128
    pg_width = 600
    model = FloodFill()
    env = ParticleMazeEnv(
        {'width': width, 'n_policies': n_policies, 'n_pop': n_pop, 'max_steps': n_sim_steps,
         'pg_width': pg_width, 'evaluate': True, 'objective_function': None})
    cv2.namedWindow("FloodFill")


    for i in range(len(eval_mazes)):
        obs = env.reset()
        env.render()
        done = {'__all__': False}
        while not done['__all__']:
            obs = th.Tensor(obs[(0, 0)]).unsqueeze(0)
            act = model(obs)
            obs, rew, done, info = env.step({(0, 0): act})
            env.render()