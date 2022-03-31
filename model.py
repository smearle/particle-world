import random
from typing import Dict, List

import cv2
import gym
from gym.spaces import Box
import numpy as np
import pygame
import torch as th
from pdb import set_trace as TT
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelConfigDict, ModelWeights
from torch import TensorType, nn

from envs import ParticleMazeEnv, eval_mazes


th.set_printoptions(profile='full')


class CustomRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=16,
                 fc_size=64,
                 lstm_state_size=256):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape
        self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.conv = nn.Conv2d(obs_space.shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=0)

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.pre_fc_size, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        x = nn.functional.relu(self.conv(input_dict["obs"].permute(0, 3, 1, 2)))
        x = x.reshape(x.size(0), -1)
        return super().forward(input_dict={"obs_flat": x}, state=state, seq_lens=seq_lens)


    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [th.unsqueeze(state[0], 0),
                th.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        return action_out, [th.squeeze(h, 0), th.squeeze(c, 0)]


class CustomConvRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=16,
                 fc_size=64,
                 lstm_state_size=256):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape

        # Size of the activation after conv_3
        self.pre_fc_size = obs_shape[-2] // 8 * obs_shape[-3] // 8 * conv_filters
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.conv_1 = nn.Conv2d(obs_space.shape[-1], out_channels=conv_filters, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(conv_filters, out_channels=conv_filters, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(conv_filters, out_channels=conv_filters, kernel_size=3, stride=2, padding=1)

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.pre_fc_size, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict["obs"].permute(0, 3, 1, 2).float()
        x = nn.functional.relu(self.conv_1(input))
        x = nn.functional.relu(self.conv_2(x))
        x = nn.functional.relu(self.conv_3(x))
        x = x.reshape(x.size(0), -1)

        # This ends up being fed into forward_rnn()
        return super().forward(input_dict={"obs_flat": x}, state=state, seq_lens=seq_lens)


    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [th.unsqueeze(state[0], 0),
                th.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        return action_out, [th.squeeze(h, 0), th.squeeze(c, 0)]



# indices of weights capturing adjacency in 3x3 kernel (left to right, top to bottom)
adjs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]

adjs_to_acts = {adj: i for i, adj in enumerate(adjs)}
RENDER = False

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10,10))

# TODO: Use strided convolutions to compute path length!
class FloodSqueeze(nn.Module):
    def __init__(self, empty_chan=0, wall_chan=1, src_chan=3, trg_chan=2):
        super().__init__()
        self.conv_0 = nn.Conv2d(4, 14, kernel_size=3, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(4, 14, kernel_size=3, stride=2, padding=0)

        w_lb = th.Tensor([
            [ 0,  0,  0],
            [-1, -1,  0],
            [ 0, -1,  0],
        ])
        w_lr = th.Tensor([
            [ 0,  0,  0],
            [-1, -1, -1],
            [ 0,  0,  0],
        ])
        w_lt = th.Tensor([
            [ 0, -1,  0],
            [-1, -1,  0],
            [ 0,  0,  0],
        ])
        w_br = th.Tensor([
            [ 0,  0,  0],
            [ 0, -1, -1],
            [ 0, -1,  0],
        ])
        w_bt = th.Tensor([
            [ 0, -1,  0],
            [ 0, -1,  0],
            [ 0, -1,  0],
        ])
        w_rt = th.Tensor([
            [ 0, -1,  0],
            [ 0, -1, -1],
            [ 0,  0,  0],
        ])

        with th.no_grad():
            sl, sb, sr, st = 0, 1, 2, 3
            tl, tb, tr, tt = 4, 5, 6, 7
            lb, lr, lt, br, bt, rt = 8, 9, 10, 11, 12, 13
            
            w0 = nn.Parameter(th.zeros_like(self.conv_0.weight), requires_grad=False)
            b0 = nn.Parameter(th.zeros_like(self.conv_0.bias), requires_grad=False)

            # Have an activation equal to path-length at channels corresponding to paths between borders of the cross-shape
            for dd_chan, w_dd in zip([lb, lr, lt, br, bt, rt], [w_lb, w_lr, w_lt, w_br, w_bt, w_rt]):
                w0[dd_chan, wall_chan, :, :] = w_dd * 3
                b0[dd_chan] = 3

            for s_chan, t_chan, d in zip([sl, sb, sr, st], [tl, tb, tr, tt], [(1, 0), (2, 1), (1, 2), (0, 1)]):
                w0[s_chan, src_chan, 1, 1] = 1
                w0[s_chan, wall_chan, d[0], d[1]] = -1
                w0[t_chan, trg_chan, 1, 1] = 1
                w0[t_chan, wall_chan, d[0], d[1]] = -1

            self.conv_0.weight = w0
            self.conv_0.bias = b0

            # Detect paths at borders
            w1 = nn.Parameter(th.zeros_like(self.conv_1.weight), requires_grad=False)
            b1 = nn.Parameter(th.zeros_like(self.conv_1.bias), requires_grad=False)

#           w1[l, lb, 1, 0] = 1
#           w1[l, lr, 1, 0] = 1
#           w1[l, lt, 1, 0] = 1
#           w1[b, lb, 2, 1] = 1
#           w1[b, br, 2, 1] = 1
#           w1[b, bt, 2, 1] = 1


    def hid_forward(self, input):
        pass 

    def forward(self, input):
        input = input.permute(0, 3, 1, 2)
        x = self.conv_0(input)
        x = th.clamp(x, 0, x.max())
        n_hid_chans = x.shape[1]
        for i in range(n_hid_chans):
            sub = fig.add_subplot(4, 4, i + 1)
            sub.imshow(x[0, i, :, :].detach().numpy().transpose(1, 0))
        plt.show()
        pass

    def get_solution_length(self, input):
        pass

    def get_dones(self, x, agent_pos):
        pass

    def flood(self, input, x):
        pass


class FloodFill(nn.Module):
    def __init__(self, empty_chan=0, wall_chan=1, src_chan=2, trg_chan=3):
        super(FloodFill, self).__init__()
        self.trg_chan = trg_chan
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
                

    def hid_forward(self, input):
        self.n_batches = n_batches = input.shape[0]
        # agent_pos = (input.shape[2] // 2, input.shape[3] // 2)
        player_pos = th.where(input[:, self.trg_chan, ...] == 1)
        player_pos = (player_pos[1].item(), player_pos[2].item())
        x = self.conv_0(input)
        self.batch_dones = batch_dones = self.get_dones(x, player_pos)
        self.i = i = 0
        while not batch_dones.all() and i < 129:
            x = self.flood(input, x)
            if RENDER:
                im = x[0, self.flood_chan].cpu().numpy()
                im = im.T
                im = im / im.max()
                im = np.expand_dims(np.vstack(im), axis=0)
                im = im.transpose(1, 2, 0)
                im = cv2.resize(im, (550, 550), interpolation=None)
                cv2.imshow("FloodFill", im)
                cv2.waitKey(1)

            self.batch_dones = batch_dones = self.get_dones(x, player_pos)
            self.i = i = i + 1
            diag_neighb = x[:, self.age_chan, player_pos[0] - 1: player_pos[0] + 2, player_pos[1] - 1: player_pos[1] + 2]
        
        return diag_neighb
            

    def forward(self, input):
        n_batches = input.shape[0]
        input = input.permute(0, 3, 1, 2)
        with th.no_grad():
            diag_neighb = self.hid_forward(input)
            neighb = th.zeros_like(diag_neighb)
            for adj in adjs:
                neighb[:, adj[0], adj[1]] = diag_neighb[:, adj[0], adj[1]]
            next_pos = neighb.reshape(n_batches, -1).argmax(dim=1)
            next_pos = th.cat(((next_pos % neighb.shape[1]).view(n_batches, -1), (next_pos // neighb.shape[2]).view(n_batches, -1)), dim=1)
            next_pos = next_pos.cpu().numpy()

            # If no path found, stay put
            # next_pos = np.where(next_pos == (0, 0), (1, 1), next_pos)
            next_pos = (1, 1) if next_pos == th.Tensor([0, 0]) else next_pos

            # act = [adjs_to_acts[tuple(pos)] for pos in next_pos]
            assert next_pos.shape[0] == 1
            act = adjs_to_acts[tuple(next_pos[0])]

        return act

    def get_solution_length(self, input):
        neighb = self.hid_forward(input)
        if not self.batch_dones.all():
            return 0
        return self.i

    def get_dones(self, x, agent_pos):
        batch_dones = x[:, self.age_chan, agent_pos[0], agent_pos[1]] > 0.1
        return batch_dones

    def flood(self, input, x):
        x = self.conv_1(x)
        x[:, self.flood_chan] = th.clamp(x[:, self.flood_chan], 0., 1.)
        x[:, :self.n_in_chans] += input
        return x


class OraclePolicy(Policy):
    """Hand-coded oracle policy based on flood-fill BFS."""

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


class NCA(TorchModelV2, nn.Module):
    """An NCA model comprising a single comprising only convolutional layers, and with the same number of in and out 
    channels. When used as a player, this model assumes that the environment is tailored to it, in that it is given 
    sufficient "planning" steps at the beginning of the episode, during which it can repeatedly make local computations 
    on the board, ultimately coming up with a global plan."""
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, 
                model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space=obs_space, action_space=action_space, num_outputs=num_outputs,
                                             model_config=model_config, name=name)
        nn.Module.__init__(self)

        # The channel that is 1 where the agent is, and 0 everywhere else. The value network will only consider the 
        # neighborhood surrounding the agent, since we expect the shortest path-length to be encoded there (True, but 
        # not on the first step...)
        player_chan = model_config.get('player_chan')
        self.obs_space = obs_space
        self.n_in_chans = n_in_chans = obs_space.shape[-1]
        self.conv_0 = nn.Conv2d(n_in_chans, n_in_chans, 3, 1, padding=1, bias=True)
        self.conv_1 = nn.Conv2d(n_in_chans, n_in_chans, 1, 1, padding=0, bias=True)
        self.conv_2 = nn.Conv2d(n_in_chans, n_in_chans, 1, 1, padding=0, bias=True)
        self.val_dense = nn.Linear(3 * 3, 1)
        self.hid_neighb = None

    def forward(self, input_dict: Dict[str, TensorType]) -> Dict[str, TensorType]:
        x = input_dict['obs']
        x = self.conv_0(x)
        x = th.relu(x)
        x = self.conv_1(x)
        x = th.relu(x)
        x = self.conv_2(x)
        act = th.sigmoid(x)
        x = x.view(x.shape[0], -1)
        val = self.val_dense(x)
        return {'value': val, 'action': act}


class FloodModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, 
                model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space=obs_space, action_space=action_space, num_outputs=num_outputs,
                                             model_config=model_config, name=name)
        nn.Module.__init__(self)
        self.player_chan = model_config['custom_model_config'].pop("player_chan")
        assert self.player_chan is not None
        self.obs_space = obs_space
        self.n_hid_chans = n_hid_chans = 64
        self.n_in_chans = n_in_chans = obs_space.shape[-1]
        self.conv_0 = nn.Conv2d(n_in_chans + n_hid_chans, n_hid_chans, 3, 1, padding=1, bias=True)
        self.conv_1 = nn.Conv2d(n_hid_chans, n_hid_chans, 1, 1, padding=0, bias=True)
        self.conv_2 = nn.Conv2d(n_hid_chans, n_hid_chans, 1, 1, padding=0, bias=True)
        # self.act_dense = nn.Linear(3 * 3 * n_hid_chans, num_outputs)
        self.act_dense = nn.Linear(n_hid_chans, num_outputs)

        # Value network is dense from hidden state (entire map). Could be more clever about this in theory (e.g., use
        # stack of (repeated?) strided convolutions, or only observe the area surrounding the agent).
        # NOTE: assuming a fixed width here.
        self.val_dense = nn.Linear(n_hid_chans * obs_space.shape[0] * obs_space.shape[1], 1)
        self.hid_neighb = None

    def get_initial_state(self):
        return [
            np.zeros((self.n_hid_chans, self.obs_space.shape[0], self.obs_space.shape[1]), dtype=np.float32),
        ]

    def forward(self, input_dict, state, seq_lens):
#       print('state shape', state[0].shape)
#       print('seq_lens', seq_lens)
        x = input_dict["obs"].permute(0, 3, 1, 2)
#       print(f'x shape: {x.shape}')
        # x = x.reshape(x.size(0), -1)
        return super().forward(input_dict={"obs_flat": x}, state=state, seq_lens=seq_lens)

    def forward_rnn(self, input, state, seq_lens):

        # What is this dimension? Time?
#       print(f'input shape: {input.shape}')
        # assert input.shape[1] == 1
        # input = input[:, 0, ...]

        # FIXME: definitely fuxcked. Ignoring some timesteps. (Where tf did these come from though?)
        input = input[:, -1, ...]

        # Only working with the previous state
        assert len(state) == 1

        x = th.cat((input, state[0]), dim=1)
        # n_batches = input.shape[0]
        x = th.relu(self.conv_0(x))
        x = th.relu(self.conv_1(x))
        self.x = x = th.sigmoid(self.conv_2(x))

        player_pos = th.where(input[:, self.player_chan, ...] == 1)

        if player_pos[0].shape == (0,):
            # This must be a dummy batch, so just put the player in the corner (against the wall)
            player_pos = (th.arange(x.shape[0], dtype=int), th.ones(x.shape[0], dtype=int), th.ones(x.shape[0], dtype=int))
        else:
            assert player_pos[0].shape == (x.shape[0],)

        # NOTE: the assumption that we always have walls is key here. Otherwise we could end up with invalid indices
        # hid_neighb = x[player_pos[0], :, player_pos[1] - 1: player_pos[1] + 2, player_pos[2] - 1: player_pos[2] + 2]

        # NOTE: taking a neighborhood is tricky, so we just take the channels at the player's tile and assume the neural
        #  net has embedded the desired movement into them.
        hid_neighb = x[player_pos[0], :, player_pos[1], player_pos[2]]
        act = th.sigmoid(self.act_dense(hid_neighb))

        return act, [x]

    def value_function(self):
        val = self.x.reshape(self.x.shape[0], -1)
        val = self.val_dense(val)
        val = val.reshape(val.shape[0])
        return val




# This is prohibitively slow
class FloodMemoryModel(TorchRNN, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, 
                model_config: ModelConfigDict, name: str):
        TorchRNN.__init__(self, obs_space=obs_space, action_space=action_space, num_outputs=num_outputs,
                                             model_config=model_config, name=name)
        nn.Module.__init__(self)
        self.player_chan = model_config['custom_model_config'].pop("player_chan")
        assert self.player_chan is not None
        self.obs_space = obs_space
        self.n_hid_chans = n_hid_chans = 64
        self.n_in_chans = n_in_chans = obs_space.shape[-1]
        self.conv_0 = nn.Conv2d(n_in_chans + n_hid_chans, n_hid_chans, 3, 1, padding=1, bias=True)
        self.conv_1 = nn.Conv2d(n_hid_chans, n_hid_chans, 1, 1, padding=0, bias=True)
        self.conv_2 = nn.Conv2d(n_hid_chans, n_hid_chans, 1, 1, padding=0, bias=True)
        # self.act_dense = nn.Linear(3 * 3 * n_hid_chans, num_outputs)
        self.act_dense = nn.Linear(n_hid_chans, num_outputs)

        # Value network is dense from hidden state (entire map). Could be more clever about this in theory (e.g., use
        # stack of (repeated?) strided convolutions, or only observe the area surrounding the agent).
        # NOTE: assuming a fixed width here.
        self.val_dense = nn.Linear(n_hid_chans * obs_space.shape[0] * obs_space.shape[1], 1)
        self.hid_neighb = None

    def get_initial_state(self):
        return [
            np.zeros((self.n_hid_chans, self.obs_space.shape[0], self.obs_space.shape[1]), dtype=np.float32),
        ]

    def forward(self, input_dict, state, seq_lens):
#       print('state shape', state[0].shape)
#       print('seq_lens', seq_lens)
        x = input_dict["obs"].permute(0, 3, 1, 2)
#       print(f'x shape: {x.shape}')
        # x = x.reshape(x.size(0), -1)
        return super().forward(input_dict={"obs_flat": x}, state=state, seq_lens=seq_lens)

    def forward_rnn(self, input, state, seq_lens):

        # What is this dimension? Time?
#       print(f'input shape: {input.shape}')
        # assert input.shape[1] == 1
        # input = input[:, 0, ...]

        # Only working with the previous state
        assert len(state) == 1

        acts = []
        last_states = []
        last_state = state[0]

        for t in range(input.shape[1]):
            input_t = input[:, t, ...]
            x = th.cat((input_t, last_state), dim=1)
            x = th.relu(self.conv_0(x))
            x = th.relu(self.conv_1(x))
            x = th.sigmoid(self.conv_2(x))
            last_state = x
            last_states.append(last_state.unsqueeze(1))

            # This is a bit weird. We expect only one 1 activation in each channel activation. We take the max so that 
            # we get some values when there are empty/dummy inputs in the sequence.
            _, player_pos_flat = input_t[:, self.player_chan, ...].view(input_t.shape[0], -1).max(-1)
            player_pos_x, player_pos_y = player_pos_flat // input_t.shape[-1], player_pos_flat % input_t.shape[-1]
            player_pos = (th.arange(input_t.shape[0]), player_pos_x, player_pos_y)

#           if player_pos[0].shape == (0,):
#               # This must be a dummy batch, so just put the player in the corner (against the wall)
#               player_pos = (th.arange(x.shape[0], dtype=int), th.ones(x.shape[0], dtype=int), th.ones(x.shape[0], dtype=int))
#           else:
#               print(f'player_pos shape', player_pos[0].shape)
#               assert player_pos[0].shape == (x.shape[0],)

            # FIXME: missing player observations here, though the same does not occur in the environment observations.
            #  This happens when max_seq_len is > 1, because we are padding with all-0 observations. Fix this by padding
            #  with dummy player-positions as appropriate.
#           if player_pos[0].shape != (x.shape[0],):
#               missing_seq_idxs = set(th.arange(x.shape[0])).difference(player_pos[0])

            # NOTE: the assumption that we always have walls is key here. Otherwise we could end up with invalid indices
            # hid_neighb = x[player_pos[0], :, player_pos[1] - 1: player_pos[1] + 2, player_pos[2] - 1: player_pos[2] + 2]

            # NOTE: taking a neighborhood is tricky, so we just take the channels at the player's tile and assume the neural
            #  net has embedded the desired movement into them.
            hid_neighb = x[player_pos[0], :, player_pos[1], player_pos[2]]
            act = th.sigmoid(self.act_dense(hid_neighb))
            acts.append(act.unsqueeze(1))

        acts = th.cat(acts, dim=1)
        self.last_states = th.cat(last_states, dim=1)
        # n_batches = input.shape[0]


#       print(f'acts shape: {acts.shape}')
        return acts.reshape(-1, self.num_outputs), [self.last_states[:, -1, ...]]

    def value_function(self):
        val = self.last_states.reshape(self.last_states.shape[0] * self.last_states.shape[1], -1)
        val = self.val_dense(val)
        val = val.reshape(val.shape[0])
#       print(f'val shape: {val.shape}')
#       print()
        return val


if __name__ == '__main__':
    n_policies = 1
    n_pop = 1
    width = 15
    n_sim_steps = 128
    pg_width = 600
    model = FloodFill()
    # model = FloodSqueeze()
    # model = FloodMemoryModel()
    env = ParticleMazeEnv(
        {'width': width, 'n_policies': n_policies, 'n_pop': n_pop, 'max_steps': n_sim_steps,
         'pg_width': pg_width, 'evaluate': True, 'objective_function': None, 'num_eval_envs': 1, 
         'fully_observable': True})
    cv2.namedWindow("FloodFill")

    # env.set_worlds(worlds = eval_mazes)
    world_keys = list(eval_mazes.keys()) * 2

    for wk in world_keys:
        env.set_worlds(worlds = {wk: eval_mazes[wk]})
        obs = env.reset()
        env.render()
        done = {(0, 0): False}
        while not done[(0, 0)]:
            obs = th.Tensor(obs[(0, 0)]).unsqueeze(0)
            act = model(obs)
            obs, rew, done, info = env.step({(0, 0): act})
            env.render() 
