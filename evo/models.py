from pdb import set_trace as TT

import cv2
import numpy as np
import torch as th
from torch import nn


# indices of weights capturing adjacency in 3x3 kernel (left to right, top to bottom)
adjs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]

adjs_to_acts = {adj: i for i, adj in enumerate(adjs)}

class NCA(nn.Module):
    def __init__(self, n_chan):
        super(NCA, self).__init__()
        self.last_aux = None
        self.n_aux = 16
        self.n_chan = n_chan
        with th.no_grad():
            self.ls1 = nn.Conv2d(n_chan + self.n_aux, 32, 3, 1, 1)  # , padding_mode='circular')
            self.ls2 = nn.Conv2d(32, 64, 1, 1, 0)
            self.ls3 = nn.Conv2d(64, n_chan + self.n_aux, 1, 1, 0)
        self.layers = [self.ls3, self.ls2, self.ls1]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            if self.last_aux is None:
                self.last_aux = th.zeros(x.shape[0], self.n_aux, *x.shape[2:])
            x = th.cat([x, self.last_aux], dim=1)
            x = th.sigmoid(self.ls3(th.relu(self.ls2(th.relu(self.ls1(x))))))
            # x = th.sin(self.ls3(th.sin(self.ls2(th.sin(self.ls1(x))))))
            self.last_aux = x[:, self.n_chan:, ...]


            return x[:, :self.n_chan, ...]

    def reset(self):
        self.last_aux = None


class FullObsPlayNCA(NCA):
    def __init__(self, n_chan, player_chan, obs_width, **kwargs):
        super().__init__(n_chan=n_chan)
        self.player_chan = player_chan
        with th.no_grad():
            # self.fc = nn.Linear(obs_width ** 2 * (self.n_chan + 1), len(adjs))
            self.fc = nn.Linear(900, len(adjs))
        set_nograd(self)
        
    def mutate(self, *args, **kwargs):
        w = get_init_weights(self)
        set_weights(self, w + th.randn_like(w) * 0.1)


    def forward(self, x):
        with th.no_grad():
            n_batches = x.shape[0]
            x = super().forward(x)        
            x = x.view(n_batches, -1)
            x = self.fc(x)

            return x

class PlayNCA(NCA):
    def __init__(self, n_chan, player_chan, **kwargs):
        super().__init__(n_chan=n_chan)
        self.player_chan = player_chan
        with th.no_grad():
            self.neighb_out = nn.Linear(3 * 3, len(adjs))
        set_nograd(self)
        
    def mutate(self, *args, **kwargs):
        w = get_init_weights(self)
        set_weights(self, w + th.randn_like(w) * 0.1)


    def forward(self, x):
        with th.no_grad():
            # The coordinates of the player's position
            player_pos = th.where(x[:, self.player_chan, ...] == 1)

            x = super().forward(x)        

            n_batches = x.shape[0]



            # TODO: sample actions from cells adjacent to players? How to batch this?
            # FIXME: here is the batching issue. x[[0, 1]:[2, 3]] <--- how do we get this? Cannot slice like this!

            # diag_neighb = x[:, 0, player_pos[1] - 1: player_pos[1] + 2, player_pos[2] - 1: player_pos[2] + 2]

            # x = [[0, 1, 2, 3],
            #      [4, 5, 6, 7]]

            # x[[0,1], [0,1]: [2,3]] = [[0, 1],
            #                           [5, 6]]  

            # diag_neighb = th.zeros_like(x[:, 0:1, 0:3, 0:3])
            neighb = th.zeros(x.shape[0], 1, 3, 3)

            # 3x3 boxes around the players
            # boo hoo @ this for loop
            for i, pl_i, in enumerate(player_pos[0]):
                pl_x = x[pl_i]
                neighb[pl_i, 0, :, :] = pl_x[0, player_pos[1][i] - 1: player_pos[1][i] + 2, player_pos[2][i] - 1: player_pos[2][i] + 2]

            adj_cross = th.Tensor([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]])
            neighb *= adj_cross

            # For now we just pass this neighborhood through a little dense layer.
            return self.neighb_out(neighb.view(n_batches, -1))

            # TODO: Could do this more deterministically, selecting max in a given direction, as attempted below.
    #       # neighb += th.finfo(neighb.dtype).min * (-1 * adj_cross + 1)

    #       next_pos = neighb.reshape(n_batches, -1).argmax(dim=1)
    #       next_pos = th.cat(((next_pos % neighb.shape[1]).view(n_batches, -1), (next_pos // neighb.shape[2]).view(n_batches, -1)), dim=1)
    #       next_pos = next_pos.cpu().numpy()

    #       # If no path found, stay put
    #       # next_pos = np.where(next_pos == (0, 0), (1, 1), next_pos)
    #       # next_pos = (1, 1) if next_pos == th.Tensor([0, 0]) else next_pos

    #       # act = [adjs_to_acts[tuple(pos)] for pos in next_pos]
    #       acts = [adjs_to_acts[tuple(pos)] for pos in next_pos]

#       return acts


def init_weights(l):
    if type(l) == th.nn.Conv2d:
        th.nn.init.orthogonal_(l.weight)

#   th.nn.init.normal_(l.weight)


def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn):
    init_params = []
    for name, param in nn.named_parameters():
        init_params.append(param.view(-1))
    init_params = th.hstack(init_params)
    return init_params


def set_weights(nn, weights):
    with th.no_grad():
        n_el = 0
        # for layer in nn.layers:
        for name, param in nn.named_parameters():
            l_param = weights[n_el: n_el + param.numel()]
            n_el += param.numel()
            l_param = l_param.reshape(param.shape)
            param = th.nn.Parameter(th.Tensor(l_param), requires_grad=False)
            # param.weight.requires_grad = False
    return nn



### Hand-weighted path-finding NCA for reference only! ###

RENDER = True

class FloodFill(nn.Module):
    """Source is the goal. Activation flows from goal toward player. When the flow reaches the player, it moves in that
    direction."""
    def __init__(self, empty_chan=0, wall_chan=1, src_chan=2, trg_chan=3):
        super(FloodFill, self).__init__()
        self.src_chan = src_chan
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

