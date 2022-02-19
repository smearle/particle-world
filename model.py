import cv2
import numpy as np
import pygame
import torch as th
th.set_printoptions(profile='full')
from torch import nn

from env import ParticleMazeEnv, eval_mazes

# indices of weights capturing adjacency in 3x3 kernel (left to right, top to bottom)
adjs = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]

adjs_to_acts = {adj: i for i, adj in enumerate(adjs)}


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

            # this convolution copies the input
            self.conv_0.weight = nn.Parameter(th.zeros_like(self.conv_0.weight), requires_grad=False)
            for i in range(n_in_chans):
                self.conv_0.weight[i, i, 0, 0] = 1

            # this convolution handles the flood
            self.conv_1.weight = nn.Parameter(th.zeros_like(self.conv_1.weight), requires_grad=False)

            # the first n_in_chans channels will hold the actual map (via additive skip connections)

            # the next channel will contain the (binary) flood, with activation flowing from source...
            self.flood_chan = flood_chan = n_in_chans
            for adj in adjs:
                self.conv_1.weight[flood_chan, src_chan, adj[0], adj[1]] = 1.

            # ...and flooded tiles...
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
            batch_dones = x[:, self.flood_chan, agent_pos[0], agent_pos[1]] > 0.1
            while not batch_dones.all():
                x = self.flood(input, x)
                if RENDER:
                    im = x[0, self.flood_chan].cpu().numpy()
                    im = im / im.max()
                    im = np.expand_dims(np.vstack(im), axis=0)
                    im = im.transpose(1, 2, 0)
                    # im = cv2.resize(im, (600, 600), interpolation=None)
                    cv2.imshow("FloodFill", im)
                    cv2.waitKey(1)
                

                batch_dones = x[:, self.flood_chan, agent_pos[0], agent_pos[1]] > 0.1
            diag_neighb = x[:, self.age_chan, agent_pos[0] - 1: agent_pos[0] + 2, agent_pos[1] - 1: agent_pos[1] + 2]
            neighb = th.zeros_like(diag_neighb)
            for adj in adjs:
                neighb[:, adj[0], adj[1]] = diag_neighb[:, adj[0], adj[1]]
            next_pos = neighb.reshape(n_batches, -1).argmax()
            next_pos = th.cat(((next_pos // neighb.shape[1]).view(-1), (next_pos % neighb.shape[2]).view(-1)), dim=0)
        return adjs_to_acts[tuple(next_pos.cpu().numpy())]

    def flood(self, input, x):
        x = self.conv_1(x)
        x[:, self.flood_chan] = th.clamp(x[:, self.flood_chan], 0., 1.)
        x[:, :self.n_in_chans] += input
        return x



if __name__ == '__main__':
    RENDER = True
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
        done = {(0, 0): False}
        while not done[(0, 0)]:
            obs = th.Tensor(obs[(0, 0)]).unsqueeze(0)
            act = model(obs)
            obs, rew, done, info = env.step({(0, 0): act})
            env.render()