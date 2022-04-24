from enum import unique
from pdb import set_trace as TT

import neat
import numpy as np
import pygame
import torch as th
from neat.genome import DefaultGenome
import pytorch_neat
from pytorch_neat.cppn import create_cppn, Leaf
from torch import nn

from evo.models import NCA, get_init_weights, init_weights, set_weights
from utils import discrete_to_onehot


class Generator(object):
    def __init__(self, width, n_chan):
        # x = np.linspace(0, width - 1, width)
        # y = np.linspace(0, width - 1, width)
        # x, y = np.meshgrid(x, y)
        # xy = (np.stack((x, y), 0)) / width * 2 - 1
        # xy = (np.sin(4 * np.pi * xy) + 1) / 2
        # self.xy = xy
        # self.world = self.xy
        self.n_chan = n_chan
        self.width = width
        pass

    def _reset(self):
        pass

    def get_init_weights(self):
        pass

    def set_weights(self, w):
        print('Attempting to set weights of non-parameterized generator. Ignoring weights.')
        pass

    def render(self, screen):
        return render_landscape(screen, self.landscape)


def render_landscape(screen, landscape):
    screen.fill((255, 255, 255))
    bg = landscape
    bg = bg[..., None]
    bg = (bg - bg.min()) / (bg.max() - bg.min())
    bg = np.concatenate((bg, bg, bg), axis=2)
    bg = pygame.surfarray.make_surface(bg * 255)
    bg = pygame.transform.scale(bg, (screen.get_width(), screen.get_height()))
    screen.blit(bg, (0, 0))
    # pygame.display.update()


class FixedGenerator(Generator):
    def get_init_weights(self):
        return self.landscape.reshape(-1)

    def generate(self, render=False, screen=None, pg_delay=1):
        pass


class TileFlipGenerator2D(FixedGenerator):
    def __init__(self, width, n_chan):
        super().__init__(width, n_chan)
        self.landscape = np.random.random((n_chan, width, width))

    def set_weights(self, w):
        # self.world = th.sigmoid(th.Tensor(w.reshape((width, width)))).numpy()
        self.landscape = w.reshape((self.n_chan, self.width, self.width))


class TileFlipGenerator3D(FixedGenerator):
    def __init__(self, width, n_chan):
        super().__init__(width, n_chan)
        self.landscape = np.random.random((n_chan, width, width, width))

    def set_weights(self, w):
        # self.world = th.sigmoid(th.Tensor(w.reshape((width, width)))).numpy()
        self.landscape = w.reshape((self.n_chan, self.width, self.width, self.width))


class Rastrigin(Generator):
    def __init__(self, width):
        super().__init__(width)
        self.landscape = rastrigin(self.xy)


class Hill(Generator):
    def __init__(self, width):
        super().__init__(width)
        self.landscape = hill(self.xy)


class NNGenerator(Generator):
    def __init__(self, width, n_chan, nn_model):
        super(NNGenerator, self).__init__(width, n_chan)
        self.nn_model = nn_model
        self.world = None

    def _reset(self, latent):
        super()._reset()
        self.nn_model.reset()

        # Could include global information in the form of xy coordinates here, but we won't for now.
        # self.world = th.Tensor(np.concatenate((self.xy, latent), axis=0)).unsqueeze(0)

        self.world = th.Tensor(latent).unsqueeze(0)
        self.landscape = self.world[0, 0].numpy()

    def _update(self, x):
        self.world = self.nn_model(x)
        #           cv2.imshow("NCA world generation", self.world)
        self.landscape = self.world[0, 0].numpy()

        return self.world

    def get_weights(self):
        return get_init_weights(self.nn_model)

    def set_weights(self, weights):
        return set_weights(self.nn_model, weights)


class SinCPPN(nn.Module):
    def __init__(self, n_hid):
        super(SinCPPN, self).__init__()
        self.ls1 = nn.Conv2d(n_hid + 2, 32, 1, 1, 0)
        self.ls2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.ls3 = nn.Conv2d(64, n_hid + 1, 1, 1, 0)
        self.layers = [self.ls3, self.ls2, self.ls1]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = th.sin(self.ls3(th.sin(self.ls2(th.sin(self.ls1(x))))))
            return x


class SinCPPNGenerator(NNGenerator):
    def __init__(self, width):
        self.n_hid = 0
        nca_model = SinCPPN(self.n_hid)
        set_nograd(nca_model)
        super(SinCPPNGenerator, self).__init__(width, nca_model)
        self.world = None
        self._reset()

    def _reset(self):
        latent = np.random.normal(0, 1, size=(self.n_hid, 1, 1))
        latent = np.tile(latent, (1, self.width, self.width))
        super()._reset(latent)

    def _update(self):
        super()._update()
        self.landscape = (self.landscape + 1) / 2  # for sine wave activation only!
        return self.world

    def generate(self, render=False, screen=None, pg_delay=1):
        self._reset()
        self._update()
        if render:
            self.render(screen=screen)
            pygame.time.delay(pg_delay)

    def render(self, screen):
        super().render(screen=screen)


class NCAGenerator(NNGenerator):
    def __init__(self, width, n_chan, n_nca_steps, save_gen_sequence=False):
        self.save_gen_sequence = save_gen_sequence
        self.n_chan = n_chan

        # Ad hoc: constrain the number of spawns/goals to be 1 each. Make this more general?
        self.n_unique_chan = 2

        self.n_nca_steps = n_nca_steps
        nca_model = NCA(n_chan)
        set_nograd(nca_model)
        super(NCAGenerator, self).__init__(width, n_chan, nca_model)
        self.world = None
        self.n_nca_steps = n_nca_steps

        # Fix the latent, making this simply an indirect encoding for now
        # self.latent = np.random.normal(0, 1, size=(self.n_chan, self.width, self.width))
        self.latent = discrete_to_onehot(np.random.randint(0, self.n_chan, size=(self.width, self.width)), self.n_chan)

        self._reset()

    def _reset(self):
        super()._reset(self.latent)
        self.discrete_world = self.world.argmax(1)

    def _update(self, x):
        self.discrete_world = super()._update(x).argmax(1)
        # self.landscape = (self.world[0, 0] + 1) / 2  # for sine wave activation only!

        # Note: we could have auxiliary (non-board-state) tiles here, in which case we won't want to argmax over them 
        # like this.
        # self.discrete_world = self.world.argmax(1)

        return self.world

    def generate(self, render=False, screen=None, pg_delay=1):
        self._reset()
        gen_sequence = [self.discrete_world]
        x = discrete_to_onehot(self.discrete_world[0], self.n_chan).astype(int)
        for _ in range(self.n_nca_steps):
            x = self._update(th.Tensor(x).unsqueeze(0))
            unique_layers = np.zeros((x.shape[0], self.n_unique_chan, x.shape[2], x.shape[3]), dtype=int)
            self.discrete_world = x[:, :self.n_unique_chan].argmax(1)
            for i in range(unique_layers.shape[1]):
                unq_layer = x[:, i]
                unq_coord = np.argmax(unq_layer.reshape(unq_layer.shape[0], -1), axis=1)
                unq_coord = np.unravel_index(unq_coord, unq_layer.shape[1:])
                self.discrete_world[:, unq_coord[0], unq_coord[1]] = self.n_unique_chan + i

                # Discourage unique tiles from overlapping.
                x[:, i+1, unq_coord[0], unq_coord[1]] = -100

            gen_sequence.append(self.discrete_world)
            if self.save_gen_sequence:
                gen_sequence.append(self.discrete_world)
            if render:
                self._render(screen=screen)
                pygame.time.delay(pg_delay)
            x = discrete_to_onehot(self.discrete_world[0], self.n_chan)
        return gen_sequence


class CPPN(Generator):
    def __init__(self, width):
        super().__init__(width)
        n_actions = 1
        neat_config_path = 'config_cppn'
        self.neat_config = neat.config.Config(DefaultGenome, neat.reproduction.DefaultReproduction,
                                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                              neat_config_path)
        self.n_actions = n_actions
        self.neat_config.genome_config.num_outputs = n_actions
        self.neat_config.genome_config.num_hidden = 2
        self.genome = DefaultGenome(0)
        self.genome.configure_new(self.neat_config.genome_config)
        self.input_names = ['x_in', 'y_in']
        self.output_names = ['tile_{}'.format(i) for i in range(n_actions)]
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    def mate(self, ind_1, fit_0, fit_1):
        self.genome.fitness = fit_0
        ind_1.genome.fitness = fit_1
        return self.genome.configure_crossover(self.genome, ind_1.genome, self.neat_config.genome_config)

    def mutate(self):
        #       print(self.input_names, self.neat_config.genome_config.input_keys, self.genome.nodes)
        self.genome.mutate(self.neat_config.genome_config)
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    # def draw_net(self):
    #     draw_net(self.neat_config, self.genome,  view=True, filename='cppn')
    #
    def generate(self, render=False, screen=None, pg_delay=0):
        X = th.arange(self.width)
        Y = th.arange(self.width)
        X, Y = th.meshgrid(X/X.max(), Y/Y.max())
        tile_probs = [self.cppn[i](x_in=X, y_in=Y) for i in range(self.n_actions)]
        multi_hot = th.stack(tile_probs, axis=0)
        multi_hot = multi_hot.unsqueeze(0)
        return multi_hot



def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def rastrigin(ps):
    """
    :param ps: coordinates in the unit square
    """
    scale = 1
    low, high = -5.12, 5.12
    low /= scale
    high /= scale
    ps = ps * high
    ndim = ps.shape[0]
    assert ndim == 2
    X = ps[0, ...]
    Y = ps[1, ...]
    Z = X ** 2 - 10 * np.cos(2 * np.pi * X) + \
        Y ** 2 - 10 * np.cos(2 * np.pi * Y)
    # set mean of zero and std of 1
    Z = (Z - Z.mean()) / Z.std()
    Z = - Z / np.abs(Z).max()
    return Z



def hill(ps):
    """
    :param ps: coordinates in the unit square
    """
    scale = 1
    low, high = -5, 5
    low /= scale
    high /= scale
    ps = ps * high
    ndim = ps.shape[0]
    assert ndim == 2
    X = ps[0, ...]
    Y = ps[1, ...]
    Z = X ** 2 + Y ** 2
    # set mean of zero and std of 1
    Z = (Z - Z.mean()) / Z.std()
    Z = Z / np.abs(Z).max()
    return Z

