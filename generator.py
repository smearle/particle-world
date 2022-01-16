from pdb import set_trace as TT

import numpy as np
import torch as th
from torch import nn
import pygame


class Generator(object):
    def __init__(self, width):
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, width - 1, width)
        x, y = np.meshgrid(x, y)
        xy = (np.stack((x, y), 0)) / width * 2 - 1
        # xy = (np.sin(4 * np.pi * xy) + 1) / 2
        self.xy = xy
        self.landscape = self.xy
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
    pygame.display.update()


class FixedGenerator(Generator):
    def get_init_weights(self):
        return self.landscape.reshape(-1)

    def generate(self, render=False, screen=None, pg_delay=1):
        pass


class TileFlipFixedGenerator(FixedGenerator):
    def __init__(self, width):
        super().__init__(width)
        self.landscape = np.random.random((self.width, self.width))

    def set_weights(self, w):
        # self.landscape = th.sigmoid(th.Tensor(w.reshape((width, width)))).numpy()
        self.landscape = w.reshape((self.width, self.width))


class Rastrigin(Generator):
    def __init__(self, width):
        super().__init__(width)
        self.landscape = rastrigin(self.xy)


class Hill(Generator):
    def __init__(self, width):
        super().__init__(width)
        self.landscape = hill(self.xy)


class NNGenerator(Generator):
    def __init__(self, width, nn_model):
        super(NNGenerator, self).__init__(width)
        self.nn_model = nn_model
        self.world = None

    def _reset(self, latent):
        super()._reset()
        self.world = th.Tensor(np.concatenate((self.xy, latent), axis=0)).unsqueeze(0)
        self.landscape = self.world[0, 0].numpy()

    def _update(self):
        self.world = self.nn_model(self.world)
        #           cv2.imshow("NCA landscape generation", self.landscape)
        self.landscape = self.world[0, 0].numpy()

    def get_init_weights(self):
        return get_init_weights(self.nn_model)

    def set_weights(self, weights):
        return set_weights(self.nn_model, weights)


class NCA(nn.Module):
    def __init__(self, n_hid):
        super(NCA, self).__init__()
        self.ls1 = nn.Conv2d(n_hid + 2, 32, 3, 1, 1, padding_mode='circular')
        self.ls2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.ls3 = nn.Conv2d(64, n_hid + 2, 1, 1, 0)
        self.layers = [self.ls3, self.ls2, self.ls1]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = th.sigmoid(self.ls3(th.relu(self.ls2(th.relu(self.ls1(x))))))
            # x = th.sin(self.ls3(th.sin(self.ls2(th.sin(self.ls1(x))))))
            return x


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
    def __init__(self, width, n_nca_steps):
        self.n_nca_steps = n_nca_steps
        self.n_hid = 3
        nca_model = NCA(self.n_hid)
        set_nograd(nca_model)
        super(NCAGenerator, self).__init__(width, nca_model)
        self.world = None
        self.n_nca_steps = n_nca_steps
        self._reset()

    def _reset(self):
        latent = np.random.normal(0, 1, size=(self.n_hid, self.width, self.width))
        super()._reset(latent)

    def _update(self):
        super()._update()
        self.landscape = (self.world[0, 0] + 1) / 2  # for sine wave activation only!
        return self.world

    def generate(self, render=False, screen=None, pg_delay=1):
        self._reset()
        for _ in range(self.n_nca_steps):
            self._update()
            if render:
                self._render(screen=screen)
                pygame.time.delay(pg_delay)


def set_weights(nn, weights):
    with th.no_grad():
        n_el = 0
        for layer in nn.layers:
            l_weights = weights[n_el: n_el + layer.weight.numel()]
            n_el += layer.weight.numel()
            l_weights = l_weights.reshape(layer.weight.shape)
            layer.weight = th.nn.Parameter(th.Tensor(l_weights))
            layer.weight.requires_grad = False
            if layer.bias is not None:
                n_bias = layer.bias.numel()
                b_weights = weights[n_el: n_el + n_bias]
                n_el += n_bias
                b_weights = b_weights.reshape(layer.bias.shape)
                layer.bias = th.nn.Parameter(th.Tensor(b_weights))
                layer.bias.requires_grad = False
    return nn


def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def init_weights(l):
    if type(l) == th.nn.Conv2d:
        th.nn.init.orthogonal_(l.weight)
#   th.nn.init.normal_(l.weight)


def get_init_weights(nn):
    init_params = []
    for name, param in nn.named_parameters():
        init_params.append(param.view(-1).numpy())
    init_params = np.hstack(init_params)
    return init_params


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
    Z = Z / np.abs(Z).max()
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
    return (Z - Z.mean()) / Z.std()

