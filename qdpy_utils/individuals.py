from pdb import set_trace as TT

import numpy as np
from qdpy.phenotype import Features, Fitness, Individual

from generator import NCAGenerator


# TODO:
# class CPPNIndividual(creator.Individual, CPPN):
#     def __init__(self):
#         CPPN.__init__(self, width)
#         creator.Individual.__init__(self)


class DiscreteIndividual(Individual):
    def __init__(self, width, n_chan, unique_chans=[2, 3]):
        Individual.__init__(self, fitness=Fitness((0,), weights=(1,)), features=Features(0,0))
        self.width = width
        self.unique_chans = unique_chans
        self.n_chan = n_chan
        self.generate()
        self.validate()

    def validate(self):
        # ensure exactly only one of each of these integers
        for i, u in enumerate(self.unique_chans):
            idxs = np.argwhere(self.discrete == u)
            if len(idxs) == 0:
                xys = range(self.width * self.width)
                occ_xys = [np.where(self.discrete.flatten() == self.unique_chans[ii]) for ii in range(i)]
                if occ_xys:
                    xys = set(xys)
                    occ_xys = np.hstack(occ_xys)
                    [xys.remove(xy[0]) for xy in occ_xys]
                    xys = list(xys)
                xy = np.random.choice(xys)
                x, y = xy // self.width, xy % self.width
                self.discrete[x, y] = u
                continue
            np.random.shuffle(idxs)
            idxs = idxs[:-1]
            new_idxs = set(range(self.n_chan))
            [new_idxs.remove(uc) for uc in self.unique_chans]
            new_idxs = list(new_idxs)
            self.discrete[idxs[:, 0], idxs[:, 1]] = np.random.choice(new_idxs, len(idxs))
        for c in self.unique_chans:
            assert np.sum(self.discrete == c) == 1

    def mutate(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and np.all(self.discrete == other.discrete))


class TileFlipIndividual(DiscreteIndividual):

    def generate(self):
        self.discrete = np.random.randint(0, self.n_chan, size=(self.width, self.width))

    def mutate(self):

        # Keeping this mutation operator for backwards compatibility
        n_mutate = np.random.randint(1, 5)

        # But I think we can reliably do better with something like this? But then even this seems to get stuck and not
        # identify the optimal zig-zag.
        # n_mutate = int(abs(np.random.normal(1, 5)))

        xys = np.random.randint(0, self.width * self.width, n_mutate)
        xys = [(xy // self.width, xy % self.width) for xy in xys]
        for xy in xys:
            self.discrete[xy[0], xy[1]] = np.random.randint(self.n_chan)
        self.validate()
        return self, 


class NCAIndividual(DiscreteIndividual):
    def __init__(self, width, n_chan, unique_chans=[2, 3]):
        Individual.__init__(self, fitness=Fitness((0,), weights=(1,)), features=Features(0,0))
        self.width = width
        self.unique_chans = unique_chans
        self.n_chan = n_chan
        self.nca_generator = NCAGenerator(width, n_chan, 30)
        self.generate()
        self.validate() 

    def generate(self):
        self.nca_generator.generate()
        onehot = self.nca_generator.world.numpy()
        self.discrete = onehot[0].argmax(axis=0)
        self.validate()

    def mutate(self):
        weights = self.nca_generator.get_weights()
        mut_mask = np.random.choice(weights.shape[0], size=int(min(abs(np.random.normal(1, weights.shape[0] * .02)), weights.shape[0])), replace=False)
        weights[mut_mask] += np.random.normal(0, 0.1, mut_mask.shape)
        self.nca_generator.set_weights(weights)
        self.generate()
        return self, 