import copy
from pdb import set_trace as TT

import numpy as np
from qdpy.phenotype import Features, Fitness
from qdpy.phenotype import Individual as QdpyIndividual

from generators.representations import NCAGenerator


def clone(ind):
    """Clone an individual for the purpose of mutation (or crossover?)."""
    # Create a deep copy of the individual, so that, e.g., the world state is not shared between the two individuals.
    new_ind = copy.deepcopy(ind)

    # Clear the individual's statistics, so that they are recomputed.
    new_ind.reset_stats()

    return new_ind


class Individual(QdpyIndividual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "heuristics": {},

            # TODO: Store player stats here.
            "player_stats": {},
        }


class DiscreteIndividual(Individual):
    """An evolvable individual (i.e. a genotype/phenotype pair), with a 2D discrete array of integers."""
    def __init__(self, width, n_chan, unique_chans=[2, 3], save_gen_sequence=False):
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
            nonunique_chans = set(range(self.n_chan))

            # Get all channel indices that do not correspond to unique tile-types.
            [nonunique_chans.remove(uc) for uc in self.unique_chans]
            nonunique_chans = list(nonunique_chans)

            # Replace surplus unique tiles with non-unique ones.
            self.replace_surplus_unique_tiles(idxs, nonunique_chans)

        for c in self.unique_chans:
            assert np.sum(self.discrete == c) == 1

    def replace_surplus_unique_tiles(self, unique_tile_idxs, nonunique_tile_chans):
        self.discrete[unique_tile_idxs[:, 0], unique_tile_idxs[:, 1]] = np.random.choice(nonunique_tile_chans, len(unique_tile_idxs))

    def mutate(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and np.all(self.discrete == other.discrete))


class TileFlipIndividual2D(DiscreteIndividual):

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


class TileFlipIndividual3D(DiscreteIndividual):

    # TODO: allow for non-cubes, you square.

    def __init__(self, width, n_chan, unique_chans=[2, 3], save_gen_sequence=False):
        super().__init__(width, n_chan, unique_chans, save_gen_sequence)


    def generate(self):
        self.discrete = np.random.randint(0, self.n_chan, size=(self.width, self.width, self.width))

    def mutate(self):
        n_mutate = np.random.randint(1, 5)

        # But I think we can reliably do better with something like this? But then even this seems to get stuck and not
        # identify the optimal zig-zag.
        # n_mutate = int(abs(np.random.normal(1, 5)))

        # Get the flattened indices of tiles to potentially mutate
        xyzs = np.random.randint(0, self.width * self.width * self.width, n_mutate)

        # Un-flatten the mutation indices
        xyzs = np.unravel_index(xyzs, (self.width, self.width, self.width))

        for xyz in xyzs:
            self.discrete[xyz[0], xyz[1], xyz[2]] = np.random.randint(self.n_chan)

        self.validate()

        return self, 

    def replace_surplus_unique_tiles(self, unique_tile_idxs, nonunique_tile_chans):
        self.discrete[unique_tile_idxs[:, 0], unique_tile_idxs[:, 1], unique_tile_idxs[:, 2]] = \
            np.random.choice(nonunique_tile_chans, len(unique_tile_idxs))


# TODO:
# class CPPNIndividual(creator.Individual, CPPN):
#     def __init__(self):
#         CPPN.__init__(self, width)
#         creator.Individual.__init__(self)


class NCAIndividual(DiscreteIndividual):
    def __init__(self, width, n_chan, unique_chans=[2, 3], save_gen_sequence=False):
        Individual.__init__(self, fitness=Fitness((0,), weights=(1,)), features=Features(0,0))
        self.save_gen_sequence = save_gen_sequence
        self.width = width
        self.unique_chans = unique_chans
        self.n_chan = n_chan
        self.nca_generator = NCAGenerator(width, n_chan, 30, save_gen_sequence=save_gen_sequence)
        self.generate()
        self.validate() 

    def generate(self):
        self.gen_sequence = [bd[0] for bd in self.nca_generator.generate()]
        batch_discrete = self.nca_generator.discrete_world.numpy()
        self.discrete = batch_discrete[0]
        self.validate()

    def mutate(self):
        weights = self.nca_generator.get_weights()
        # mut_mask = np.random.choice(weights.shape[0], size=int(min(abs(np.random.normal(1, weights.shape[0] * .02)), weights.shape[0])), replace=False)
        # weights[mut_mask] += np.random.normal(0, 0.1, mut_mask.shape)
        weights += np.random.normal(0, 0.1, weights.shape)
        self.nca_generator.set_weights(weights)
        self.generate()
        return self, 