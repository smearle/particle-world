import gym
import numpy as np
import ray


class MineRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MineRLWrapper, self).__init__(env)
        self.env = env
        self.n_chan = len(self.blocks)

    def reset(self):
        self.world_arr = self.next_world
        return super(MineRLWrapper, self).reset()

    def set_world(self, world):
        """
        Set the world (from some external process, e.g. world-generator optimization), and set the env to be reset at
        the next step.
        """
        # This will be set as the current world at the next reset
        self.next_world = world.reshape(self.width, self.width, self.width)
        self.need_world_reset = True
        # self.worlds = {idx: worlds[idx]}
        # print('set worlds ', worlds.keys())


class WorldEvolutionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env


    def set_worlds(self, worlds: dict, idx_counter=None, next_n_pop=None, world_gen_sequences=None):

        # Figure out which world to evaluate.
        # self.world_idx = 0
        if idx_counter:
            self.world_key = ray.get(idx_counter.get.remote(hash(self)))
        else:
            self.world_key = np.random.choice(list(worlds.keys()))

        # Assign this world to myself.
        self.set_world(worlds[self.world_key])
        self.worlds = worlds
        if next_n_pop is not None:
            self.next_n_pop = next_n_pop
        if world_gen_sequences is not None and len(world_gen_sequences) > 0:
            self.world_gen_sequence = world_gen_sequences[self.world_key]

