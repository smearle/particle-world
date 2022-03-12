import copy
from pdb import set_trace as TT

import gym
import numpy as np
import ray

from minerl.herobraine.env_spec import EnvSpec
from envs.minecraft.touchstone import TouchStone


def make_env(env_config):
    # Copying config here because we pop certain settings in env subclasses before passing to parent classes
    env_config = copy.copy(env_config)

    environment_class = env_config.pop('environment_class')

    if issubclass(environment_class, EnvSpec):
        env = gym.make("TouchStone-v0")
        env =  MineRLWrapper(env)
    else:
        env = environment_class(env_config)

    env = WorldEvolutionWrapper(env)

    return env
    

def gen_init_world(width, depth, height, block_types):
    # return np.random.randint(0, len(block_types), size=(width, depth, height))
    return np.ones((width, depth, height), dtype=np.int32)


class MineRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MineRLWrapper, self).__init__(env)
        self.env = env
        self.width = self.task.width
        self.n_chan = len(self.task.block_types)
        self.unique_chans = self.task.unique_chans
        self.max_episode_steps = self.task.max_episode_steps
        self.next_world = None

    def reset(self):
        if self.next_world is not None:
            self.task.world_arr = self.next_world

        return super(MineRLWrapper, self).reset()

    def step(self, actions):
        obs, rew, done, info = super(MineRLWrapper, self).step(actions)
        done = done or self.need_world_reset

        return obs, rew, done, info

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

