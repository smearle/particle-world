import copy
from pdb import set_trace as TT

import gym
import numpy as np
import ray
from ray.rllib import MultiAgentEnv

from minerl.herobraine.env_spec import EnvSpec
from envs.minecraft.touchstone import TouchStone
from utils import discrete_to_onehot


def make_env(env_config):
    # Copying config here because we pop certain settings in env subclasses before passing to parent classes
    env_config = copy.copy(env_config)

    environment_class = env_config.pop('environment_class')

    if issubclass(environment_class, EnvSpec):
        env = gym.make("TouchStone-v0")
        env =  MineRLWrapper(env)

        # DEBUG with built-in minerl environment
        # env = gym.make("MineRLObtainDiamondDense-v0")
        # env.max_episode_steps = 10000  # dummy
        # env.unique_chans = []  # dummy
    else:
        env = environment_class(env_config)

    if issubclass(environment_class, MultiAgentEnv):
        env = WorldEvolutionMultiAgentWrapper(env)
    else:
        env = WorldEvolutionWrapper(env)

    return env
    

def gen_init_world(width, depth, height, block_types):
    # return np.random.randint(0, len(block_types), size=(width, depth, height))
    return np.ones((width, depth, height), dtype=np.int32)


class MineRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MineRLWrapper, self).__init__(env)
        self.env = env
        self.width, self.height, self.depth = self.task.width, self.task.height, self.task.depth
        self.n_chan = len(self.task.block_types)
        self.unique_chans = self.task.unique_chans
        self.max_episode_steps = self.task.max_episode_steps
        self.next_world = None

    def reset(self):
        if self.next_world is not None:
            self.task.world_arr = self.next_world
        obs = super(MineRLWrapper, self).reset()
        return obs

    def set_world(self, world):
        """
        Set the world (from some external process, e.g. world-generator optimization), and set the env to be reset at
        the next step.
        """
        # This will be set as the current world at the next reset
        self.next_world = world.reshape(self.width, self.width, self.width)
        # self.worlds = {idx: worlds[idx]}
        # print('set worlds ', worlds.keys())

    def set_world(self, world):
        """
        Convert an encoding produced by the generator into a world map. The encoding has channels (empty, wall, start/goal)
        :param world: Encoding, optimized directly or produced by a world-generator.
        """
        w = np.zeros((self.width, self.height, self.depth), dtype=np.int)
        w.fill(self.task.empty_chan)
        w[1:-1, 1:-1, 1:-1] = world
        self.goal_idx = np.argwhere(w == self.task.goal_chan)
        assert len(self.goal_idx) == 1
        self.goal_idx = self.goal_idx[0]
        # self.world_flat = w
        # self.next_world = discrete_to_onehot(w)
        self.next_world = w


class WorldEvolutionWrapper(gym.Wrapper):
    """A wrapper facilitating world-evolution in a gym environment, allowing an external process to set the world (i.e.,
    the level layout), and collect statistics of interest (e.g., a player-agent's performance or "regret" on that 
    world)."""
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.need_world_reset = False


    def set_worlds(self, worlds: dict, idx_counter=None, next_n_pop=None, world_gen_sequences=None):

        # Figure out which world to evaluate.
        # self.world_idx = 0
        if idx_counter:
            self.world_key = ray.get(idx_counter.get.remote(hash(self)))
        else:
            self.world_key = np.random.choice(list(worlds.keys()))

        # FIXME: hack
        self.unwrapped.world_key = self.world_key

        # Assign this world to myself.
        self.set_world(worlds[self.world_key])
        self.worlds = self.unwrapped.worlds = worlds
        if next_n_pop is not None:
            self.next_n_pop = next_n_pop
        if world_gen_sequences is not None and len(world_gen_sequences) > 0:
            self.world_gen_sequence = world_gen_sequences[self.world_key]

        self.need_world_reset = self.unwrapped.need_world_reset = True


    def step(self, actions):
        obs, rew, done, info = super().step(actions)

        # TODO: adapt to multi-agent dones?
        done = done or self.need_world_reset

        return obs, rew, done, info


    def reset(self):
        self.need_world_reset = False
        return super().reset()


class WorldEvolutionMultiAgentWrapper(WorldEvolutionWrapper, MultiAgentEnv):
    def __init__(self, env):
        WorldEvolutionWrapper.__init__(self, env)
        self.env = env
        # MultiAgentEnv.__init__(self)