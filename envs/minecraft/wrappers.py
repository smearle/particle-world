import gym
import numpy as np


class MineRLWrapper(gym.Wrapper):
    """An environment wrapper specific to minerl."""
    def __init__(self, env):
        super(MineRLWrapper, self).__init__(env)
        self.env = env
        self.width, self.height, self.depth = self.task.width, self.task.height, self.task.depth
        self.n_chan = len(self.task.block_types)
        self.unique_chans = self.task.unique_chans
        self.max_episode_steps = self.task.max_episode_steps
        self.next_world = None
        self.n_pop = 1
        self.n_policies = 1
        self.n_step = 0

        # TODO: minecraft-specific eval maps
        self.evaluate = False

        # FIXME: kind of a hack. Should support Dict observation space.
        self.observation_space = self.observation_space.spaces['pov']
        # self.unwrapped.observation_space = self.observation_space

    def process_observation(self, obs):
        # FIXME: kind of a hack. Should support Dict observation space.
        return obs['pov']

    def reset(self):
        self.world = self.next_world
        self.next_world = None if not self.enjoy else self.next_world
        if self.world is not None:
            self.task.world_arr = self.world
        obs = super(MineRLWrapper, self).reset()
        obs = self.process_observation(obs)

        return obs

    def step(self, action):
        obs, rew, done, info = super().step(action)
        obs = self.process_observation(obs)
        self.n_step += 1

        return obs, rew, done, info

#   def set_world(self, world):
#       """
#       Set the world (from some external process, e.g. world-generator optimization), and set the env to be reset at
#       the next step.
#       """
#       # This will be set as the current world at the next reset
#       self.next_world = world.reshape(self.width, self.width, self.width)
#       # self.worlds = {idx: worlds[idx]}
#       # print('set worlds ', worlds.keys())

    def set_world(self, world: np.ndarray):
        """
        Convert an encoding produced by the generator into a world map. The encoding has channels (empty, wall, start/goal)
        :param world: Encoding, optimized directly or produced by a world-generator.
        """
        w = np.zeros((self.width, self.height, self.depth), dtype=np.int)

        # TODO: fill this with walls/bedrock or some such, and ensure the agent spawns in the corner
        w.fill(self.task.empty_chan)
        w[1:-1, 1:-1, 1:-1] = world
#       self.goal_idx = np.argwhere(w == self.task.goal_chan)
#       assert len(self.goal_idx) == 1
#       self.goal_idx = self.goal_idx[0]
#       # self.world_flat = w
#       # self.next_world = discrete_to_onehot(w)
        self.next_world = w