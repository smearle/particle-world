from pdb import set_trace as TT

import cv2
import gym
import minihack
from minihack import LevelGenerator
from minihack.tiles.window import Window
from networkx import grid_graph
import numpy as np

from envs.minihack.adversarial import MiniHackAdversarialEnv

# Set numpy to print full arrays and not wrap rows
np.set_printoptions(threshold=np.inf, linewidth=np.inf)



class MiniHackEvoWorld(MiniHackAdversarialEnv):
    """A wrapper of the MiniHack environment that allows to evolve the environment in a co-learning loop.
    
    Inherits from the env subclass from ucl-dard which allows for adversarial environment generation (i.e. by an RL 
    agent). Could dust off this functionality later when we support RL environment generation!
    """
    empty_chan = 0
    wall_chan = 1
    start_chan = 2
    goal_chan = 3
    unique_chans = (2, 3)
    n_pop = 1
    n_policies = 1
    min_reward = 0
    max_reward = 1
    max_episode_steps = 256

    # MiniHack glyphs (probably best to get these from MiniHack itself somewhere).
    # WALL = 

    def __init__(self, *args, render: bool = True, env_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_env: bool = render
        self.reset()
        self.window = None
            # self.win = cv2.namedWindow("MiniHack")

    def reset(self):
        obs = super().reset_agent()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._last_obs_dict = obs
        return obs, reward, done, info

    def set_world(self, world: np.ndarray):
        """

        We assume the given world is valid/playable (though not necessarily solvable).

        Args:
            world (np.ndarray): _description_
        """
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                if world[i][j] == self.goal_chan:
                    self.goal_pos = (i, j)
                    self.grid.set(i, j, '>')
                elif world[i][j] == self.start_chan:
                    self.agent_start_pos = (i, j)
                    self.grid.set(i, j, '<')
                elif world[i][j] == self.wall_chan:
                    self.grid.set(i, j, '-')
                elif world[i][j] == self.empty_chan:
                    self.grid.set(i, j, '.')
                else:
                    raise Exception

        mapping = [(int(x % (self.width)), int(x // (self.width))) for x in range(self.adversary_action_dim)]
        self.agent_loc = mapping.index(self.agent_start_pos)
        self.goal_loc = mapping.index(self.goal_pos)

        # Finalize level creation
        image = self._get_image_obs()
        self.graph = grid_graph(dim=[self.width, self.height])
        self.wall_locs = [(x, y) for x in range(self.width) for y in range(self.height) if
                            self.grid.get(x, y) in ['-', 'L']]
        for w in self.wall_locs:
            self.graph.remove_node(w)
        try:
            self.compute_shortest_path()
        except:
            print(self.grid.map)

        self.grid.finalize_agent_goal()  # Appends the locations to des file
        self.compile_env() # Makes the environment
        # self.get_map()
        # self.get_grid_obs()
        return
    
    def redraw(self, obs):
        img = obs["pixel"]
        msg = obs["message"]
        msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
        self.window.show_obs(img, msg)

    def render(self):
        if self.window is None:
            self.window = Window("MiniHack")
        assert self.render_env is True
        self.redraw(self._last_obs_dict)
        # im = super().render()
        # cv2.imshow("MiniHack", im)
        # cv2.waitKey(1)
        # return im