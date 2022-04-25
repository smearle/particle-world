import math
import sys
from functools import partial
from pdb import set_trace as TT

import gym
import numpy as np
import pygame
import ray
from ray import rllib
from ray.rllib import MultiAgentEnv

# from envs import eval_mazes
from generators.representations import render_landscape
from generators.objectives import max_reward_fitness
from envs.maze.swarm import MazeSwarm, NeuralSwarm, GreedySwarm, contrastive_pop
from utils import discrete_to_onehot

player_colors = [
    (0, 0, 255),
    (255, 0, 0),
    (119, 15, 184),
    (0, 255, 255),
    (255, 0, 255),
    # (0, 255, 0),  # green
    # (255, 255, 0),  # yellow
]
goal_color = (0, 255, 0)
start_color = (255, 255, 0)


class ParticleSwarmEnv(object):
    """An environment in continuous 2D space in which populations of particles can accelerate in certain directions,
    propelling themselves toward desirable regions in the fitness world."""
    def __init__(self, width, swarm_cls, n_policies, n_pop, n_chan=1, pg_width=None, fully_observable=False, 
                 field_of_view=4, rotated_observations=True, translated_observations=True, **kwargs):
        self.rotated_observations = rotated_observations
        self.n_chan = n_chan
        if not pg_width:
            pg_width = width
        self.pg_width=pg_width
        self.world = None
        self.swarms = None
        self.width = width
        self.n_pop = n_pop
        self.n_policies = n_policies
        # self.fields_of_view = [si+1 for si in range(n_policies)]

        if fully_observable:

            # When not translating observation to center the agent, field_of_view is a kind of placeholder, computed 
            # from the middle of the map.
            if not translated_observations:
                self.fields_of_view = [math.floor(width/2) for si in range(n_policies)]

            # If translating, we ensure that the agent can always observe the full map, even when it is in the corner.
            if translated_observations:
                self.fields_of_view = [width - 1 for si in range(n_policies)]

        else:
        # Partially observable map
            self.fields_of_view = [field_of_view for si in range(n_policies)]

        self.patch_widths = [int(field_of_view * 2 + 1) for field_of_view in self.fields_of_view]

        self._gen_swarms(swarm_cls, n_policies, n_pop, self.fields_of_view)

        # NOTE: We assume all swarms have the same reward function/bounds
        self.max_reward = self.swarms[0].max_reward
        self.min_reward = self.swarms[0].min_reward

        self.particle_draw_size = 0.3
        self.n_steps = None
        self.screen = None

    def _gen_swarms(self, swarm_cls, n_policies, n_pop, fields_of_view):
        self.swarms = [
            # GreedySwarm(
            # NeuralSwarm(
            swarm_cls(
                world_width=self.width,
                n_pop=n_pop,
                field_of_view=fields_of_view[si],
                # trg_scape_val=trg)
                trg_scape_val=1.0)
            for si, trg in zip(range(n_policies), np.arange(n_policies) / max(1, (n_policies - 1)))]

    def reset(self):
        assert self.world is not None
        # assert len(self.world.shape) == 2
        [swarm.reset(scape=self.world, n_pop=self.n_pop) for swarm in self.swarms]

    def step_swarms(self):
        [s.update(scape=self.world) for s in self.swarms]

    def render(self, mode='human', pg_delay=0, render_player=True):
        pg_scale = self.pg_width / self.width
        if not self.screen:
            self.screen = pygame.display.set_mode([self.pg_width, self.pg_width])
        render_landscape(self.screen, self.world)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if render_player:
            for pi, policy_i in enumerate(self.swarms):
                for agent_pos in policy_i.ps:
                    agent_pos = agent_pos.astype(int) + 0.5
                    pygame.draw.circle(self.screen, player_colors[pi], agent_pos * pg_scale,
                                    self.particle_draw_size * pg_scale)
        pygame.display.update()
        pygame.time.delay(pg_delay)
        arr = pygame.surfarray.array3d(self.screen)
        # arr = arr.transpose(2, 0, 1)
        # arr = arr / 255
        # return arr
        return True

    # def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=0):
    #     pg_delay = 50
    #     self.reset()
    #     for i in range(n_steps):
    #         self.step_swarms()
    #         if render:
    #             self.screen = screen
    #             self.render(screen, pg_delay)
    #     # p1, p2 = self.swarms[0], self.swarms[1]
    #     # objs = fit_dist([p1, p2], self.world)
    #     ps1, ps2 = self.swarms[0].ps, self.swarms[1].ps
    #     objs = contrastive_pop([swarm.ps for swarm in self.swarms], self.width)
    #     bcs = ps1.mean(0)
    #     return objs, bcs

    def set_landscape(self, landscape):
        assert landscape is not None
        self.world = landscape


class ParticleGym(ParticleSwarmEnv, MultiAgentEnv):

    # TODO: support multiple impassable channels / tile types.
    wall_chan = None
    goal_chan = None

    def __init__(self, width, swarm_cls, n_policies, n_pop, pg_width=500, n_chan=1, fully_observable=False, field_of_view=4,
                 rotated_observations=False, **kwargs):
        ParticleSwarmEnv.__init__(self, 
            width, swarm_cls, n_policies, n_pop, n_chan=n_chan, pg_width=pg_width, fully_observable=fully_observable, field_of_view=field_of_view,
            rotated_observations=rotated_observations, **kwargs)
        MultiAgentEnv.__init__(self)
        
        self.actions = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]

        # Each agent observes 2D patch around itself. Each cell has multiple channels. 3D observation.
        # Map policies to agent observations.
        self.observation_spaces = {i: gym.spaces.Box(-1.0, 1.0, shape=(n_chan, self.patch_widths[i], self.patch_widths[i]))
                                   for i in range(n_policies)}
        # self.observation_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(n_chan, patch_ws[i], patch_ws[i]))
        # self.action_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(2,))
        #                       for i in range(n_policies)}

        # Can move to one of four adjacent tiles
        self.action_spaces = {i: gym.spaces.Discrete(len(self.actions))
                              for i in range(n_policies)}

        self.max_episode_steps = self.get_max_steps()

        if self.evaluate:
            self.max_episode_steps *= 1

        self.n_step = 0
        self.dead_action = [0, 0]

    def set_policies(self, policies, trainer_config):
        # self.swarms = policies
        [swarm.set_nn(policy, i, self.observation_spaces[i], self.action_spaces[i], trainer_config) for i, (swarm, policy) in enumerate(zip(self.swarms, policies))]

    def reset(self):
        # print('reset', self.worlds.keys())
        self.n_step = 0
        # TODO: reset to a world in the archive, via rllib config args?
        super().reset()
        obs = self.get_particle_observations()
        return obs

    def step(self, actions):
        actions = {agent_key: self.actions[act] for agent_key, act in actions.items()}
        assert self.world is not None
        swarm_acts = {i: {} for i in range(len(self.swarms))}
        [swarm_acts[i].update({j: action}) for (i, j), action in actions.items()]
#       batch_swarm_acts = {j: np.vstack([swarm_acts[j][i] for i in range(self.swarms[j].n_pop)])
#                           for j in range(len(self.swarms))}
        batch_swarm_acts = {i: np.vstack([swarm_acts[i][j] if (i, j) in actions else self.dead_action for j in range(self.swarms[i].n_pop)])
                            for i in range(len(self.swarms))}

        # TODO: support multiple impassable channels / tile types
        obstacles = None if not self.wall_chan else self.world[self.wall_chan, ...]

        [swarm.update(scape=self.world, actions=batch_swarm_acts[i], obstacles=obstacles) 
            for i, swarm in enumerate(self.swarms)]
        obs = self.get_particle_observations()
        # Dones before rewards, in case reward is different e.g. at the last step
        self.rew = self.get_reward()
        dones = {(i, j): self.rew[(i, j)] > 0 for i, j in self.rew}
        info = {}
        self.n_step += 1
        assert self.world is not None
#       print(rew)
        return obs, self.rew, dones, info

    def get_particle_observations(self):
        obs = {(i, j): swarm.get_observations(scape=self.world, flatten=False)[j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}

        return obs

    def get_reward(self):
        """Return the per-player rewards for training RL agents."""
        swarm_rewards = [swarm.get_rewards(self.world, self.goal_chan, self.n_step, self.max_episode_steps) for swarm in self.swarms]
        rew = {(i, j): swarm_rewards[i][j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        return rew

    def get_max_steps(self):

        # This is kind of arbitrary
        return self._width **2 // 2


class ParticleGymRLlib(ParticleGym):
    def __init__(self, cfg):
        # This will be a history of statistics for each world that this environment has evaluated. 
        # Of the form: [('world_key', {
        #                               ('policy_id', 'player_id): reward, ...}
        #                          ), ...]
        # TODO: clean up this data structure.
        # self.stats = []

        evaluate = cfg.get("evaluate")
        # self.need_world_reset = False
        self.fully_observable = cfg.get('fully_observable')
        # self.regret_losses = []

        # Global knowledge of number of eval envs for incrementing eval world idx
        self.num_eval_envs = cfg.get('num_eval_envs', None)
        if self.evaluate:
            assert self.num_eval_envs is not None

        super().__init__(**cfg)
        # self.next_n_pop = self.n_pop
        
        # if evaluate:
            # Agents should be able to reach any tile within the initial neighborhood by a shortest path.
            # self.max_steps = max(self.fields_of_view) * 2
            # self.reset = partial(ParticleEvalEnv.reset, self)
            # self.get_reward = partial(ParticleEvalEnv.get_eval_reward, self, self.get_reward)
#       self.world_key = None
#       self.last_world_key = self.world_key
#       self.next_world = None

    def set_world(self, world):
        """
        Set the world (from some external process, e.g. world-generator optimization), and set the env to be reset at
        the next step.
        """
        # This will be set as the current world at the next reset
        self.next_world = world.reshape(self.width, self.width)
        # self.worlds = {idx: worlds[idx]}
        # print('set worlds ', worlds.keys())

    def reset(self):
        self.world = self.next_world
        obs = super().reset()

        return obs


class ParticleMazeEnv(ParticleGymRLlib):
    empty_chan = 0
    wall_chan = 1
    start_chan = 2
    goal_chan = 3
    unique_chans = [start_chan, goal_chan]

    def __init__(self, cfg):
        n_chan = 4
        cfg.update({'n_chan': n_chan})
        cfg['swarm_cls'] = cfg.get('swarm_cls', MazeSwarm)
        width = cfg['width']

        self.evaluate = cfg.get('evaluate')
        self.fully_observable = cfg.get('fully_observable')
        self.translated_observations = cfg.get('translated_observations')
        if self.evaluate:
            self.eval_maze_i = 0
        super().__init__(cfg)

        # Set this broken dummy world only to placate RLlib during dummy reset. Immediately after, we should queue_worlds,
        # which will instigate another reset at the next step.
        dummy_world = np.zeros((width-2, width-2))
        dummy_world[1, 1] = self.start_chan
        dummy_world[2, 2] = self.goal_chan
        self.set_world(dummy_world)

        # TODO: maze-specific evaluation scenarios (will currently break)
        self.n_policies = n_policies = len(self.swarms)
        # self.patch_ws = patch_ws = [int(field_of_view * 2 + 1) for field_of_view in self.fields_of_view]

        # Observe empty, wall, and goal tiles (not start tiles)
        if self.fully_observable and not self.translated_observations:
            # If fully observable and not translating observations to center the player, then we must observe the 
            # player.
            self.n_obs_chan = self.n_chan
            self.player_chan = -1
        else:
            # If not fully observable, or using rotation, then we translate relative to player's position and do not 
            # need to observe the player itself.
            self.n_obs_chan = self.n_chan - 1
            self.player_chan = None

        self.observation_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(self.patch_widths[i], self.patch_widths[i], self.n_obs_chan))
                                   for i in range(n_policies)}
        self.observation_space = None


    def get_particle_observations(self, padding_mode='wrap', surplus_padding=0):

        # Remove the spawn-point channel and replace the spawn point with an empty tile
        obs_scape = np.vstack((self.world[:self.start_chan], self.world[self.start_chan+1:]))
        obs_scape[self.empty_chan, self.start_idx[0], self.start_idx[1]] = 1

        obs = {}

        # If level is fully observable, and we are not rotating, nor translating observations, then we observe the whole map, including
        # an additional channel for the player.
        if self.fully_observable and not self.rotated_observations and not self.translated_observations:
            swarms_obs = []
            for i, swarm in enumerate(self.swarms):
                obs_i = swarm.get_full_observations(
                    scape=obs_scape,
                    flatten=False,
                    )
                swarms_obs.append(obs_i)

        # Otherwise, we get a local observation, centered on the player, of the surrounding area. Note that if the level
        # is fully observable, with rotated observations, we basically take a very large, padded, local observation, and
        # rotate it.
        else:
            swarms_obs = []
            for i, swarm in enumerate(self.swarms):
                obs_i = swarm.get_observations(
                    scape=obs_scape,
                    flatten=False,
                    padding_mode=padding_mode, 
                    surplus_padding=surplus_padding,
                    )
                swarms_obs.append(obs_i)

        # TODO: we could also have full observation and rotation without translation.
        # e.g.: with 1 as player and 2 as wall...
        # [0 0 2]  --player turns right-->  [2 2 2] 
        # [0 0 2]                           [0 0 0]
        # [1 0 2]                           [0 0 1]

        # Collect the observations of each swarm, flatten them into a dictionary with (swarm_is, player_id) tuples as
        # keys.
        for i, (obs_i, swarm) in enumerate(zip(swarms_obs, self.swarms)):
            for j in range(swarm.n_pop):
                obs[(i, j)] = obs_i[j]

        return obs


    def set_world(self, world):
        """
        Convert an encoding produced by the generator into a world map. The encoding has channels (empty, wall, start/goal)

        :param world: Encoding, optimized directly or produced by a world-generator.
        """
        next_world = np.zeros((self.width, self.width), dtype=np.int)
        next_world.fill(1)
        next_world[1:-1, 1:-1] = world
        self.start_idx = np.argwhere(next_world == 2)
        self.goal_idx = np.argwhere(next_world == 3)

        # Double-check we only have one spawn/goal point.
        assert len(self.start_idx) == 1
        assert len(self.goal_idx) == 1

        # NOTE: We set the positions of the spawn/goal immediately, but only "queue up" the world to be properly 
        #   loaded at the next reset. So we need to ensure that we reset immediately after this.
        # FIXME: Queue these up in a similar way. Potential buggy goal-reaching in our final step before reset?
        # Store the spawn/goal coordinates.
        self.start_idx = self.start_idx[0]
        self.goal_idx = self.goal_idx[0]
        self.world_flat = next_world

        # Queue up the next world, ready for loading.
        self.next_world = discrete_to_onehot(next_world, self.n_chan)

    def step(self, actions):

        obs, rew, done, info = super().step(actions)
        # print(f"step {self.n_step} world {self.world_key}, done: {done}, max steps {self.max_steps}")

#       assert np.all([agent_key not in self.dead for agent_key in actions]), "Received an action for a dead agent."

        [obs.pop(k) for k in self.dead if k in obs]
        [rew.pop(k) for k in self.dead if k in rew]
        [done.pop(k) for k in self.dead if k in done]
        [info.pop(k) for k in self.dead if k in info]

        self.dead.update(k for k in obs if done[k])

        return obs, rew, done, info

    def step_swarms(self):
        [s.update(scape=self.world, obstacles=self.world) for s in self.swarms]

    def reset(self):
        # print(f'reset world {self.world_key} on step {self.n_step}')
        # These are the agents who have won the game, and will not be taking further observations/actions.
        self.dead = set({})

        # FIXME: redundant observations are being taken here

        obs = super().reset()

        for swarm in self.swarms:
            swarm.ps[:] = self.start_idx

        obs = self.get_particle_observations()
        # print(f"MazeEnv observation shape: {obs[(0,0)].shape}")

        return obs

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=0):
        generator.world = self.world  # just for rendering
        return super().simulate(n_steps=n_steps, generator=generator, render=render, screen=screen, pg_scale=pg_scale, pg_delay=pg_delay)

    def get_max_steps(self):
        width = self.width

        # Longest path between spawn and goal consists of the optimal zig-zag + hanging tile if width is odd
        max_path = width * width // 2 + width % 2

        # If agent does not receive full observations, it might have to explore dead-ends. Give it twice as much time 
        # (though it may in fact need much more)
        return max_path * 2

    def render(self, mode='human', pg_delay=0, pg_width=None, render_player=True, screen=None, enforce_constraints=True):
        if not pg_width:
            pg_width = self.pg_width if mode == 'human' else self.width
        pg_scale = pg_width / self.width
        if not self.screen:
            self.screen = screen if screen else pygame.display.set_mode([pg_width, pg_width])
#       if self.n_step == 1 and self.world_gen_sequence is not None:
#           pass
#           for i in range(len(self.world_gen_sequence)):
#               world = self.world_gen_sequence[i]
#               w = np.zeros((self.width, self.width), dtype=np.int)
#               w.fill(1)
#               w[1:-1, 1:-1] = world
#               start_idx = np.argwhere(w == 2)
#               goal_idx = np.argwhere(w == 3)
#               # assert len(self.start_idx) == 1
#               # assert len(self.goal_idx) == 1
#               start_idx = start_idx[0]
#               goal_idx = goal_idx[0]
#               world_flat = w
#               onehot_world = discrete_to_onehot(w)
#               self._render_level(onehot_world, start_idx, goal_idx, pg_scale, pg_delay, mode)
#       else:
        return self.render_level(self.world, [self.start_idx], [self.goal_idx], pg_scale, pg_delay, mode, 
                           render_player=render_player, enforce_constraints=enforce_constraints)

    def render_level(self, world, start_idx, goal_idx, pg_scale, pg_delay, mode, render_player, enforce_constraints):
        """_summary_

        Args:
            world (np.ndarray): A onehot-encoded representation of the world. Shape: (n_tile_types, width, width).
            start_idx (np.ndarray): _description_
            goal_idx (np.ndarray): A list of (x, y) coordinates corresponding to goal tile positions. Shape (n_goals, 2).
            pg_scale (_type_): _description_
            pg_delay (_type_): _description_
            mode (_type_): _description_
            render_player (_type_): _description_
            enforce_constraints (_type_): _description_

        Returns:
            _type_: _description_
        """
        render_landscape(self.screen, -1 * world[1] + 1)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if enforce_constraints:
            assert len(goal_idx) == len(start_idx) == 1
        for gidx in goal_idx:
            pygame.draw.rect(self.screen, goal_color, (gidx[0] * pg_scale, gidx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        for sidx in start_idx:
            pygame.draw.rect(self.screen, start_color, (sidx[0] * pg_scale, sidx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        if render_player:
            for pi, policy_i in enumerate(self.swarms):
                for agent_pos in policy_i.ps:
                    agent_pos = agent_pos.astype(int) + 0.5
                    pygame.draw.circle(self.screen, player_colors[pi], agent_pos * pg_scale,
                                    self.particle_draw_size * pg_scale)
        if mode == 'human':
            pygame.display.update()
            pygame.time.delay(pg_delay)
            return True

        arr = pygame.surfarray.array3d(self.screen)
        if mode == 'rgb':
            return arr


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


# triangles facing left, down, right, up
triangle = ((-5, -5), (-2, 0), (-5, 5), (5, 0))
directed_triangles = [[rotate((0, 0), triangle[j], math.pi / 2 * i) for j in range(len(triangle))] for i in range(4)]
triangle_scale = 0.08


class DirectedMazeEnv(ParticleMazeEnv):
    """A maze environment in which agents are egocentric or "rotated".
    
    Agents can turn left and right, and move forward and backward. Their observation is a one-hot encoding of the 
    neighborhood that surrounds them, rotated to that they are "looking forward."
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dead_action = 3
        self.actions = [0, 1, 2, 3]
#       if self.fully_observable:
#           n_chan = self.n_chan  # visible player
#       else:
#           n_chan = self.n_chan - 1

# TODO: support distinct Box & Discrete observations, to be processed at separate points in the neural architecture?
#  rllib/models/preprocessors.py seems to not consider this >:(

#       self.observation_spaces = {i: gym.spaces.Dict({
#           'map': gym.spaces.Box(0.0, 1.0, shape=(self.patch_ws[i], self.patch_ws[i], n_chan)),
#           'direction': gym.spaces.Discrete(4)})
#                                  for i in range(self.n_policies)}

#       self.observation_spaces = {i: gym.spaces.Tuple((
#           gym.spaces.Box(0.0, 1.0, shape=(self.patch_ws[i], self.patch_ws[i], n_chan)),
#           gym.spaces.Discrete(4)))
#                                  for i in range(self.n_policies)}

        self.observation_spaces = {i: 
            gym.spaces.Box(0.0, 1.0, shape=(self.patch_widths[i], self.patch_widths[i], self.n_obs_chan + 4))
                                   for i in range(self.n_policies)}
        self.action_spaces = {i : gym.spaces.Discrete(4) for i in range(self.n_policies)}

    def render(self, mode='human', pg_delay=0, pg_width=None, render_player=True, enforce_constraints=True):
        """
        Args:
            enforce_constraints: Whether to check for (1 goal, 1 spawn) constraints. Can set to false if rendering 
                world-gen to fully show the generation process.
        """
        if not pg_width:
            pg_width = self.pg_width
        pg_scale = pg_width / self.width
        if not self.screen:
            self.screen = pygame.display.set_mode([pg_width, pg_width])
        render_landscape(self.screen, -1 * self.world[1] + 1)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.draw.rect(self.screen, goal_color, (self.goal_idx[0] * pg_scale, self.goal_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        pygame.draw.rect(self.screen, start_color, (self.start_idx[0] * pg_scale, self.start_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        if render_player:
            for pi, policy_i in enumerate(self.swarms):
                ag_i = 0
                for agent_pos, agent_direction in zip(policy_i.ps, policy_i.directions):
                    agent_pos = agent_pos.astype(int) + 0.5
                    points = [np.array(point) * triangle_scale * pg_scale + agent_pos * pg_scale for point in directed_triangles[agent_direction]]
                    pygame.draw.polygon(self.screen, player_colors[pi], points)
                    ag_i = (ag_i + 1) % 4
        if mode == 'human':
            pygame.display.update()
            pygame.time.delay(pg_delay)
            return True

        arr = pygame.surfarray.array3d(self.screen)
        if mode == 'rgb':
            return arr

    def _render_level(self, world, start_idx, goal_idx, pg_scale, pg_delay, mode, enforce_constraints):
        render_landscape(self.screen, -1 * world[1] + 1)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if enforce_constraints:
            assert len(goal_idx) == len(start_idx) == 1
        for gidx in goal_idx:
            pygame.draw.rect(self.screen, goal_color, (gidx * pg_scale, gidx * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        for sidx in start_idx:
            pygame.draw.rect(self.screen, start_color, (sidx * pg_scale, sidx * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        for pi, policy_i in enumerate(self.swarms):
            for agent_pos, agent_direction in policy_i.ps, policy_i.directions:
                agent_pos = agent_pos.astype(int) + 0.5
#               pygame.draw.circle(self.screen, player_colors[pi], agent_pos * pg_scale,
#                                  self.particle_draw_size * pg_scale)
                pygame.draw.polygon(self.screen, player_colors[pi], directed_triangles[agent_direction])
        if mode == 'human':
            pygame.display.update()
            pygame.time.delay(pg_delay)
            return True

        arr = pygame.surfarray.array3d(self.screen)
        if mode == 'rgb':
            return arr

    def get_particle_observations(self):
        obs = super().get_particle_observations(padding_mode='constant', surplus_padding=self.fields_of_view[0])
        return obs

    def get_max_steps(self):
        max_steps = super().get_max_steps()

        # Very imprecisely suppose the agent might have to turn at every other step along its route
        return int(max_steps * 1.5)


# TODO: ...
class MazeEnvForNCAgents(ParticleMazeEnv):
    def __init__(self, *args, **kwargs):
        ParticleMazeEnv.__init__(self, *args, **kwargs)
        # steps where the agent plans, and does not act
        self.n_plan_step = 0
        self.max_plan_steps = self.max_episode_steps // 2
        # auxiliary channels where the agent can plan its moves across the map. In theory we need only 2 to solve a 
        # maze: one for flooding a path, another for counting the age of the path (so that a shortest path can be
        # reconstructed).
        n_aux_chans = 16
        self.aux_maps = np.zeros((n_aux_chans, self.width, self.width))
        self.observation_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(self.n_chan + 1 + n_aux_chans, 
                                   self.patch_widths[i], self.patch_widths[i])) for i in range(self.n_policies)}
        self.action_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(self.n_chan + 1 + n_aux_chans,
                                   self.patch_widths[i], self.patch_widths[i])) for i in range(self.n_policies)}

    def get_max_steps(width=15):
        # max. path length times 2: first the agent computes, then traverses it
        return super().get_max_steps(width=width) * 2

    def reset(self):
        obs_map = super().reset()
        self.n_plan_step = 0
        return obs_map
    
    def step(self, actions):
        raise NotImplementedError
        if self.n_plan_step < self.max_plan_steps:
            self.aux_map += actions
        else:
            actions = actions[self.ps]


## This is a dummy class not currently used, except by its parent ParticleGymRLlib, which borrows its methods when instantiating,
## if in evaluation mode.
#class ParticleEvalEnv(ParticleGymRLlib):
#    """
#    An environment that assumes that a feed-forward (i.e. memoryless) neural network is "best" at the task of navigating
#    a continuous fitness world when it simply greedly moves to the best tile in its field of vision. Randomly generate
#    a map consisting of a neighborhood, and reward 1 when the policy moves to the tile with greatest value, otherwise 0.
#    Episodes last one step.
#    """
#    def __init__(self, **cfg):
#        """
#        :param fields_of_view: The fields of vision of the policies (how far they can see in each direction.
#        """
#        super().__init__(**cfg)
#        raise Exception(f"{type(self)} is a dummy class.")
#
#    def reset(self):
#        """Generate uniform random fitness world, then set to 0 all tiles not in the initial field of vision of any agent."""
#        self.fitnesses = {}
#        og_scape = np.random.random((self.width, self.width))
#        # Note that we're calling set_world on ourselves. Normally this is called externally before reset
#        self.set_world_eval(og_scape, hash(self))
#        obs = super(ParticleGymRLlib, self).reset()
#        self.agent_ids = [agent_id for agent_id in obs]
#        # Note that this is weird, allows borrowing by parent class. Will break the
#        self.init_nbs = [swarm.get_observations(og_scape, flatten=False) for swarm in self.swarms]
#        landscape = np.ones(og_scape.shape)
#        for swarm, init_nb, field_of_view in zip(self.swarms, self.init_nbs, self.fields_of_view):
#            for ps, nb in zip(swarm.ps.astype(int), init_nb):
#                # TODO: vectorize this
#                landscape[ps[0] - field_of_view: ps[0] + field_of_view + 1, ps[1] - field_of_view: ps[1] + field_of_view + 1] = nb
#        self.set_landscape(landscape)
#        # obs = [np.reshape(nb, (nb.shape[0], nb.shape[1], -1)) for nb in self.init_nbs]
#        return obs
#
#    def get_eval_reward(self, og_get_reward):
#        """Reward for policies when their agents move the best tile that was in their initial field of vision."""
#        if not self.dones['__all__']:
#            return {agent_id: 0 for agent_id in self.agent_ids}
#        fields_of_view = [int((nb[0].shape[0] - 1) / 2) for nb in self.init_nbs]
#        # nbs = [nb[field_of_view - 1: field_of_view + 2] for nb, field_of_view in zip(self.init_nbs, fields_of_view)]
#        nbs = self.init_nbs
#        # Condition is satisfied when
#        og_rewards = og_get_reward()
#        rewards = {agent_id: int(np.max(nb) == og_rewards[agent_id]) for agent_id, nb in zip(self.agent_ids, self.init_nbs)}
#        return rewards