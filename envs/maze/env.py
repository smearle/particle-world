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
from swarm import MazeSwarm, NeuralSwarm, GreedySwarm, contrastive_pop, contrastive_fitness, min_solvable_fitness
from utils import discrete_to_onehot

player_colors = [
    (0, 0, 255),
    (255, 0, 0),
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
    def __init__(self, width, swarm_cls, n_policies, n_pop, n_chan=1, pg_width=None, fully_observable=False, fov=4,
                 rotated_observations=False):
        self.rotated_observations=rotated_observations
        self.n_chan = n_chan
        if not pg_width:
            pg_width = width
        self.pg_width=pg_width
        self.world = None
        self.swarms = None
        self.width = width
        self.n_pop = n_pop
        # self.fovs = [si+1 for si in range(n_policies)]

        if fully_observable:
            # if not rotated_observations:

            # just viewing the whole map, not centered at agent
            self.fovs = [math.floor(width/2) for si in range(n_policies)]

            # TODO: add a `translation` arg to toggle this (automatically True if not fully_observable)
#           else:
#               # viewing the whole map, centered at agent, guaranteed via padding
#               self.fovs = [math.floor(width) for si in range(n_policies)]

        else:
        # Partially observable map
            self.fovs = [fov for si in range(n_policies)]

        self.patch_ws = [int(fov * 2 + 1) for fov in self.fovs]

        self._gen_swarms(swarm_cls, n_policies, n_pop, self.fovs)
        self.particle_draw_size = 0.3
        self.n_steps = None
        self.screen = None

    def _gen_swarms(self, swarm_cls, n_policies, n_pop, fovs):
        self.swarms = [
            # GreedySwarm(
            # NeuralSwarm(
            swarm_cls(
                world_width=self.width,
                n_pop=n_pop,
                fov=fovs[si],
                # trg_scape_val=trg)
                trg_scape_val=1.0)
            for si, trg in zip(range(n_policies), np.arange(n_policies) / max(1, (n_policies - 1)))]

    def reset(self):
        assert self.world is not None
        # assert len(self.world.shape) == 2
        [swarm.reset(scape=self.world, n_pop=self.n_pop) for swarm in self.swarms]

    def step_swarms(self):
        [s.update(scape=self.world) for s in self.swarms]

    def render(self, mode='human', pg_delay=0):
        pg_scale = self.pg_width / self.width
        if not self.screen:
            self.screen = pygame.display.set_mode([self.pg_width, self.pg_width])
        render_landscape(self.screen, self.world)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
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

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=0):
        pg_delay = 50
        self.reset()
        for i in range(n_steps):
            self.step_swarms()
            if render:
                self.screen = screen
                self.render(screen, pg_delay)
        # p1, p2 = self.swarms[0], self.swarms[1]
        # objs = fit_dist([p1, p2], self.world)
        ps1, ps2 = self.swarms[0].ps, self.swarms[1].ps
        objs = contrastive_pop([swarm.ps for swarm in self.swarms], self.width)
        bcs = ps1.mean(0)
        return objs, bcs

    def set_landscape(self, landscape):
        assert landscape is not None
        self.world = landscape


class ParticleGym(ParticleSwarmEnv, MultiAgentEnv):
    def __init__(self, width, swarm_cls, n_policies, n_pop, pg_width=500, n_chan=1, fully_observable=False, fov=4,
                 rotated_observations=False, translated_observations=False):
        super().__init__(
            width, swarm_cls, n_policies, n_pop, n_chan=n_chan, pg_width=pg_width, fully_observable=fully_observable, fov=fov,
            rotated_observations=rotated_observations)
        self.actions = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]

        # Each agent observes 2D patch around itself. Each cell has multiple channels. 3D observation.
        # Map policies to agent observations.
        self.observation_spaces = {i: gym.spaces.Box(-1.0, 1.0, shape=(n_chan, self.patch_ws[i], self.patch_ws[i]))
                                   for i in range(n_policies)}
        # self.observation_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(n_chan, patch_ws[i], patch_ws[i]))
        # self.action_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(2,))
        #                       for i in range(n_policies)}

        # Can move to one of four adjacent tiles
        self.action_spaces = {i: gym.spaces.Discrete(len(self.actions))
                              for i in range(n_policies)}

        self.max_episode_steps = self.get_max_steps()
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
        actions = {k: self.actions[v] for k, v in actions.items()}
        assert self.world is not None
        swarm_acts = {i: {} for i in range(len(self.swarms))}
        [swarm_acts[i].update({j: action}) for (i, j), action in actions.items()]
#       batch_swarm_acts = {j: np.vstack([swarm_acts[j][i] for i in range(self.swarms[j].n_pop)])
#                           for j in range(len(self.swarms))}
        batch_swarm_acts = {i: np.vstack([swarm_acts[i][j] if (i, j) in actions else self.dead_action for j in range(self.swarms[i].n_pop)])
                            for i in range(len(self.swarms))}
        [swarm.update(scape=self.world, accelerations=batch_swarm_acts[i]) for i, swarm in enumerate(self.swarms)]
        obs = self.get_particle_observations()
        # Dones before rewards, in case reward is different e.g. at the last step
        self.dones = self.get_dones()
        rew = self.get_reward()
        self.dones.update({(i, j): rew[(i, j)] > 0 for i, j in rew})
        info = {agent_k: {'world_key': self.world_key, 'agent_id': agent_k} for agent_k in obs}
        self.n_step += 1
        assert self.world is not None
#       print(rew)
        # print(dones)
        return obs, rew, self.dones, info

    def get_dones(self):
        dones = {(i, j): False for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        dones.update({'__all__': self.n_step > 0 and self.n_step == self.max_episode_steps})
        return dones

    def get_particle_observations(self):
        return {(i, j): swarm.get_observations(scape=self.world, flatten=False)[j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}

    def get_reward(self):
        swarm_rewards = [swarm.get_rewards(self.world, self.n_step, self.max_episode_steps) for swarm in self.swarms]
        rew = {(i, j): swarm_rewards[i][j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        return rew

    def get_max_steps(self):

        # This is kind of arbitrary
        return self._width **2 // 2


def regret_fitness(regret_loss):
    return regret_loss


class ParticleGymRLlib(ParticleGym):
    def __init__(self, cfg):
        self.stats = []
        self.world = None
        self.world_gen_sequence = None
        evaluate = cfg.pop("evaluate")
        self.need_world_reset = False
        self.obj_fn_str = cfg.pop('objective_function')
        self.fully_observable = cfg.get('fully_observable')
        self.regret_losses = []

        # Global knowledge of number of eval envs for incrementing eval world idx
        self.num_eval_envs = cfg.pop('num_eval_envs', None)
        if self.evaluate:
            assert self.num_eval_envs is not None

        # Target reward world should elicit if using min_solvable objective
        trg_rew = cfg.pop('target_reward', 0)

        super().__init__(**cfg)
        self.next_n_pop = self.n_pop
        obj_fn = globals()[self.obj_fn_str + '_fitness'] if self.obj_fn_str else None
        
        if obj_fn == min_solvable_fitness:
            # TODO: this is specific to the maze subclass
            # The maximum reward
            max_rew = self.max_episode_steps
            obj_fn = partial(obj_fn, max_rew=max_rew, trg_rew=trg_rew)
        self.objective_function = obj_fn

        # if evaluate:
            # Agents should be able to reach any tile within the initial neighborhood by a shortest path.
            # self.max_steps = max(self.fovs) * 2
            # self.reset = partial(ParticleEvalEnv.reset, self)
            # self.get_reward = partial(ParticleEvalEnv.get_eval_reward, self, self.get_reward)
        self.world_key = None
        self.last_world_key = self.world_key
        self.next_world = None

    def set_world(self, world):
        """
        Set the world (from some external process, e.g. world-generator optimization), and set the env to be reset at
        the next step.
        """
        # This will be set as the current world at the next reset
        self.next_world = world.reshape(self.width, self.width)
        self.need_world_reset = True
        # self.worlds = {idx: worlds[idx]}
        # print('set worlds ', worlds.keys())

    def get_dones(self):
        dones = super().get_dones()

        # We'll take an additional step in the old world, then reset. Not the best but it works.
        dones['__all__'] = dones['__all__'] or self.need_world_reset
        return dones

    def set_world_eval(self, world: np.array, idx):
        self.world_key = idx
        self.set_world(world)
        self.set_landscape(self.world)

    def reset(self):
        self.n_pop = self.next_n_pop
        self.world = self.next_world
        # self.next_world = None
        self.need_world_reset = False
        # print('reset w/ worlds', self.worlds.keys())
        # world_idx = list(self.worlds.keys())[self.world_idx]
        # world = self.worlds[world_idx]
        # self.set_landscape(np.array(world).reshape(self.width, self.width))
        # self.world_idx = (self.world_idx + 1) % len(self.worlds)

        obs = super().reset()

        self.stats.append((self.world_key, {agent_id: 0 for agent_id in obs}))

        # This should be getting flushed out regularly
        # assert len(self.stats) <= 100

        return obs

    def set_regret_loss(self, losses):
        if self.last_world_key not in losses:
            # assert self.evaluate, 'Samples should cover all simultaneously evaluated worlds except during evaluation.'
            return
        loss = losses[self.last_world_key]
        self.regret_losses.append((self.last_world_key, loss))

    def get_world_stats(self, evaluate=False, quality_diversity=False):
        """
        Return the fitness (and behavior characteristics) achieved by the world after an episode of simulation. Note
        that this only returns the fitness of the latest episode.
        """
        # On the first iteration, the episode runs for max_steps steps. On subsequent calls to rllib's trainer.train(), the
        # reset() call occurs on the first step (resulting in max_steps - 1).
        if not evaluate:
            # print(f'get world stats at step {self.n_step}')
            assert self.max_episode_steps - 1 <= self.n_step <= self.max_episode_steps + 1

        n_pop = self.swarms[0].ps.shape[0]
        qd_stats = []

        if not evaluate:
            assert len(self.stats) == 1

        for (world_key, agent_rewards), (world_key_2, regret_loss) in zip(self.stats, self.regret_losses):
            assert world_key == world_key_2

            # Convert agent to policy rewards
            swarm_rewards = [[agent_rewards[(i, j)] for j in range(n_pop)] for i in range(len(self.swarms))]

            # Return a mapping of world_key to a tuple of stats in a format that is compatible with qdpy
            # stats are of the form (world_key, qdpy_stats, policy_rewards)
            if quality_diversity:
                # Objective (negative fitness of protagonist population) and measures (antagonist population fitnesses)
                world_stats = {world_key: ((-np.mean(swarm_rewards[0]),), [np.mean(sr) for sr in swarm_rewards[1:]])}
            else:
                # Objective and placeholder measures
                if self.obj_fn_str == 'regret':
                    obj = self.objective_function(regret_loss)
                else:
                    obj = self.objective_function(swarm_rewards)
                world_stats = (world_key, ((obj,), [0, 0]))

            # Add per-policy stats
            world_stats = (*world_stats, [np.mean(sr) for sr in swarm_rewards])

            qd_stats.append(world_stats)

        self.stats = []
        self.regret_losses = []

        return qd_stats

    def get_reward(self):
        rew = super().get_reward()

        if len(self.stats) == 0:
            assert self.need_world_reset
            return rew

        # Store rewards so that we can compute world fitness according to progress over duration of level
        for k, v in rew.items():
            self.stats[-1][1][k] += v

        return rew
 

class ParticleMazeEnv(ParticleGymRLlib):
    empty_chan = 0
    wall_chan = 1
    start_chan = 2
    goal_chan = 3
    unique_chans = [start_chan, goal_chan]

    def __init__(self, cfg):
        cfg.update({'n_chan': 4})
        cfg['swarm_cls'] = cfg.get('swarm_cls', MazeSwarm)
        self.evaluate = cfg.get('evaluate')
        self.fully_observable = cfg.get('fully_observable')
        if self.evaluate:
            self.eval_maze_i = 0
        super().__init__(cfg)
        # TODO: maze-specific evaluation scenarios (will currently break)
        self.n_policies = n_policies = len(self.swarms)
        # self.patch_ws = patch_ws = [int(fov * 2 + 1) for fov in self.fovs]

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

        self.observation_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(self.patch_ws[i], self.patch_ws[i], self.n_obs_chan))
                                   for i in range(n_policies)}
        self.observation_space = None


    def get_particle_observations(self, padding_mode='wrap', surplus_padding=0):

        # Remove the spawn-point channel and replace the spawn point with an empty tile
        obs_scape = np.vstack((self.world[:self.start_chan], self.world[self.start_chan+1:]))
        obs_scape[self.empty_chan, self.start_idx[0], self.start_idx[1]] = 1

        if self.fully_observable and not self.rotated_observations:
            obs = {}
            for i, swarm in enumerate(self.swarms):
                obs_i = swarm.get_full_observations(
                    scape=obs_scape,
                    flatten=False,
                    )
        else:
            obs = {}
            for i, swarm in enumerate(self.swarms):
                obs_i = swarm.get_observations(
                    scape=obs_scape,
                    flatten=False,
                    padding_mode=padding_mode, 
                    surplus_padding=surplus_padding,
                    )

        for j in range(swarm.n_pop):
            obs[(i, j)] = obs_i[j]

#           obs = {(i, j): swarm.get_observations(
#                                   scape=np.vstack((self.world[:self.start_chan], self.world[self.start_chan+1:])),
#                                   flatten=False)[j].transpose(1, 2, 0) for i, swarm in enumerate(self.swarms)
#               for j in range(swarm.n_pop)}
        return obs


    def set_world(self, world):
        """
        Convert an encoding produced by the generator into a world map. The encoding has channels (empty, wall, start/goal)
        :param world: Encoding, optimized directly or produced by a world-generator.
        """
        w = np.zeros((self.width, self.width), dtype=np.int)
        w.fill(1)
        w[1:-1, 1:-1] = world
        self.start_idx = np.argwhere(w == 2)
        self.goal_idx = np.argwhere(w == 3)
        assert len(self.start_idx) == 1
        assert len(self.goal_idx) == 1
        self.start_idx = self.start_idx[0]
        self.goal_idx = self.goal_idx[0]
        self.world_flat = w
        self.next_world = discrete_to_onehot(w)
        self.need_world_reset = True

    def step(self, actions):

        obs, rew, done, info = super().step(actions)
        # print(f"step {self.n_step} world {self.world_key}, done: {done}, max steps {self.max_steps}")

        [obs.pop(k) for k in self.dead if k in obs]
        [rew.pop(k) for k in self.dead if k in rew]
        [done.pop(k) for k in self.dead if k in done]
        [info.pop(k) for k in self.dead if k in info]

        self.dead.update(k for k in obs if done[k])

        return obs, rew, done, info

    def step_swarms(self):
        [s.update(scape=self.world, obstacles=self.world) for s in self.swarms]

    def reset(self):
        self.dead = set({})
        # FIXME: redundant observations are being taken here
        # print(f'reset world {self.world_key} on step {self.n_step}')

        # Ugly hack to deal with eval envs resetting before end of sample. last_world_key = world_key during training,
        # since the new world key is set right before reset. During eval, this is actually the last world_key, since
        # reset happens before the end of the evaluate() batch.
        # print(f"reset world {self.world_key} on step {self.n_step}")
        self.last_world_key = self.world_key

        # Incrementing eval worlds to ensure each world is evaluated an equal number of times over training
        if self.evaluate:
            world_keys = list(self.worlds.keys())
            # print('eval\n')
            # FIXME: maybe inefficient to call index
            self.world_key = world_keys[(world_keys.index(self.world_key) + self.num_eval_envs) % len(self.worlds)]
            self.set_world(self.worlds[self.world_key])
#           w = eval_mazes_onehots[self.eval_maze_i].astype(int)
#           self.start_idx = np.argwhere(w[2:3] == 1)[0, 1:]
#           self.goal_idx = np.argwhere(w[3:4] == 1)[0, 1:]
#           self.next_world = w
#           
#           # Unfancy,
#           self.eval_maze_i = (self.eval_maze_i + 1) % len(eval_mazes)

        obs = super().reset()

        for swarm in self.swarms:
            swarm.ps[:] = self.start_idx

        obs = self.get_particle_observations()

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

    def render(self, mode='human', pg_delay=0, pg_width=None):
        if not pg_width:
            pg_width = self.pg_width
        pg_scale = pg_width / self.width
        if not self.screen:
            self.screen = pygame.display.set_mode([pg_width, pg_width])
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
        self._render_level(self.world, self.start_idx, self.goal_idx, pg_scale, pg_delay, mode)

    def _render_level(self, world, start_idx, goal_idx, pg_scale, pg_delay, mode):
        render_landscape(self.screen, -1 * world[1] + 1)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.draw.rect(self.screen, goal_color, (goal_idx[0] * pg_scale, goal_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        pygame.draw.rect(self.screen, start_color, (start_idx[0] * pg_scale, start_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
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
triangle_scale = 2.6


class DirectedMazeEnv(ParticleMazeEnv):
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
            gym.spaces.Box(0.0, 1.0, shape=(self.patch_ws[i], self.patch_ws[i], self.n_obs_chan + 4))
                                   for i in range(self.n_policies)}
        self.action_spaces = {i : gym.spaces.Discrete(4) for i in range(self.n_policies)}

    def _render_level(self, world, start_idx, goal_idx, pg_scale, pg_delay, mode):
        render_landscape(self.screen, -1 * world[1] + 1)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.draw.rect(self.screen, goal_color, (goal_idx[0] * pg_scale, goal_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        pygame.draw.rect(self.screen, start_color, (start_idx[0] * pg_scale, start_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
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

    def render(self, mode='human', pg_delay=0, pg_width=None):
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
        for pi, policy_i in enumerate(self.swarms):
            ag_i = 0
            for agent_pos, agent_direction in zip(policy_i.ps, policy_i.directions):
                agent_pos = agent_pos.astype(int) + 0.5
                points = [np.array(point) * triangle_scale + agent_pos * pg_scale for point in directed_triangles[agent_direction]]
                pygame.draw.polygon(self.screen, player_colors[pi], points)
                ag_i = (ag_i + 1) % 4
        if mode == 'human':
            pygame.display.update()
            pygame.time.delay(pg_delay)
            return True

        arr = pygame.surfarray.array3d(self.screen)
        if mode == 'rgb':
            return arr

    def get_particle_observations(self):
        obs = super().get_particle_observations(padding_mode='constant', surplus_padding=self.fovs[0])
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
                                   self.patch_ws[i], self.patch_ws[i])) for i in range(self.n_policies)}
        self.action_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(self.n_chan + 1 + n_aux_chans,
                                   self.patch_ws[i], self.patch_ws[i])) for i in range(self.n_policies)}

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
#        :param fovs: The fields of vision of the policies (how far they can see in each direction.
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
#        for swarm, init_nb, fov in zip(self.swarms, self.init_nbs, self.fovs):
#            for ps, nb in zip(swarm.ps.astype(int), init_nb):
#                # TODO: vectorize this
#                landscape[ps[0] - fov: ps[0] + fov + 1, ps[1] - fov: ps[1] + fov + 1] = nb
#        self.set_landscape(landscape)
#        # obs = [np.reshape(nb, (nb.shape[0], nb.shape[1], -1)) for nb in self.init_nbs]
#        return obs
#
#    def get_eval_reward(self, og_get_reward):
#        """Reward for policies when their agents move the best tile that was in their initial field of vision."""
#        if not self.dones['__all__']:
#            return {agent_id: 0 for agent_id in self.agent_ids}
#        fovs = [int((nb[0].shape[0] - 1) / 2) for nb in self.init_nbs]
#        # nbs = [nb[fov - 1: fov + 2] for nb, fov in zip(self.init_nbs, fovs)]
#        nbs = self.init_nbs
#        # Condition is satisfied when
#        og_rewards = og_get_reward()
#        rewards = {agent_id: int(np.max(nb) == og_rewards[agent_id]) for agent_id, nb in zip(self.agent_ids, self.init_nbs)}
#        return rewards