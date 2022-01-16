import sys
from pdb import set_trace as TT

import gym
import numpy as np
import pygame
import ray
from ray import rllib

from generator import render_landscape
from swarm import NeuralSwarm, GreedySwarm, contrastive_pop

player_colors = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]


class ParticleSwarmEnv(object):
    """An environment in continuous 2D space in which populations of particles can accelerate in certain directions,
    propelling themselves toward desirable regions in the fitness landscape."""
    def __init__(self, width, n_policies, n_pop, pg_width=None):
        if not pg_width:
            pg_width = width
        self.pg_width=pg_width
        self.landscape = None
        self.swarms = None
        self.width = width
        # self.fovs = [si+1 for si in range(n_policies)]
        self.fovs = [3 for si in range(n_policies)]
        self._gen_swarms(n_policies, n_pop, self.fovs)
        self.particle_draw_size = 1
        self.n_steps = None
        self.screen = None

    def _gen_swarms(self, n_policies, n_pop, fovs):
        self.swarms = [
            # GreedySwarm(
            NeuralSwarm(
                world_width=self.width,
                n_pop=n_pop,
                fov=fovs[si],
                # trg_scape_val=trg)
                trg_scape_val=1.0)
            for si, trg in zip(range(n_policies), np.arange(n_policies) / (n_policies - 1))]

    def set_policies(self, policies):
        self.swarms = policies

    def reset(self, landscape):
        self.landscape = landscape
        [swarm.reset(landscape) for swarm in self.swarms]

    def step_swarms(self):
        [s.update() for s in self.swarms]

    def render(self, screen=None, pg_delay=0):
        pg_scale = self.pg_width / self.width
        if not self.screen:
            self.screen = pygame.display.set_mode([self.pg_width, self.pg_width])
        render_landscape(self.screen, self.landscape)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        for pi, policy_i in enumerate(self.swarms):
            for agent_pos in policy_i.ps:
                pygame.draw.circle(self.screen, player_colors[pi], agent_pos * pg_scale,
                                   self.particle_draw_size * pg_scale)
        pygame.display.update()
        pygame.time.delay(pg_delay)
        return True

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=1):
        for i in range(n_steps):
            self.step_swarms()
            if render:
                self.screen = screen
                self.render(screen, pg_delay)
        # p1, p2 = self.swarms[0], self.swarms[1]
        # objs = fit_dist([p1, p2], self.landscape)
        ps1, ps2 = self.swarms[0].ps, self.swarms[1].ps
        objs = contrastive_pop([swarm.ps for swarm in self.swarms], generator.width)
        bcs = ps1.mean(0)
        return objs, bcs


class ParticleGym(ParticleSwarmEnv, rllib.env.multi_agent_env.MultiAgentEnv):
    def __init__(self, width, n_policies, n_pop, max_steps, pg_width=1000):
        n_chan = 1
        super().__init__(width, n_policies, n_pop, pg_width=pg_width)
        patch_ws = [fov * 2 + 1 for fov in self.fovs]
        # self.observation_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(n_chan, patch_ws[i], patch_ws[i]))
        self.observation_spaces = {i: gym.spaces.Box(-1.0, 1.0, shape=(n_chan, patch_ws[i] * patch_ws[i]))
                                   for i in range(n_policies)}
        # self.action_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(2,))
        #                       for i in range(n_policies)}
        self.action_spaces = {i: gym.spaces.MultiDiscrete((3, 3))
                              for i in range(n_policies)}
        self.max_steps = max_steps
        self.n_step = 0

    def set_landscape(self, landscape):
        self.landscape = landscape

    def reset(self):
        self.n_step = 0
        # TODO: reset to a landscape in the archive, via rllib config args?
        super().reset(landscape=self.landscape)
        obs = self.get_particle_observations()
        return obs

    def step(self, actions):
        swarm_acts = {i: {} for i in range(len(self.swarms))}
        [swarm_acts[i].update({j: action}) for (i, j), action in actions.items()]
        batch_swarm_acts = {j: np.vstack([swarm_acts[j][i] for i in range(self.swarms[j].n_pop)])
                            for j in range(len(self.swarms))}
        [swarm.update(accelerations=batch_swarm_acts[i]) for i, swarm in enumerate(self.swarms)]
        obs = self.get_particle_observations()
        rew = self.get_reward()
        done = self.get_dones()
        info = {}
        self.n_step += 1
        return obs, rew, done, info

    def get_dones(self):
        dones = {(i, j): False for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        dones.update({'__all__': self.n_step > 0 and self.n_step % self.max_steps == 0})
        return dones

    def get_particle_observations(self):
        return {(i, j): swarm.get_observations()[j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}

    def get_reward(self):
        return {(i, j): swarm.get_rewards()[j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}


class ParticleGymRLlib(ParticleGym):
    def __init__(self, cfg):
        super().__init__(**cfg)


class ParticleMazeEnv(ParticleSwarmEnv):
    def __init__(self, width, n_policies, n_pop):
        super().__init__(width, n_policies, n_pop)
        self.particle_draw_size = 0.8

    def reset(self, landscape):
        self.landscape = landscape.round().astype(int)
        # TODO: invisible fitness landscape atm!
        [swarm.reset(landscape) for swarm in self.swarms]

    def step_swarms(self):
        [s.update(obstacles=self.landscape) for s in self.swarms]

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=1):
        generator.landscape = self.landscape  # just for rendering
        return super().simulate(n_steps=n_steps, generator=generator, render=render, screen=screen, pg_scale=pg_scale, pg_delay=pg_delay)
