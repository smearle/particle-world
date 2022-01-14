import sys

import gym
import numpy as np
import pygame

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
    def __init__(self, width, n_policies, n_pop):
        self.landscape = None
        self.swarms = None
        self.width = width
        self.gen_swarms(n_policies, n_pop)
        self.particle_draw_size = 1

    def gen_swarms(self, n_policies, n_pop):
        self.swarms = [
            # GreedySwarm(
            NeuralSwarm(
                world_width=self.width,
                n_pop=n_pop,
                # fov=si+1,
                fov=3,
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

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=1):
        for i in range(n_steps):
            self.step_swarms()
            if render:
                generator.render(screen)
                # Did the user click the window close button? Exit if so.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                for pi, policy_i in enumerate(self.swarms):
                    for agent_pos in policy_i.ps:
                        pygame.draw.circle(screen, player_colors[pi], agent_pos * pg_scale,
                                           self.particle_draw_size * pg_scale)
                pygame.display.update()
                pygame.time.delay(pg_delay)
        # p1, p2 = self.swarms[0], self.swarms[1]
        # objs = fit_dist([p1, p2], self.landscape)
        ps1, ps2 = self.swarms[0].ps, self.swarms[1].ps
        objs = contrastive_pop([swarm.ps for swarm in self.swarms], generator.width)
        bcs = ps1.mean(0)
        return objs, bcs


class ParticleGym(ParticleSwarmEnv, gym.Env):
    def __init__(self, width, n_policies, n_pop):
        super().__init__(self, width, n_policies, n_pop)

    def reset(self):
        obs = self.get_particle_observations()
        return obs

    def step(self, actions):
        self.step_particles(accelerations=actions)
        obs = self.get_particle_observations()
        rew = self.get_reward()
        done = False
        info = {}
        return obs, reward, done, info

    def get_reward(self):
        return 1


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
