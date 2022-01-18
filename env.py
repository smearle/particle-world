import sys
from pdb import set_trace as TT

import gym
import numpy as np
import pygame
import ray
from ray import rllib
from ray.rllib.policy.policy import PolicySpec

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
        self.landscape_set = False
        self.swarms = None
        self.width = width
        # self.fovs = [si+1 for si in range(n_policies)]
        self.fovs = [3 for si in range(n_policies)]
        self._gen_swarms(n_policies, n_pop, self.fovs)
        self.particle_draw_size = 0.3
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

    def set_policies(self, policies, trainer_config):
        # self.swarms = policies
        [swarm.set_nn(policy, i, self.observation_spaces[i], self.action_spaces[i], trainer_config) for i, (swarm, policy) in enumerate(zip(self.swarms, policies))]

    def reset(self):
        # assert self.landscape_set
        assert self.landscape is not None
        assert len(self.landscape.shape) == 2
        [swarm.reset(scape=self.landscape) for swarm in self.swarms]

    def step_swarms(self):
        [s.update(scape=self.landscape) for s in self.swarms]

    def render(self, mode='human', pg_delay=0):
        pg_delay = 0
        # print('render')
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

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=1):
        for i in range(n_steps):
            self.step_swarms()
            if render:
                self.screen = screen
                self.render(screen, pg_delay)
        # p1, p2 = self.swarms[0], self.swarms[1]
        # objs = fit_dist([p1, p2], self.landscape)
        ps1, ps2 = self.swarms[0].ps, self.swarms[1].ps
        objs = contrastive_pop([swarm.ps for swarm in self.swarms], self.width)
        bcs = ps1.mean(0)
        return objs, bcs

    def set_landscape(self, landscape):
        assert landscape is not None
        self.landscape_set = True
        self.landscape = landscape


def gen_policy(i, observation_space, action_space, fov):
    config = {
        "model": {
            "custom_model_config": {
                "fov": fov,
            }
        }
    }
    return PolicySpec(config=config, observation_space=observation_space, action_space=action_space)


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
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer


    def reset(self):
        # print('reset', self.worlds.keys())
        self.n_step = 0
        # TODO: reset to a landscape in the archive, via rllib config args?
        super().reset()
        obs = self.get_particle_observations()
        return obs

    def step(self, actions):
        assert self.landscape is not None
        swarm_acts = {i: {} for i in range(len(self.swarms))}
        [swarm_acts[i].update({j: action}) for (i, j), action in actions.items()]
        batch_swarm_acts = {j: np.vstack([swarm_acts[j][i] for i in range(self.swarms[j].n_pop)])
                            for j in range(len(self.swarms))}
        [swarm.update(scape=self.landscape, accelerations=batch_swarm_acts[i]) for i, swarm in enumerate(self.swarms)]
        obs = self.get_particle_observations()
        rew = self.get_reward()
        done = self.get_dones()
        info = {}
        self.n_step += 1
        assert self.landscape is not None
        return obs, rew, done, info

    def get_dones(self):
        dones = {(i, j): False for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        dones.update({'__all__': self.n_step > 0 and self.n_step % (self.max_steps - 1) == 0})
        if dones['__all__']:
            self.landscape_set = False
        return dones

    def get_particle_observations(self):
        return {(i, j): swarm.get_observations(scape=self.landscape)[j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}

    def get_reward(self):
        return {(i, j): swarm.get_rewards(self.landscape)[j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}


class ParticleGymRLlib(ParticleGym):
    def __init__(self, cfg):
        super().__init__(**cfg)
        self.world_idx = None

    def set_world(self, worlds):
        self.world_idx = 0
        self.worlds = worlds
        self.fitnesses = {}
        # print('set worlds ', worlds.keys())

    def reset(self):
        if not hasattr(self, 'worlds'):
            print(self)
        assert hasattr(self, 'worlds')
        # print('reset w/ worlds', self.worlds.keys())
        world_idx = list(self.worlds.keys())[self.world_idx]
        world = self.worlds[world_idx]
        self.set_landscape(np.array(world).reshape(self.width, self.width))
        self.world_idx = (self.world_idx + 1) % len(self.worlds)

        return super().reset()

    def get_fitness(self):
        return self.fitnesses

    def step(self, actions):
        obs, rew, dones, info = super().step(actions)
        if dones['__all__']:
            world_idx = list(self.worlds.keys())[self.world_idx]
            self.fitnesses[world_idx] = (contrastive_pop([swarm.ps for swarm in self.swarms], self.width), ), (0, 0)

        return obs, rew, dones, info


class ParticleMazeEnv(ParticleSwarmEnv):
    def __init__(self, width, n_policies, n_pop):
        super().__init__(width, n_policies, n_pop)
        self.particle_draw_size = 0.1

    def set_landscape(self, landscape):
        self.landscape = landscape.round().astype(int)

    def reset(self):
        # TODO: invisible fitness landscape atm!
        [swarm.reset(self.landscape) for swarm in self.swarms]

    def step_swarms(self):
        [s.update(scape=self.landscape, obstacles=self.landscape) for s in self.swarms]

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=1):
        generator.landscape = self.landscape  # just for rendering
        return super().simulate(n_steps=n_steps, generator=generator, render=render, screen=screen, pg_scale=pg_scale, pg_delay=pg_delay)
