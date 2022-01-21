import sys
from functools import partial
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
    def __init__(self, swarm_cls, width, n_policies, n_pop, pg_width=None):
        if not pg_width:
            pg_width = width
        self.pg_width=pg_width
        self.landscape = None
        self.landscape_set = False
        self.swarms = None
        self.width = width
        # self.fovs = [si+1 for si in range(n_policies)]
        self.fovs = [3 for si in range(n_policies)]
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

    def render(self, mode='human', pg_delay=100):
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
        pg_delay = 50
        self.reset()
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
    def __init__(self, swarm_cls, width, n_policies, n_pop, max_steps, pg_width=500):
        n_chan = 1
        super().__init__(swarm_cls, width, n_policies, n_pop, pg_width=pg_width)
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
        # Dones before rewards, in case reward is different e.g. at the last step
        self.dones = self.get_dones()
        rew = self.get_reward()
        info = {}
        self.n_step += 1
        assert self.landscape is not None
        return obs, rew, self.dones, info

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
        swarm_rewards = [swarm.get_rewards(self.landscape) for swarm in self.swarms]
        rew = {(i, j): swarm_rewards[i][j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        return rew


class ParticleGymRLlib(ParticleGym):
    def __init__(self, cfg):
        evaluate = cfg.pop("evaluate")
        super().__init__(**cfg)
        if evaluate:
            # Agents should be able to reach any tile within the initial neighborhood by a shortest path.
            self.max_steps = max(self.fovs) * 2
            self.reset = partial(ParticleEvalEnv.reset, self)
            self.get_reward = partial(ParticleEvalEnv.get_eval_reward, self, self.get_reward)
        self.world_idx = None

    def set_world(self, worlds: dict, idx_counter=None):
        # self.world_idx = 0
        if idx_counter:
            self.world_idx = ray.get(idx_counter.get.remote(hash(self)))
        else:
            self.world_idx = np.random.choice(list(worlds.keys()))
        self.world = worlds[self.world_idx].reshape(self.width, self.width)
        # self.worlds = {idx: worlds[idx]}
        self.fitnesses = {}
        # print('set worlds ', worlds.keys())

    def set_world_eval(self, world: np.array, idx):
        self.world_idx = idx
        self.world = world
        self.set_landscape(world)
        self.fitnesses = {}

    def reset(self):
        # print('reset w/ worlds', self.worlds.keys())
        # world_idx = list(self.worlds.keys())[self.world_idx]
        # world = self.worlds[world_idx]
        world = self.world
        self.set_landscape(world)
        # self.set_landscape(np.array(world).reshape(self.width, self.width))
        # self.world_idx = (self.world_idx + 1) % len(self.worlds)

        return super().reset()

    def get_fitness(self):
        return self.fitnesses

    def step(self, actions):
        obs, rew, dones, info = super().step(actions)
        if dones['__all__']:
            # world_idx = list(self.worlds.keys())[self.world_idx]
            self.fitnesses[self.world_idx] = (contrastive_pop([swarm.ps for swarm in self.swarms], self.width), ), (0, 0)

        return obs, rew, dones, info


# This is a dummy class not currently used, except by its parent ParticleGymRLlib, which borrows its methods when instantiating,
# if in evaluation mode.
class ParticleEvalEnv(ParticleGymRLlib):
    """
    An environment that assumes that a feed-forward (i.e. memoryless) neural network is "best" at the task of navigating
    a continuous fitness landscape when it simply greedly moves to the best tile in its field of vision. Randomly generate
    a map consisting of a neighborhood, and reward 1 when the policy moves to the tile with greatest value, otherwise 0.
    Episodes last one step.
    """
    def __init__(self, **cfg):
        """
        :param fovs: The fields of vision of the policies (how far they can see in each direction.
        """
        super().__init__(**cfg)
        raise Exception(f"{type(self)} is a dummy class.")

    def reset(self):
        """Generate uniform random fitness landscape, then set to 0 all tiles not in the initial field of vision of any agent."""
        self.fitnesses = {}
        og_scape = np.random.random((self.width, self.width))
        # Note that we're calling set_world on ourselves. Normally this is called externally before reset
        self.set_world_eval(og_scape, hash(self))
        obs = super(ParticleGymRLlib, self).reset()
        self.agent_ids = [agent_id for agent_id in obs]
        # Note that this is weird, allows borrowing by parent class. Will break the
        self.init_nbs = [swarm.get_observations(og_scape, flatten=False) for swarm in self.swarms]
        landscape = np.ones(og_scape.shape)
        for swarm, init_nb, fov in zip(self.swarms, self.init_nbs, self.fovs):
            for ps, nb in zip(swarm.ps.astype(int), init_nb):
                # TODO: vectorize this
                landscape[ps[0] - fov: ps[0] + fov + 1, ps[1] - fov: ps[1] + fov + 1] = nb
        self.set_landscape(landscape)
        # obs = [np.reshape(nb, (nb.shape[0], nb.shape[1], -1)) for nb in self.init_nbs]
        return obs

    def get_eval_reward(self, og_get_reward):
        """Reward for policies when their agents move the best tile that was in their initial field of vision."""
        if not self.dones['__all__']:
            return {agent_id: 0 for agent_id in self.agent_ids}
        fovs = [int((nb[0].shape[0] - 1) / 2) for nb in self.init_nbs]
        # nbs = [nb[fov - 1: fov + 2] for nb, fov in zip(self.init_nbs, fovs)]
        nbs = self.init_nbs
        # Condition is satisfied when
        og_rewards = og_get_reward()
        rewards = {agent_id: int(np.max(nb) == og_rewards[agent_id]) for agent_id, nb in zip(self.agent_ids, self.init_nbs)}
        return rewards


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
