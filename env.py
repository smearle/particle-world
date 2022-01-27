import sys
from functools import partial
from pdb import set_trace as TT

import gym
import numpy as np
import pygame
import ray
from ray import rllib
from ray.rllib import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec

from generator import render_landscape
from swarm import NeuralSwarm, GreedySwarm, contrastive_pop, contrastive_fitness
from utils import discrete_to_onehot

player_colors = [
    (0, 0, 255),
    (255, 0, 0),
    # (0, 255, 0),  # green
    (255, 0, 255),
    # (255, 255, 0),  # yellow
    (0, 255, 255),
]
goal_color = (0, 255, 0)
start_color = (255, 255, 0)


class ParticleSwarmEnv(object):
    """An environment in continuous 2D space in which populations of particles can accelerate in certain directions,
    propelling themselves toward desirable regions in the fitness world."""
    def __init__(self, width, swarm_cls, n_policies, n_pop, n_chan=1, pg_width=None):
        self.n_chan = n_chan
        if not pg_width:
            pg_width = width
        self.pg_width=pg_width
        self.world = None
        self.landscape_set = False
        self.swarms = None
        self.width = width
        # self.fovs = [si+1 for si in range(n_policies)]
        self.fovs = [2 for si in range(n_policies)]
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
        assert self.world is not None
        # assert len(self.world.shape) == 2
        [swarm.reset(scape=self.world) for swarm in self.swarms]

    def step_swarms(self):
        [s.update(scape=self.world) for s in self.swarms]

    def render(self, mode='human', pg_delay=0):
        # print('render')
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
        self.landscape_set = True
        self.world = landscape


def gen_policy(i, observation_space, action_space, fov):
    config = {
        "model": {
            "custom_model_config": {
                "fov": fov,
            }
        }
    }
    return PolicySpec(config=config, observation_space=observation_space, action_space=action_space)


class ParticleGym(ParticleSwarmEnv, MultiAgentEnv):
    def __init__(self, width, swarm_cls, n_policies, n_pop, max_steps, pg_width=500, n_chan=1):
        super().__init__(width, swarm_cls, n_policies, n_pop, n_chan=n_chan, pg_width=pg_width)
        patch_ws = [fov * 2 + 1 for fov in self.fovs]

        # Each agent observes 2D patch around itself. Each cell has multiple channels. 3D observation.
        # Map policies to agent observations.
        self.observation_spaces = {i: gym.spaces.Box(-1.0, 1.0, shape=(patch_ws[i], patch_ws[i], n_chan))
                                   for i in range(n_policies)}
        # self.observation_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(n_chan, patch_ws[i], patch_ws[i]))
        # self.action_spaces = {i: gym.spaces.Box(0.0, 1.0, shape=(2,))
        #                       for i in range(n_policies)}

        # Can move to one of four adjacent tiles
        self.action_spaces = {i: gym.spaces.Discrete(4)
                              for i in range(n_policies)}

        self.max_steps = max_steps
        self.n_step = 0


    def reset(self):
        # print('reset', self.worlds.keys())
        self.n_step = 0
        # TODO: reset to a world in the archive, via rllib config args?
        super().reset()
        obs = self.get_particle_observations()
        return obs

    def step(self, actions):
        actions = {k: [(1, 0), (0, 1), (0, -1), (-1, 0), (0, 0)][v] for k, v in actions.items()}
        assert self.world is not None
        swarm_acts = {i: {} for i in range(len(self.swarms))}
        [swarm_acts[i].update({j: action}) for (i, j), action in actions.items()]
        batch_swarm_acts = {j: np.vstack([swarm_acts[j][i] for i in range(self.swarms[j].n_pop)])
                            for j in range(len(self.swarms))}
        [swarm.update(scape=self.world, accelerations=batch_swarm_acts[i]) for i, swarm in enumerate(self.swarms)]
        obs = self.get_particle_observations()
        # Dones before rewards, in case reward is different e.g. at the last step
        self.dones = self.get_dones()
        rew = self.get_reward()
        info = {}
        self.n_step += 1
        assert self.world is not None
        return obs, rew, self.dones, info

    def get_dones(self):
        dones = {(i, j): False for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        dones.update({'__all__': self.n_step > 0 and self.n_step % (self.max_steps - 1) == 0})
        if dones['__all__']:
            self.landscape_set = False
        return dones

    def get_particle_observations(self):
        return {(i, j): swarm.get_observations(scape=self.world, flatten=False)[j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}

    def get_reward(self):
        swarm_rewards = [swarm.get_rewards(self.world) for swarm in self.swarms]
        rew = {(i, j): swarm_rewards[i][j] for i, swarm in enumerate(self.swarms)
                for j in range(swarm.n_pop)}
        return rew


class ParticleGymRLlib(ParticleGym):
    def __init__(self, cfg):
        self.rewards = None
        self.world = None
        evaluate = cfg.pop("evaluate")
        self.need_world_reset = False
        super().__init__(**cfg)
        # if evaluate:
            # Agents should be able to reach any tile within the initial neighborhood by a shortest path.
            # self.max_steps = max(self.fovs) * 2
            # self.reset = partial(ParticleEvalEnv.reset, self)
            # self.get_reward = partial(ParticleEvalEnv.get_eval_reward, self, self.get_reward)
        self.world_idx = None
        self.next_world = None

    def set_worlds(self, worlds: dict, idx_counter=None):
        # self.world_idx = 0
        if idx_counter:
            self.world_idx = ray.get(idx_counter.get.remote(hash(self)))
        else:
            self.world_idx = np.random.choice(list(worlds.keys()))
        self.set_world(worlds[self.world_idx])

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
        dones['__all__'] = self.need_world_reset or self.n_step == self.max_steps + 1
        return dones

    def set_world_eval(self, world: np.array, idx):
        self.world_idx = idx
        self.set_world(world)
        self.set_landscape(self.world)

    def reset(self):
        self.world = self.next_world
        # self.next_world = None
        self.need_world_reset = False
        # print('reset w/ worlds', self.worlds.keys())
        # world_idx = list(self.worlds.keys())[self.world_idx]
        # world = self.worlds[world_idx]
        # self.set_landscape(np.array(world).reshape(self.width, self.width))
        # self.world_idx = (self.world_idx + 1) % len(self.worlds)

        obs = super().reset()
        self.rewards = {agent_id: 0 for agent_id in obs}

        return obs

    def get_fitness(self, evaluate=False):
        """
        Return the fitness (and behavior characteristics) achieved by the world after an episode of simulation. Note
        that this only returns the fitness of the latest episode.
        """
        # On the first iteration, the episode runs for max_steps steps. On subsequent calls to rllib's trainer.train(), the
        # reset() call occurs on the first step (resulting in max_steps - 1).
        if not evaluate:
            assert self.max_steps - 1 <= self.n_step <= self.max_steps
        n_pop = self.swarms[0].ps.shape[0]

        # Convert agent to policy rewards
        swarm_rewards = [[self.rewards[(i, j)] for j in range(n_pop)] for i in range(len(self.swarms))]
        # Storing the objective and BCs corresponding mapped to the world_idx, for evolving worlds.

        qd_stats = {self.world_idx: ((-np.mean(swarm_rewards[0]),), [np.mean(sr) for sr in swarm_rewards[1:]])}
        return qd_stats

    def get_reward(self):
        rew = super().get_reward()
        # Store rewards so that we can compute world fitness according to progress over duration of level
        for k, v in rew.items():
            self.rewards[k] += v

        return rew


# This is a dummy class not currently used, except by its parent ParticleGymRLlib, which borrows its methods when instantiating,
# if in evaluation mode.
class ParticleEvalEnv(ParticleGymRLlib):
    """
    An environment that assumes that a feed-forward (i.e. memoryless) neural network is "best" at the task of navigating
    a continuous fitness world when it simply greedly moves to the best tile in its field of vision. Randomly generate
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
        """Generate uniform random fitness world, then set to 0 all tiles not in the initial field of vision of any agent."""
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
    
    
eval_mazes = [
    np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
    np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]),
]
# Convert to 3-channel probability distribution (or agent action)-type representation
eval_mazes_probdists = []
for i, y in enumerate(eval_mazes):
    y = discrete_to_onehot(y)
    z = np.empty((y.shape[0] - 1, y.shape[1], y.shape[2]))
    z[:3] = y[:3]
    z[2] -= y[3]
    eval_mazes_probdists.append(z)

eval_mazes_onehots = []
for y in eval_mazes:
    eval_mazes_onehots.append(discrete_to_onehot(y))


class ParticleMazeEnv(ParticleGymRLlib):
    def __init__(self, cfg):
        cfg.update({'n_chan': 3})
        self.evaluate = cfg.get('evaluate')
        if self.evaluate:
            self.eval_maze_i = 0
        super().__init__(cfg)
        # TODO: maze-specific evaluation scenarios (will currently break)
        n_policies = len(self.swarms)
        patch_ws = [fov * 2 + 1 for fov in self.fovs]

        # We only ask the generator for 3 chans. 3rd shares goal and start, so we observe a 4-channel onehot encoding
        self.observation_spaces = {i: gym.spaces.Box(-1.0, 1.0, shape=(patch_ws[i], patch_ws[i], self.n_chan + 1))
                                   for i in range(n_policies)}

        # Represent empty, wall, and goal as onehot. This attribute is to inform the generator.
        # self.particle_draw_size = 0.1

    def set_world(self, world):
        """
        Convert an encoding produced by the generator into a world map. The encoding has channels (empty, wall, starg/goal)
        :param world: Encoding, optimized directly or produced by a world-generator.
        """
        # Convert world from 3D to 2D, collapsing channel axis.
        v = np.reshape(world, (self.n_chan, self.width, self.width))

        # Empty and wall tiles
        w = np.argmax(v[:2], axis=(0))

        # Force the map to include border walls
        w[0] = w[-1] = w[:, 0] = w[:, -1] = 1

        # Goal and start tiles
        # Cannot end on obstacles
        vg = v[2] - v[1]
        goal_idxs = np.argwhere(vg == vg.max())
        self.goal_idx = goal_idxs[np.random.randint(goal_idxs.shape[0])]

        # Cannot start on obstacles
        vs = v[2] + v[1]
        start_idxs = np.argwhere(vs == vs.min())
        self.start_idx = start_idxs[np.random.randint(start_idxs.shape[0])]
        w[self.start_idx[0], self.start_idx[1]] = 2
        w[self.goal_idx[0], self.goal_idx[1]] = 3
        self.world_flat = w
        self.next_world = discrete_to_onehot(w)
        self.need_world_reset = True



    def step(self, actions):
        # print(f"step {self.n_step} world {self.world_idx}")

        return super().step(actions)

    def step_swarms(self):
        [s.update(scape=self.world, obstacles=self.world) for s in self.swarms]

    def reset(self):
        # FIXME: redundant observations are being taken here
        # print(f'reset world {self.world_idx} on step {self.n_step}')
        if self.evaluate:
            w = eval_mazes_onehots[self.eval_maze_i].astype(int)
            self.start_idx = np.argwhere(w[2:3] == 1)[0, 1:]
            self.goal_idx = np.argwhere(w[3:4] == 1)[0, 1:]
            self.next_world = w
            
            # Unfancy,
            self.eval_maze_i = (self.eval_maze_i + 1) % len(eval_mazes)

        obs = super().reset()

        for swarm in self.swarms:
            swarm.ps[:] = self.start_idx

        obs = self.get_particle_observations()

        return obs

    def simulate(self, n_steps, generator, render=False, screen=None, pg_scale=1, pg_delay=0):
        generator.world = self.world  # just for rendering
        return super().simulate(n_steps=n_steps, generator=generator, render=render, screen=screen, pg_scale=pg_scale, pg_delay=pg_delay)

    def render(self, mode='human', pg_delay=0):
        # print('render')
        pg_scale = self.pg_width / self.width
        if not self.screen:
            self.screen = pygame.display.set_mode([self.pg_width, self.pg_width])
        render_landscape(self.screen, -1 * self.world[1] + 1)
        # Did the user click the window close button? Exit if so.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.draw.rect(self.screen, goal_color, (self.goal_idx[0] * pg_scale, self.goal_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
        pygame.draw.rect(self.screen, start_color, (self.start_idx[0] * pg_scale, self.start_idx[1] * pg_scale, 1.0 * pg_scale, 1.0 * pg_scale))
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
