import copy
from functools import partial
from pdb import set_trace as TT

import gym
import numpy as np
import ray
from ray.rllib import MultiAgentEnv

from minerl.herobraine.env_spec import EnvSpec
from envs.minecraft.touchstone import TouchStone
from envs.maze.swarm import min_solvable_fitness
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
        env = WorldEvolutionMultiAgentWrapper(env, env_config)
    else:
        env = WorldEvolutionWrapper(env, env_config)

    return env
    

def gen_init_world(width, depth, height, block_types):
    # return np.random.randint(0, len(block_types), size=(width, depth, height))
    return np.ones((width, depth, height), dtype=np.int32)


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
        self.next_world = None
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

    def set_world(self, world):
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


class WorldEvolutionWrapper(gym.Wrapper):
    """A wrapper facilitating world-evolution in a gym environment, allowing an external process to set the world (i.e.,
    the level layout), and collect statistics of interest (e.g., a player-agent's performance or "regret" on that 
    world)."""
    def __init__(self, env, env_cfg):
        super().__init__(env)
        self.agent_ids = [(i, j) for j in range(self.n_pop) for i in range(self.n_policies)]
        self.env = env
        self.need_world_reset = False
        self.stats = []
        self.regret_losses = []
        self.world_key = None
        self.last_world_key = self.world_key
        self.world = None
        self.world_gen_sequence = None
        self.next_world = None
        args = env_cfg.get('args')

        # Target reward world should elicit if using min_solvable objective
        trg_rew = args.target_reward

        self.obj_fn_str = args.objective_function
        obj_func = globals()[self.obj_fn_str + '_fitness'] if self.obj_fn_str else None
        if obj_func == min_solvable_fitness:

            # FIXME: This is never true! Also this is specific to the maze subclass
            # The maximum reward
            max_rew = 1

            obj_func = partial(obj_func, max_rew=max_rew, trg_rew=trg_rew)
        self.objective_function = obj_func


    def set_worlds(self, worlds: dict, idx_counter=None, next_n_pop=None, world_gen_sequences=None):
        """Assign a ``next_world`` to the environment, which will be loaded after the next step.
        
        We set flag ``need_world_reset`` to True, so that the next step will return done=True, and the environment will
        be reset, at which point, ``next_world`` will be loaded."""

        # Figure out which world to evaluate.
        # self.world_idx = 0
        if idx_counter:
            self.world_key = ray.get(idx_counter.get.remote(hash(self)))
        else:
            self.world_key = np.random.choice(list(worlds.keys()))

        # FIXME: hack
        # self.unwrapped.world_key = self.world_key

        # Assign this world to myself.
        self.set_world(worlds[self.world_key])
        self.worlds = self.unwrapped.worlds = worlds
        if next_n_pop is not None:
            self.next_n_pop = next_n_pop
        if world_gen_sequences is not None and len(world_gen_sequences) > 0:
            self.world_gen_sequence = world_gen_sequences[self.world_key]

        self.need_world_reset = self.unwrapped.need_world_reset = True

# TODO: we should be able to do this in a wrapper.
#   def set_world(self, world):
#       """Set self.next_world. At reset, set self.world = self.next_world."""
#       raise NotImplementedError('Implement set_world in domain-specific wrapper or environment.')

    def step(self, actions):
        """Step a single-agent environment."""
        obs, rews, done, info = super().step(actions)
        self.log_rewards(rews)
        self.n_step += 1

        # We'll take an additional step in the old world, then reset. Not the best but it works.
        done = done or self.need_world_reset

        # Not using this?
        # info.update({agent_k: {'world_key': self.world_key, 'agent_id': agent_k} for agent_k in obs})

        return obs, rews, done, info

    def log_rewards(self, rew):
        """Log rewards for each agent on each world."""
        if len(self.stats) > 0:
            self.stats[-1][1][(0, 0)] += rew

    def reset_stats(self):
        self.stats.append([self.world_key, {k: 0 for k in self.agent_ids}])

    def reset(self):
        """Reset the environment. This will also load the next world."""
        # print(f'Resetting world {self.world_key} at step {self.n_step}.')
        self.last_world_key = self.world_key

        # We are now resetting and loading the next world. So we switch this flag off.
        self.need_world_reset = False

        # Incrementing eval worlds to ensure each world is evaluated an equal number of times over training
        if self.evaluate:
            world_keys = list(self.worlds.keys())
            # print('eval\n')
            # FIXME: maybe inefficient to call index
            self.world_key = world_keys[(world_keys.index(self.world_key) + self.num_eval_envs) % len(self.worlds)]
            self.set_world(self.worlds[self.world_key])

        obs = super().reset()
        self.reset_stats()
        self.n_step = 0

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
            # print(f'get world stats at step {self.n_step} with max step {self.max_episode_steps}')
            assert self.max_episode_steps - 1 <= self.n_step <= self.max_episode_steps + 1

        qd_stats = []

        if not evaluate:
            assert len(self.stats) == 1
            # print(f'stats: {self.stats}')

        for i, (world_key, agent_rewards) in enumerate( self.stats):
            world_key_2, regret_loss = self.regret_losses[i] if len(self.regret_losses) > 0 else (None, None)
            
            # FIXME: Mismatched world keys between agent_rewards and regret_losses during evaluation. Ignoring so we can
            #  render all eval mazes
            if not self.evaluate \
                and world_key_2:  # i.e. we have a regret loss
                    assert world_key == world_key_2

            # Convert agent to policy rewards
            swarm_rewards = [[agent_rewards[(i, j)] for j in range(self.n_pop)] for i in range(self.n_policies)]

            # Return a mapping of world_key to a tuple of stats in a format that is compatible with qdpy
            # stats are of the form (world_key, qdpy_stats, policy_rewards)
            if quality_diversity:
                # Objective (negative fitness of protagonist population) and measures (antagonist population fitnesses)
                world_stats = {world_key: ((-np.mean(swarm_rewards[0]),), [np.mean(sr) for sr in swarm_rewards[1:]])}
            else:
                # Objective and placeholder measures
                if self.obj_fn_str == 'regret':
                    obj = self.objective_function(regret_loss)
                # If we have no objective function (i.e. evaluating on fixed worlds), objective is None.
                elif not self.objective_function:
                    obj = None
                # Most objectives are functions of agent reward.
                else:
                    obj = self.objective_function(swarm_rewards)
                world_stats = (world_key, ((obj,), [0, 0]))

            # Add per-policy stats
            world_stats = (*world_stats, [np.mean(sr) for sr in swarm_rewards])

            qd_stats.append(world_stats)

        self.stats = []
        self.regret_losses = []

        return qd_stats

    def validate_stats(self):
        if len(self.stats) == 0:

            # I guess we don't keep stats while evaluating...?
            if self.evaluate:
                pass

            # We have cleared stats, which means we need to assign new world and reset (adding a new entry to stats)
            if not self.evaluate:
                assert self.need_world_reset


class WorldEvolutionMultiAgentWrapper(WorldEvolutionWrapper, MultiAgentEnv):
    """Wrap underlying MultiAgentEnv's as MultiAgentEnv's so that rllibn will recognize them as such."""
    def __init__(self, env, env_config):
        WorldEvolutionWrapper.__init__(self, env, env_config)
        self.env = env
        # MultiAgentEnv.__init__(self)

    def clear_stats(self):
        self.stats.append((self.world_key, {agent_id: 0 for agent_id in self.agent_ids}))

    def step(self, actions):
        # We skip the step() function in WorldEvolutionWrapper and call *it's* parent's step() function instead.
        obs, rews, dones, infos = super(WorldEvolutionWrapper, self).step(actions)

        self.validate_stats()
        self.log_rewards(rews)
        dones['__all__'] = dones['__all__'] or self.need_world_reset
        self.n_step += 1

        return obs, rews, dones, infos

    def log_rewards(self, rews):
        """Log rewards for each agent on each world.
        
        We store rewards so that we can compute world fitness according to progress over duration of level. Might need
        separate rewards from multiple policies."""
        if len(self.stats) > 0:
            for k, v in rews.items():
                self.stats[-1][1][k] += v