import copy
from functools import partial
from pdb import set_trace as TT

import gym
import numpy as np
import ray
from ray.rllib import MultiAgentEnv

# from minerl.herobraine.env_spec import EnvSpec
# from envs.minecraft.touchstone import TouchStone
from envs.maze.swarm import min_solvable_fitness, contrastive_fitness, regret_fitness
from envs.minecraft.wrappers import MineRLWrapper


def make_env(env_config):
    cfg = env_config.get('cfg')
    environment_class = env_config.get('environment_class')

    if cfg.env_is_minerl:
    # if issubclass(environment_class, EnvSpec):
        env = gym.make(environment_class)
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
        self.world_queue = None
        cfg = env_cfg.get('cfg')
        self.enjoy = cfg.enjoy

        # Target reward world should elicit if using min_solvable objective
        trg_rew = cfg.target_reward

        self.obj_fn_str = cfg.objective_function if not cfg.gen_adversarial_worlds else 'min_solvable'
        obj_func = globals()[self.obj_fn_str + '_fitness'] if self.obj_fn_str else None
        if obj_func == min_solvable_fitness:

            # TODO: get this directly from the environment.
            # The maximum reward (specific to maze environment)
            max_rew = env.max_reward

            obj_func = partial(obj_func, max_rew=max_rew, trg_rew=trg_rew)
        self.objective_function = obj_func

    def queue_worlds(self, worlds: dict, idx_counter=None, next_n_pop=None, world_gen_sequences=None):
        """Assign a ``next_world`` to the environment, which will be loaded after the next step.
        
        We set flag ``need_world_reset`` to True, so that the next step will return done=True, and the environment will
        be reset, at which point, ``next_world`` will be loaded."""

        # Figure out which world to evaluate.
        # self.world_idx = 0
        if idx_counter:
            self.world_key_queue = ray.get(idx_counter.get.remote(hash(self)))
        else:
            # Do we ever use this?
            self.world_key = np.random.choice(list(worlds.keys()))

        # FIXME: hack
        # self.unwrapped.world_key = self.world_key

        # Assign this world to myself.
        self.world_queue = {wk: worlds[wk] for wk in self.world_key_queue} if not self.evaluate else worlds
        # self.set_world(worlds[self.world_key])
        # self.worlds = self.unwrapped.worlds = worlds

#       # Support changing number of players per policy between episodes (in case we want to evaluate policies 
#      #  deterministically, for example).   
#       if next_n_pop is not None:
#           self.next_n_pop = next_n_pop

        if world_gen_sequences is not None and len(world_gen_sequences) > 0:
            self.world_gen_sequence = world_gen_sequences[self.world_key]

        self.need_world_reset = True

# TODO: we should be able to do this in a wrapper.
#   def set_world(self, world):
#       """Set self.next_world. At reset, set self.world = self.next_world."""
#       raise NotImplementedError('Implement set_world in domain-specific wrapper or environment.')

    def step(self, actions):
        """Step a single-agent environment."""
        print(f'Stepping world {self.world_key}, step {self.n_step}.')
        obs, rew, done, info = super().step(actions)
        self.log_stats(rew=rew)
        info.update({'world_key': self.world_key})
        self.n_step += 1

        # We'll take an additional step in the old world, then reset. Not the best but it works.
        done = done or self.need_world_reset

        return obs, rew, done, info

    def log_stats(self, rew):
        """Log rewards for each agent on each world in the ``self.stats`` attribute."""
        # (0, 0) is treated as the first agent's id in single-agent environments.
        self.stats[-1][f'agent_{(0, 0)}_reward'] += rew
        self.stats[-1]['n_steps'] = self.n_step

    def reset_stats(self):
        new_ep_stats = {'world_key': self.world_key, 'n_steps': 0}
        new_ep_stats.update({f'agent_{k}_reward': 0 for k in self.agent_ids})
        self.stats.append(new_ep_stats)

    def reset(self):
        """Reset the environment. This will also load the next world."""
        if self.evaluate:
            # print(f'Resetting world {self.world_key} at step {self.n_step}.')
            pass
        self.last_world_key = self.world_key

        # We are now resetting and loading the next world. So we switch this flag off.
        self.need_world_reset = False

        # Incrementing eval worlds to ensure each world is evaluated an equal number of times over training
        if self.evaluate:
            # 0th world key is that assigned by the global idx_counter. This ensures no two eval envs will be 
            # evaluating the same world if it can be avoided. 
            self.last_world_key = self.world_key_queue[0] if self.last_world_key is None else self.last_world_key

            # From here on, the next world we evaluate is num_eval_envs away from the one we just evaluated.
            world_keys = list(self.world_queue.keys())
            # FIXME: maybe inefficient to call index
            self.world_key = world_keys[(world_keys.index(self.last_world_key) + self.num_eval_envs) % len(self.world_queue)]
            self.set_world(self.world_queue[self.world_key])
        else:
            self.world_key = self.world_key_queue[0] if self.world_key_queue else None
            self.world_key_queue = self.world_key_queue[1:] if self.world_key_queue else []
            if self.world_key:
                self.set_world(self.world_queue.pop(self.world_key))

        # self.next_world = None if not self.enjoy and not self.world_queue else self.world_queue[self.world_key_queue[0]]
        # self.world_queue = self.world_queue[1:] if self.world_queue else []

        obs = super().reset()
        self.reset_stats()
        self.n_step = 0

        return obs

    def set_regret_loss(self, losses):
        # If this environment is handled by a training worker, then the we care about `last_world_key`. Something about
        # resetting behavior during evaluation vs. training.
        if self.world_key not in losses:
            assert self.evaluate, f"No regret loss for world {self.world_key}.Samples should cover all simultaneously "\
                "evaluated worlds except during evaluation."
            return
        loss = losses[self.world_key]
        self.regret_losses.append((self.world_key, loss))

    def get_world_stats(self, evaluate=False, quality_diversity=False):
        """
        Return the fitness (and behavior characteristics) achieved by the world after an episode of simulation. Note
        that this only returns the fitness of the latest episode.
        """
        # On the first iteration, the episode runs for max_steps steps. On subsequent calls to rllib's trainer.train(), the
        # reset() call occurs on the first step (resulting in max_steps - 1).
        if not evaluate:
            # print(f'Get world {self.world_key} stats at step {self.n_step} with max step {self.max_episode_steps}')
            # assert self.max_episode_steps - 1 <= self.n_step <= self.max_episode_steps + 1
            pass

        world_stats = []
        next_stats = []

        if not evaluate:
            # assert len(self.stats) == cfg.n_eps_on_train // cfg.n_eps_per_world 
            # print(f'stats: {self.stats}')
            pass

        for i, stats_dict in enumerate(self.stats):

            # We will simulate on this world on future steps, so keep this empty stats dict around.
            if stats_dict["n_steps"] == 0:
                next_stats.append(stats_dict)
                continue

            world_key = stats_dict['world_key']
            world_key_2, regret_loss = self.regret_losses[i] if len(self.regret_losses) > 0 else (None, None)
            
            # FIXME: Mismatched world keys between agent_rewards and regret_losses during evaluation. Ignoring so we can
            #  render all eval mazes. This is fixed yeah?
            if not self.evaluate and world_key_2:  # i.e. we have a regret loss
                assert world_key == world_key_2

            # Convert agent to policy rewards. Get a list of lists, where outer list is per-policy, and inner list is 
            # per-agent reward.
            swarm_rewards = [[stats_dict[f'agent_{(i, j)}_reward'] for j in range(self.n_pop)] \
                for i in range(self.n_policies)]

            # Return a mapping of world_key to a tuple of stats in a format that is compatible with qdpy
            # stats are of the form (world_key, qdpy_stats, policy_rewards)
            if quality_diversity:
                # Objective (negative fitness of protagonist population) and measures (antagonist population fitnesses)
                obj = min_solvable_fitness(swarm_rewards[0:1], max_rew=self.max_reward)
                measures = [np.mean(sr) for sr in swarm_rewards[1:]]

            else:
                # Objective and placeholder measures
                if self.obj_fn_str == 'regret':
                    obj = self.objective_function(regret_loss)
                # If we have no objective function (i.e. evaluating on fixed worlds), objective is None.
                elif not self.objective_function:
                    obj = 0
                # Most objectives are functions of agent reward.
                else:
                    obj = self.objective_function(swarm_rewards)

                # Placeholder measures
                measures = [0, 0]

            # Format things for qdpy consumption.
            qd_stats_i = ((obj,), measures)

            # Add some additional per-policy stats (just mean reward for now).
            world_stats_i = {"world_key": world_key, "qd_stats": qd_stats_i, "n_steps": stats_dict["n_steps"]}  #, [np.mean(sr) for sr in swarm_rewards])
            world_stats_i.update({f'policy_{i}': {} for i in range(len(swarm_rewards))})
            [world_stats_i[f'policy_{i}'].update({'mean_reward': np.mean(sr)}) for i, sr in enumerate(swarm_rewards)]
            [world_stats_i[f'policy_{i}'].update({'pct_win': np.sum(np.array(sr) > 0) / len(sr)}) for i, sr in enumerate(swarm_rewards)]
            # TODO: log time-to-win?
            world_stats.append(world_stats_i)

        self.stats = next_stats
        self.regret_losses = []

        return world_stats

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

    def step(self, actions):
        # print(f"Step env with world key {self.world_key}, step {self.n_step}.")
        
        # We skip the step() function in WorldEvolutionWrapper and call *it's* parent's step() function instead.
        obs, rews, dones, infos = super(WorldEvolutionWrapper, self).step(actions)

        self.validate_stats()
        self.log_stats(rews=rews)
        dones['__all__'] = dones['__all__'] or self.need_world_reset
#       if dones['__all__']:
#           print(f'world {self.world_key} is done.')
        infos.update({agent_k: {'world_key': self.world_key, 'agent_id': agent_k} for agent_k in obs})
        self.n_step += 1

        return obs, rews, dones, infos

    def log_stats(self, rews):
        """Log rewards for each agent on each world.
        
        We store rewards so that we can compute world fitness according to progress over duration of level. Might need
        separate rewards from multiple policies."""
        # On training, we don't write stats for the last, "hangover" step from the previous world before loading the 
        # next world up for evaluation. The previous world is from the previous batch of evolved world candidates, so 
        # we don't want to return qd stats on it.
        if len(self.stats) > 0:
            for k, v in rews.items():
                self.stats[-1][f'agent_{k}_reward'] += v
            self.stats[-1]['n_steps'] = self.n_step
        else:
            assert not self.evaluate
            # assert self.max_episode_steps - 1 <= self.n_step <= self.max_episode_steps + 1