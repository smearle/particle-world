import copy
from functools import partial
from pdb import set_trace as TT

import gym
import numpy as np
import pygame
import ray
from ray.rllib import MultiAgentEnv
import torch as th

# from minerl.herobraine.env_spec import EnvSpec
# from envs.minecraft.touchstone import TouchStone
from evo.objectives import min_solvable_fitness, contrastive_fitness, regret_fitness, paired_fitness
from envs.minecraft.wrappers import MineRLWrapper
from generators.representations import render_landscape
from utils import discrete_to_onehot
from envs.minecraft.wrappers import MineRLWrapper


def make_env(env_config):
    cfg = env_config.get('cfg')
    environment_class = env_config.get('environment_class')

    if cfg.env_is_minerl:
    # if issubclass(environment_class, EnvSpec):
        from envs.minecraft.wrappers import MineRLWrapper
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
        self.world_gen_sequences = None
        self.world_queue = None
        cfg = env_cfg.get('cfg')

        self.evolve_player = self.training_world = self.evaluation_world = self.evo_eval_world = False

        # Is this world being used to train learning player agents?
        # self.training_world = env_cfg["training_world"]
        self.enjoy = cfg.enjoy

        # Target reward world should elicit if using min_solvable objective
        trg_rew = cfg.target_reward

        self.obj_fn_str = cfg.objective_function if not cfg.gen_adversarial_worlds else 'min_solvable'
        obj_func = globals()[self.obj_fn_str + '_fitness'] if self.obj_fn_str else None
        if obj_func == min_solvable_fitness:
            max_rew = env.max_reward
            obj_func = partial(obj_func, max_rew=max_rew, trg_rew=trg_rew)
        self.objective_function = obj_func

    def set_mode(self, mode: str):
        modes = ['evo_eval_world', 'training_world', 'evaluation_world', 'evolve_player']
        assert mode in modes
        [setattr(self, m, False) for m in modes]
        setattr(self, mode, True)

    def set_player_policies(self, players: dict):
        """Set the player agent for this environment."""
        self.player_keys = list(players.keys())
        self.players = [players[k] for k in self.player_keys]

        # TODO: support multiple players?
        assert len(self.player_keys) == 1

        self.need_world_reset = True


    def queue_worlds(self, worlds: dict, load_now: bool, idx_counter=None, next_n_pop=None, world_gen_sequences=None):
        """Assign a ``next_world`` to the environment, which will be loaded after the next step.
        
        We set flag ``need_world_reset`` to True, so that the next step will return done=True, and the environment will
        be reset, at which point, ``next_world`` will be loaded."""

        # Figure out which world to evaluate.
        # self.world_idx = 0
        if self.evolve_player:
            self.world_key_queue = copy.copy(list(worlds.keys()))
            self.world_queue = worlds
        # This handles training env evo eval
        elif idx_counter:
            self.world_key_queue = ray.get(idx_counter.get.remote(hash(self)))
            self.world_queue = {wk: worlds[wk] for wk in self.world_key_queue} if not self.evaluate else worlds
        else:
            # If... something else? shuffle world keys.
            self.world_key_queue = list(worlds.keys())
            self.world_queue = worlds

#       if self.training_world:
#           env_type = "Training"
#       elif self.evaluation_world:
#           env_type = "Evaluation"
#       elif self.evo_eval_world:
#           env_type = "EvoEval"
#       print(f"{env_type} env. Current world: {self.world_key}.\nQueued worlds: {self.world_key_queue}")
#       if self.training_world:

#       # Support changing number of players per policy between episodes (in case we want to evaluate policies 
#      #  deterministically, for example).   
#       if next_n_pop is not None:
#           self.next_n_pop = next_n_pop

        if world_gen_sequences is not None and len(world_gen_sequences) > 0:
            self.world_gen_sequences = {wk: world_gen_sequences[wk] for wk in self.world_key_queue}

        if load_now:
            self.need_world_reset = True

# TODO: we should be able to do this in a wrapper.
#   def set_world(self, world):
#       """Set self.next_world. At reset, set self.world = self.next_world."""
#       raise NotImplementedError('Implement set_world in domain-specific wrapper or environment.')

    def step(self, actions):
        """Step a single-agent environment."""
        # print(f'Stepping world {self.world_key}, step {self.n_step}.')
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
        # print(f"Resetting stats for world {self.world_key}.")
        new_ep_stats = {'world_key': self.world_key, 'n_steps': 0}
        new_ep_stats.update({f'agent_{k}_reward': 0 for k in self.agent_ids})
        self.stats.append(new_ep_stats)

    def reset(self):
        """Reset the environment. This will also load the next world."""
        # if self.evo_eval_world:
            # print(f'Resetting world {self.world_key} at step {self.n_step}.')
        self.has_rendered_world_gen = False
        self.last_world_key = self.world_key

        # We are now resetting and loading the next world. So we switch this flag off.
        self.need_world_reset = False

        # Cycle through each world once.
        if self.evo_eval_world:  # or self.evolve_player:
            if self.world_key_queue:
                self.world_key = self.world_key_queue[0]
                # Remove first key from the queue.
                self.world_key_queue = self.world_key_queue[1:] if self.world_key_queue else []
                self.set_world(self.world_queue[self.world_key])
            # Not changing world_key to None here is a workaround to allow setting regret loss after reset (which is not avoidable while calling batch. sample() ...?)

        # Increment eval worlds to ensure each world is evaluated an equal number of times over training
        elif self.evaluate:
            # 0th world key is that assigned by the global idx_counter. This ensures no two eval envs will be 
            # evaluating the same world if it can be avoided. 
            self.last_world_key = self.world_key_queue[0] if self.last_world_key is None else self.last_world_key

            # From here on, the next world we evaluate is num_eval_envs away from the one we just evaluated.
            world_keys = list(self.world_queue.keys())
            # FIXME: maybe inefficient to call index
            self.world_key = world_keys[(world_keys.index(self.last_world_key) + self.num_eval_envs) % len(self.world_queue)]
            self.set_world(self.world_queue[self.world_key])

        # Randomly select a world from the training set.
        # elif self.training_world:
        else:
            self.world_key = np.random.choice(self.world_key_queue)
            self.set_world(self.world_queue[self.world_key])
        
        # else:
            # raise Exception

        # self.next_world = None if not self.enjoy and not self.world_queue else self.world_queue[self.world_key_queue[0]]
        # self.world_queue = self.world_queue[1:] if self.world_queue else []

        obs = super().reset()
        # self.reset_stats()
        self.n_step = 0

        return obs

    def set_regret_loss(self, losses):
        # If this environment is handled by a training worker, then the we care about `last_world_key`. Something about
        # resetting behavior during evaluation vs. training.
        if self.evaluate:
            wk = self.last_world_key
        else:
            wk = self.world_key
        if wk not in losses:
            # print(f"World key: {self.world_key}. Last world key: {self.last_world_key}. Loss keys: {losses.keys()}")
            # assert self.evaluate, f"No regret loss for world {wk}.Samples should cover all simultaneously "\
            #     "evaluated worlds except during evaluation."
            return
        loss = losses[wk]
        self.regret_losses.append((wk, loss))

    def get_world_stats(self, quality_diversity=False):
        """
        Return the objective score (and diversity measures) achieved by the world after an episode of simulation. Note
        that this only returns the metrics corresponding to the latest episode.
        """
        # On the first iteration, the episode runs for max_steps steps. On subsequent calls to rllib's trainer.train(), the
        # reset() call occurs on the first step (resulting in max_steps - 1).

        world_stats = []
        next_stats = []

        for i, stats_dict in enumerate(self.stats):

            # We will simulate on this world on future steps, so keep this empty stats dict around.
            if stats_dict["n_steps"] == 0:
            #     next_stats.append(stats_dict)
            #     TT()

                continue

            # print(f"self.regret_losses: {self.regret_losses}\nself.stats: {self.stats}")
            world_key = stats_dict['world_key']
            # world_key_2, regret_loss = self.regret_losses[i] if len(self.regret_losses) > 0 else (None, None)
            
            # FIXME: Mismatched world keys between agent_rewards and regret_losses during evaluation. Ignoring so we can
            #  render all eval mazes. This is fixed yeah?
            # if not self.evaluate and world_key_2:  # i.e. we have a regret loss
                # assert world_key == world_key_2

            # Convert agent to policy rewards. Get a list of lists, where outer list is per-policy, and inner list is 
            # per-agent reward.
            swarm_rewards = [[stats_dict[f'agent_{(i, j)}_reward'] for j in range(self.n_pop)] \
                for i in range(self.n_policies)]

            # Return a mapping of world_key to a tuple of stats in a format that is compatible with qdpy
            # stats are of the form (world_key, qdpy_stats, policy_rewards)

            # Objective and placeholder measures
            if self.obj_fn_str == 'regret':
                # obj = self.objective_function(regret_loss)

                #Placeholder regret objective as we move this logic outside the environment.
                obj = None

            # If we have no objective function (i.e. evaluating on fixed worlds), objective is None.
            elif not self.objective_function:
                obj = 0
            # Most objectives are functions of agent reward.
            else:
                obj = self.objective_function(swarm_rewards)

            if quality_diversity:
                # Objective (negative fitness of protagonist population) and measures (antagonist population fitnesses)
                # obj = min_solvable_fitness(swarm_rewards[0:1], max_rew=self.max_reward)

                # Default measures are the mean rewards of policies 2 and 3.
                measures = [np.mean(sr) for sr in swarm_rewards[1:]]

            else:
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

#           # We have cleared stats, which means we need to assign new world and reset (adding a new entry to stats)
#           if not self.evaluate:
#               assert self.need_world_reset

    def render(self, mode='human', pg_width=None, render_player=True):
        pg_width = self.pg_width if pg_width is None else pg_width
        if mode == 'human' and not self.screen:
            # A redundant render at first just to initialize the pygame screen so we can render world-generation on it.
            super().render(enforce_constraints=False, pg_width=pg_width, render_player=render_player)

        # Sometimes world_key is not in world_gen_sequences... presumably due to forced reset for world loading?
        if self.world_gen_sequences is not None and not self.has_rendered_world_gen and self.world_key in self.world_gen_sequences:
            # print(f"Render world {self.world_key} on step {self.n_step}.")
            # Render the sequence of worlds that were generated in the generation process.
            for w in self.world_gen_sequences[self.world_key]:
                world = np.ones((w.shape[0]+2, w.shape[1]+2), dtype=int) * self.wall_chan
                world[1:-1, 1:-1] = w
                # render_landscape(self.screen, -1 * world[1] + 1)
                sidxs, gidxs = np.argwhere(world == self.start_chan), np.argwhere(world == self.goal_chan)
                world = discrete_to_onehot(world, n_chan=self.n_chan)
                self.render_level(world, sidxs, gidxs, pg_scale=pg_width/self.width, pg_delay=10, 
                                mode=mode, render_player=False, enforce_constraints=False)
                
            self.has_rendered_world_gen = True
    
        # Render the final world and players.
        return super().render(mode=mode, enforce_constraints=True, pg_width=pg_width, render_player=render_player)


class WorldEvolutionMultiAgentWrapper(WorldEvolutionWrapper, MultiAgentEnv):
    """Wrap underlying MultiAgentEnv's as MultiAgentEnv's so that rllibn will recognize them as such."""
    def __init__(self, env, env_config):
        WorldEvolutionWrapper.__init__(self, env, env_config)
        self.env = env
        self.evolve_player = env_config['cfg'].evolve_players
        # MultiAgentEnv.__init__(self)

    def step(self, actions):
        # print(f"Step env with world key {self.world_key}, step {self.n_step}.")
        
        # We skip the step() function in WorldEvolutionWrapper and call *it's* parent's step() function instead.
        obs, rews, dones, infos = super(WorldEvolutionWrapper, self).step(actions)

        self.validate_stats()
        self.log_stats(rews=rews)
#       if dones['__all__']:
#           print(f'world {self.world_key} is done.')
        infos.update({agent_k: {'world_key': self.world_key, 'agent_id': agent_k} for agent_k in obs})
        dones = self.get_dones(dones)
        self.n_step += 1

        return obs, rews, dones, infos

    def get_dones(self, dones):
        assert '__all__' not in dones
        dones['__all__'] = self.n_step > 0 and self.n_step == self.max_episode_steps or\
             self.need_world_reset
        if not self.evo_eval_world:
            dones['__all__'] = dones['__all__'] or np.all(list(dones.values()))
        return dones

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
            pass
            # This happens on evaluate now for some reason. But everything seems fine otherwise.
            # assert not self.evaluate
            # assert self.max_episode_steps - 1 <= self.n_step <= self.max_episode_steps + 1

    def _preprocess_obs(self, obs):
        policy_batch_obs = {i: th.zeros((self.n_pop, *self.observation_spaces[i].shape)) for i in range(len(self.players))}
        for (i, j) in obs:
            policy_batch_obs[i][j] = th.Tensor(obs[(i, j)])

        return policy_batch_obs

    def simulate(self, render_env=False):
        assert self.evolve_player
        ep_rews = None
        # world_rews = {}
        player_rew = 0
        
        for _ in range(len(self.world_key_queue)):

            # Load up a new world and simulate player behavior in it.
            obs = self.reset()
            self.reset_stats()
            # world_rews[self.world_key] = {k: 0 for k in self.player_keys}
            dones = {"__all__": False}
            net_rews = {k: 0 for k in obs}
            [self.players[i].reset() for i in range(len(self.players))]

            while dones["__all__"] == False:
                batch_obs = self._preprocess_obs(obs)
                actions = {i: self.players[i].get_actions(batch_obs[i]) for i in range(len(self.players))}
                actions = {(i, j): actions[i][j] for i in actions for j in range(len(actions[i])) if (i, j) not in self.dead}
                obs, rews, dones, infos = self.step(actions)
                player_rew += sum(rews.values())
                net_rews = {k: net_rews[k] + v for k, v in rews.items()}
                if render_env:
                    self.render()

                # for (i, j) in net_rews:
                    # world_rews[self.world_key][i] += net_rews[(i, j)]
        
        world_stats = self.get_world_stats()
        # print(f"World {self.world_key} stats: {world_stats}")
        
        return {self.player_keys[0]: player_rew / self.n_pop}, world_stats

