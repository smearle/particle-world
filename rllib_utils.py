# Each policy can have a different configuration (including custom model).
import copy
import math
import os
from pdb import set_trace as TT

import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print

from env import ParticleGymRLlib, gen_policy, ParticleEvalEnv


def rllib_evaluate_worlds(trainer, worlds, idx_counter=None):
    idxs = np.random.permutation(list(worlds.keys()))
    workers = trainer.workers
    worlds = {k: np.array(world) for k, world in worlds.items()}
    world_id = 0
    fitnesses = {}
    all_stats = []

    while world_id < len(worlds):
        # print(len(fitnesses), len(worlds))
        #     envs = []
        #     for (wrk_i, worker) in enumerate([workers.local_worker()] + workers.remote_workers()):
        #         if wrk_i == 0:  # only ever 1 local worker
        #             envs += worker.foreach_env(lambda env: env)
        #         else:
        #             world_i = idxs[wrk_i % len(idxs)]
        #             TT()
        #             worker.foreach_env.remote(lambda env: env.set_world(worlds={world_i: worlds[world_i]}))
        #             envs += ray.get(worker.foreach_env.remote(lambda env: env))
        #     if len(worlds) < len(envs):
        #         idxs = idxs * len(envs)
        #
        #     [env.set_world(worlds={i: worlds[i]}) for i, env in zip(idxs[world_id:], envs)]
        #     world_id += len(envs)

        # When running parallel envs, is each env is to evaluate a separate world, map envs to worlds
        if idx_counter:
            envs = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env))
            envs = [e for we in envs for e in we]
            sub_idxs = idxs[world_id:min(world_id + len(envs), len(idxs))]
            idx_counter.set_idxs.remote(sub_idxs)
            hashes = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
            hashes = [h for wh in hashes for h in wh]
            idx_counter.set_hashes.remote(hashes)

            # FIXME: Sometimes hash-to-idx dict is not set by the above call?
            assert ray.get(idx_counter.scratch.remote())

        # Assign envs to worlds
        workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.set_world(worlds=worlds, idx_counter=idx_counter)))

        # Train/evaluate
        stats = trainer.train()
        # print(pretty_print(stats))
        all_stats.append(stats)

        # Collect stats
        new_fitnesses = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env.get_fitness()))
        new_fitnesses = [fit for worker_fits in new_fitnesses for fit in worker_fits]
        new_fits = {}
        [new_fits.update(nf) for nf in new_fitnesses]
        # if not len(new_fits) == len(sub_idxs):
        #     TT()
        fitnesses.update(new_fits)
        # if len(fitnesses) == world_id:
        #     TT()

        # If we've mapped envs to specific worlds, then we count the number of unique worlds evaluated (assuming worlds are
        # deterministic, so re-evaluation is redundant, and we may sometimes have redundant evaluations because we have too many envs).
        # Otherwise, we count the number of evaluations (e.g. when evaluating on a single fixed landscape).
        if idx_counter:
            world_id = len(fitnesses)
        else:
            world_id += len(new_fitnesses)

        # for (wrk_i, worker) in enumerate([workers.local_worker()] + workers.remote_workers()):
        #     world_i = idxs[wrk_i % len(idxs)]
        #     if wrk_i == 0:  # only ever 1 local worker
        #         new_fitnesses = worker.foreach_env(lambda env: env.get_fitness())
        #     else:
        #         new_fitnesses = worker.foreach_env.remote(lambda env: env.get_fitness())
        #     for nf in new_fitnesses:
        #         fitnesses.update(nf)

    return all_stats, fitnesses


def train_players(n_itr, n_policies, trainer, landscapes, save_dir, n_rllib_envs, idx_counter=None, logbook=None):
    trainer.workers.local_worker().set_policies_to_train([f'policy_{i}' for i in range(n_policies)])
    for i in range(n_policies):
        trainer.get_policy(f'policy_{i}').config["explore"] = True
    for i in range(n_itr):

        # Saving before training, so that we have a checkpoint of the model after the evolution phase, and before model
        # weights start changing.
        if i % 10 == 0:
            checkpoint = trainer.save(save_dir)
            with open(os.path.join(save_dir, 'model_checkpoint_path.txt'), 'w') as f:
                f.write(checkpoint)
            print("checkpoint saved at", checkpoint)
        curr_lands = np.array(landscapes)[np.random.choice(np.array(landscapes).shape[0], n_rllib_envs)]
        worlds = {i: l for i, l in enumerate(curr_lands)}
        all_stats, fitnesses = rllib_evaluate_worlds(trainer, worlds)
        assert len(all_stats) == 1
        rllib_stats = all_stats[0]
        # if logbook:
        #     logbook.record(iteration=curr_itr, meanAgentReward=rllib_stats["episode_reward_mean"],
        #                    maxAgentReward=rllib_stats["episode_reward_max"],
        #                    minAgentReward=rllib_stats["episode_reward_min"])
        keys = ['episode_reward_max', 'episode_reward_mean']
        # TODO: track stats over calls to train (shouldn't be necessary during evolution
        print('\n'.join([f'Training iteration {i}'] + [f'{k}: {rllib_stats[k]}' for k in keys]))
        if 'evaluation' in rllib_stats:
            print('\n'.join(['evaluation:'] +[f"  {k}: {rllib_stats['evaluation'][k]}" for k in keys]))
    for i in range(n_policies):
        trainer.get_policy(f'policy_{i}').config["explore"] = False
    trainer.workers.local_worker().set_policies_to_train([])


@ray.remote
class IdxCounter:
    ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
    responsible for providing unique indexes to parallel environments.'''
    def __init__(self):
        self.count = 0
        self.idxs = None

    def get(self, hsh):
        return self.hashes_to_idxs[hsh]

        # if self.idxs is None:
        #     Then we are doing inference and have set the idx directly
            #
            # return self.count
        #
        # idx = self.idxs[self.count % len(self.idxs)]
        # self.count += 1
        #
        # return idx

    def set(self, i):
        # For inference
        self.count = i

    def set_idxs(self, idxs):
        self.count = 0
        self.idxs = idxs

    def set_hashes(self, hashes):
        assert len(hashes) >= len(self.idxs)
        idxs = self.idxs

        # If we have more hashes than indices, map many-to-one
        if len(hashes) > len(idxs):
            n_repeats = math.ceil(len(hashes) / len(idxs))
            idxs = np.tile(idxs, n_repeats)
        self.hashes_to_idxs = {hsh: id for hsh, id in zip(hashes, idxs[:len(hashes)])}

    def scratch(self):
        return self.hashes_to_idxs



def init_particle_trainer(env, num_rllib_workers, n_rllib_envs, enjoy, save_dir):
    # env is currently a dummy environment that will not be used in actual training
    MODEL_CONFIG = copy.copy(MODEL_DEFAULTS)
    MODEL_CONFIG.update({
        # "use_lstm": True,
        # "fcnet_hiddens": [32, 32],
    })
    workers = 1 if num_rllib_workers == 0 or enjoy else num_rllib_workers

    trainer_config = {
        "multiagent": {
            "policies": {f'policy_{i}': gen_policy(i, env.observation_spaces[i], env.action_spaces[i], env.fovs[i])
                         for i, swarm in enumerate(env.swarms)},
            # the first tuple value is None -> uses default policy
            # "car1": (None, particle_obs_space, particle_act_space, {"gamma": 0.85}),
            # "car2": (None, particle_obs_space, particle_act_space, {"gamma": 0.99}),
            # "traffic_light": (None, tl_obs_space, tl_act_space, {}),
            "policy_mapping_fn":
                lambda agent_id, episode, worker, **kwargs: f'policy_{agent_id[0]}',
        },
        "model": MODEL_CONFIG,
        # {
        # "custom_model": RLlibNN,
        # },
        "env_config": {
            "swarm_cls": type(env.swarms[0]),  # Assuming env has only one type of swarm
            "width": env.width,
            "n_policies": len(env.swarms),
            "n_pop": env.swarms[0].n_pop,
            "max_steps": env.max_steps,
            # "pg_width": pg_width,
            "evaluate": False,
        },
        "num_gpus": 0,
        "num_workers": num_rllib_workers if not enjoy else 0,
        "num_envs_per_worker": math.ceil(n_rllib_envs / workers),
        "framework": "torch",
        "render_env": False if not enjoy else True,
        "evaluation_interval": 10 if not enjoy else 10,
        "evaluation_config": {
            "env_config": {
                "n_pop" : 1,
                "evaluate": True,
            },
            "evaluation_parallel_to_training": True,
            "evaluation_interval": 1,
            "evaluation_num_episodes": 10,
            # "render_env": True,
            "explore": False,
        },
        "logger_config": {
            "log_dir": save_dir,
        },
        # "explore": False,
        # "lr": 0.1,
        # "log_level": "INFO",
        # "record_env": True,
        "rollout_fragment_length": env.max_steps,
        "train_batch_size": env.max_steps * n_rllib_envs,
    }
    trainer = ppo.PPOTrainer(env=ParticleGymRLlib, config=trainer_config)
    return trainer


