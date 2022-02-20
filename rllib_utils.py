# Each policy can have a different configuration (including custom model).
import copy
import math
import os
import shutil
from pathlib import Path
from pdb import set_trace as TT

import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from timeit import default_timer as timer

from env import ParticleGymRLlib, gen_policy, ParticleEvalEnv, eval_mazes
from model import FloodModel
from utils import get_solution

ModelCatalog.register_custom_model('flood_model', FloodModel)

def rllib_evaluate_worlds(trainer, worlds, idx_counter=None, evaluate_only=False, quality_diversity=False, n_trials=1):
    """
    Simulate play on a set of worlds, returning statistics corresponding to players/generators, using rllib's
    train/evaluate functions.
    :param trainer:
    :param worlds:
    :param idx_counter:
    :param evaluate_only: If True, we are not training, just evaluating some trained players/generators. (Normally,
    during training, we also evaluate at regular intervals. This won't happen here.) If True, we do not collect stats
    about generator fitness.
    :param quality_diversity: Whether we are running a QD experiment, in which case we'll return measures corresponding
    to fitnesses of distinct populations, and an objective corresponding to fitness of an additional "protagonist"
    population. Otherwise, return placeholder measures, and an objective corresponding to a contrastive measure of
    population fitnesses.
    :return:
    """
    idxs = np.random.permutation(list(worlds.keys()))
    if evaluate_only:
        workers = trainer.evaluation_workers
    else:
        workers = trainer.workers
    worlds = {k: np.array(world) for k, world in worlds.items()}
    # fitnesses = {k: [] for k in worlds}
    all_stats = []

    # Train/evaluate on all worlds n_trials many times each
    # for i in range(n_trials):
    trial_fitnesses = {}
    world_id = 0

    # Train/evaluate until we have simulated in all worlds
    while world_id < len(worlds):

        # When running parallel envs, if each env is to evaluate a separate world, map envs to worlds
        if idx_counter:
            n_envs = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: 1))
            n_envs = sum([e for we in n_envs for e in we])
            sub_idxs = idxs[world_id:min(world_id + n_envs, len(idxs))]
            idx_counter.set_idxs.remote(sub_idxs)
            hashes = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
            hashes = [h for wh in hashes for h in wh]
            idx_counter.set_hashes.remote(hashes)

            # FIXME: Sometimes hash-to-idx dict is not set by the above call?
            assert ray.get(idx_counter.scratch.remote())

        # Assign envs to worlds
        workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.set_worlds(worlds=worlds, idx_counter=idx_counter)))

        # Train/evaluate
        if evaluate_only:
            stats = trainer.evaluate()
        else:
            stats = trainer.train()
        # print(pretty_print(stats))
        all_stats.append(stats)

        # Collect stats for generator
        new_fitnesses = workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.get_world_stats(evaluate=evaluate_only, quality_diversity=quality_diversity)))
        new_fitnesses = [fit for worker_fits in new_fitnesses for fit in worker_fits]
        new_fits = {}
        [new_fits.update(nf) for nf in new_fitnesses]
        trial_fitnesses.update(new_fits)

        # If we've mapped envs to specific worlds, then we count the number of unique worlds evaluated (assuming worlds are
        # deterministic, so re-evaluation is redundant, and we may sometimes have redundant evaluations because we have too many envs).
        # Otherwise, we count the number of evaluations (e.g. when evaluating on a single fixed world).
        if idx_counter:
            world_id = len(trial_fitnesses)
        else:
            world_id += len(new_fitnesses)

        # [fitnesses[k].append(v) for k, v in trial_fitnesses.items()]
    # fitnesses = {k: ([np.mean([vi[0][fi] for vi in v]) for fi in range(len(v[0][0]))],
    #         [np.mean([vi[1][mi] for vi in v]) for mi in range(len(v[0][1]))]) for k, v in fitnesses.items()}

    return all_stats, trial_fitnesses


def train_players(play_phase_len, n_policies, n_pop, trainer, landscapes, save_dir, n_rllib_envs, n_sim_steps, 
                  idx_counter=None, logbook=None):
    trainer.workers.local_worker().set_policies_to_train([f'policy_{i}' for i in range(n_policies)])
    toggle_exploration(trainer, explore=True, n_policies=n_policies)
    i = 0
    staleness_window = 10
    done_training = False
    recent_rewards = np.empty(staleness_window)
    while not done_training:
        start_time = timer()
        # Saving before training, so that we have a checkpoint of the model after the evolution phase, and before model
        # weights start changing.
        if i % 10 == 0:
            rllib_save_model(trainer, save_dir)
        world_idxs = np.random.choice(np.array(landscapes).shape[0], n_rllib_envs, replace=False)
        curr_lands = np.array(landscapes)[world_idxs]
        worlds = {i: l for i, l in enumerate(curr_lands)}
        all_stats, fitnesses = rllib_evaluate_worlds(trainer, worlds, idx_counter=idx_counter)
        if i == 0:
            mean_path_length = get_mean_env_path_length(trainer)
            max_mean_reward = (n_sim_steps - mean_path_length) * n_pop
            print(f'Mean path length: {mean_path_length}\nMax mean reward: {max_mean_reward}')
        assert len(all_stats) == 1
        rllib_stats = all_stats[0]
        # if logbook:
        #     logbook.record(iteration=curr_itr, meanAgentReward=rllib_stats["episode_reward_mean"],
        #                    maxAgentReward=rllib_stats["episode_reward_max"],
        #                    minAgentReward=rllib_stats["episode_reward_min"])
        keys = ['episode_reward_max', 'episode_reward_mean']
        # TODO: track stats over calls to train (shouldn't be necessary during evolution)
        print('\n'.join([f'Training iteration {i}, time elapsed: {timer() - start_time}'] + [f'{k}: {rllib_stats[k]}' for k in keys]))
        if 'evaluation' in rllib_stats:
            print('\n'.join(['evaluation:'] + [f"  {k}: {rllib_stats['evaluation'][k]}" for k in keys]))
        recent_rewards[:-1] = recent_rewards[1:]
        recent_rewards[-1] = rllib_stats['episode_reward_mean']

        # End training if within a certain margin of optimal performance
        done_training = recent_rewards[-1] >= 0.9 * max_mean_reward

        if play_phase_len != -1:
            done_training = done_training or i >= play_phase_len
        # if play_phase_len == -1:
            # if i >= staleness_window:
                # running_std = np.std(recent_rewards)
                # print(f'Running reward std dev: {running_std}')
                # done_training = running_std < 1.0 or i >= 100
            # else:
                # done_training = False
        i += 1
    # toggle_exploration(trainer, explore=False, n_policies=n_policies)
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


def init_particle_trainer(env, num_rllib_workers, n_rllib_envs, evaluate, enjoy, render, save_dir, num_gpus):
    """
    Initialize an RLlib trainer object for training neural nets to control (populations of) particles/players in the
    environment.
    #TODO: abstract this to support training generators as well.
    :param env: a dummy environment, created in the main script, from which we will draw environment variables. Will not
    be used in actual training.
    :param num_rllib_workers: how many RLlib workers to use for training (1 core per worker)
    :param n_rllib_envs: how many environments to use for training. This determines how many worlds/generators can be
    evaluated with each call to train. When evolving worlds, the determines the batch size.
    :param evaluate: if True, then we are evaluating only (no training), and will launch n_rllib_workers-many
    evaluation workers.
    :param enjoy: if True, then the trainer is being initialized only to render and observe trained particles, so we
    will set other rllib parameters accordingly.
    :param render: whether to render an environment during training.
    :param save_dir: The directory in which experiment logs and checkpoints are to be saved.
    :param num_gpus: How many GPUs to use for training.
    :return: An rllib PPOTrainer object
    """
    model_config = copy.copy(MODEL_DEFAULTS)
    model_config.update({
        "use_lstm": True,
        "lstm_cell_size": 32,
        "fcnet_hiddens": [32, 32],  # Looks like this is unused because of LSTM?

        # Arranging the second convolution to leave us with an activation of size just >= 32 when flattened (3*3*4=36)
        "conv_filters": [[16, [5, 5], 1], [4, [3, 3], 1]],
        # "post_fcnet_hiddens": [32, 4],
    })
    workers = 1 if num_rllib_workers == 0 or enjoy else num_rllib_workers
    num_envs_per_worker = math.ceil(n_rllib_envs / workers) if not enjoy else 1

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
        # "model": model_config,
        "model": {
            "custom_model": "flood_model",
        },
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
            "objective_function": env.obj_fn_str,
        },
        "num_gpus": num_gpus,
        "num_workers": num_rllib_workers if not (enjoy or evaluate) else 0,
        "num_envs_per_worker": num_envs_per_worker,
        "framework": "torch",
        "render_env": render if not enjoy else True,

        # If enjoying, evaluation_interval is nonzero only to ensure eval workers get created for playback.
        "evaluation_interval": 100 if not enjoy else 10,

        "evaluation_num_workers": 0 if not (evaluate) else num_rllib_workers,
        # FIXME: Hack workaround: during evaluation (after training), all but the first call to trainer.evaluate() will be preceded by calls to env.set_world(), which require an immediate reset to take effect. (And unlike trainer.train(), evaluate() waits until n episodes are completed, as opposed to proceeding for a fixed number of steps.)
        "evaluation_num_episodes": len(eval_mazes * num_envs_per_worker) if not (evaluate or enjoy) else len(eval_mazes) + 1,
        "evaluation_config": {
            "env_config": {
                # "n_pop": 1,

                # If enjoying, then we look at generated levels instead of eval levels. (Because we user trainer.evaluate() when enjoying.)
                "evaluate": True if not enjoy else evaluate,
            },
            "evaluation_parallel_to_training": True,
            "render_env": render,
            "explore": True if enjoy or evaluate else True,
        },
        "logger_config": {
            "log_dir": save_dir,
        },
        # "lr": 0.1,
        # "log_level": "INFO",
        # "record_env": True,

        "rollout_fragment_length": env.max_steps,
        # This guarantees that each call to train() simulates 1 episode in each environment/world.
        "train_batch_size": env.max_steps * n_rllib_envs,
    }
    trainer = ppo.PPOTrainer(env=type(env), config=trainer_config)
    n_policies = len(env.swarms)
    for i in range(n_policies):
        n_params = 0
        param_dict = trainer.get_weights()[f'policy_{i}']
        for v in param_dict.values():
            n_params += np.prod(v.shape)
        print(f'policy_{i} has {n_params} parameters.')
        print('model overview: \n', trainer.get_policy(f'policy_{i}').model)
    return trainer


def rllib_save_model(trainer, save_dir):
    checkpoint = trainer.save(save_dir)
    # Delete previous checkpoint
    ckp_path_file = os.path.join(save_dir, 'model_checkpoint_path.txt')
    if os.path.isfile(ckp_path_file):
        with open(ckp_path_file, 'r') as f:
            ckp_path = Path(f.read())
            ckp_path = ckp_path.parent.absolute()
            if os.path.isdir(ckp_path):
                shutil.rmtree(ckp_path)
    # Record latest checkpoint path in case of re-loading
    with open(ckp_path_file, 'w') as f:
        f.write(checkpoint)
    print("checkpoint saved at", checkpoint)

def toggle_exploration(trainer, explore: bool, n_policies: int):
    for i in range(n_policies):
        trainer.get_policy(f'policy_{i}').config["explore"] = explore
        # Need to update each remote training worker as well (if they exist)
        trainer.workers.foreach_worker(lambda w: w.get_policy(f'policy_{i}').config.update({'explore': explore}))


def get_mean_env_path_length(trainer, n_policies: int):
    path_lengths = trainer.workers.foreach_worker(
        lambda w: w.foreach_env(lambda e: len(get_solution(e.world_flat))))
    path_lengths = [p for worker_paths in path_lengths for p in worker_paths]
    mean_path_length = np.mean(path_lengths)
    return mean_path_length
