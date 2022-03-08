# Each policy can have a different configuration (including custom model).
import copy
import math
import os
from pathlib import Path
from pdb import set_trace as TT
import shutil
from typing import Iterable

import numpy as np
import ray
import torch as th
from ray.rllib.agents.ppo import ppo
# from ray.tune.logger import pretty_print
from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import Logger
from timeit import default_timer as timer

from envs.env import ParticleGymRLlib, ParticleEvalEnv, eval_mazes
from model import FloodMemoryModel, OraclePolicy, CustomRNNModel, NCA
from paired_models.multigrid_models import MultigridRLlibNetwork
from rllib_utils.callbacks import RegretCallbacks
from rllib_utils.eval_worlds import rllib_evaluate_worlds
from utils import get_solution


def train_players(net_itr, play_phase_len, n_policies, n_pop, trainer, landscapes, save_dir, n_rllib_envs, n_sim_steps, 
                  idx_counter=None, logbook=None, quality_diversity=False, fixed_worlds=False):
    trainer.workers.local_worker().set_policies_to_train([f'policy_{i}' for i in range(n_policies)])
    # toggle_exploration(trainer, explore=True, n_policies=n_policies)
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
        replace = True if fixed_worlds else True
        # 3) Custom logger (see `MyPrintLogger` class above).
        logger_config = None if not fixed_worlds else {
            # Provide the class directly or via fully qualified class
            # path.
            "type": MyPrintLogger,
            # `config` keys:
            "prefix": "ABC",
            # Optional: Custom logdir (do not define this here
            # for using ~/ray_results/...).
            "logdir": save_dir,
        }
        if isinstance(landscapes, dict):
            world_keys = list(landscapes.keys())
        else:
            assert isinstance(landscapes, Iterable)
            world_keys = list(range(len(landscapes)))
        world_keys = np.random.choice(world_keys, n_rllib_envs, replace=replace)
        # curr_lands = np.array(landscapes)[world_keys]
        curr_lands = [landscapes[wid] for wid in world_keys]
        worlds = {i: l for i, l in enumerate(curr_lands)}

        # Get world heuristics on first iteration only, since they won't change after this inside training loop
        calc_world_heuristics = (i == 0)

        rl_stats, qd_stats, logbook_stats = rllib_evaluate_worlds(
            net_itr=net_itr, trainer=trainer, worlds=worlds, idx_counter=idx_counter, calc_agent_stats=True,
            start_time=start_time, evaluate_only=False, quality_diversity=quality_diversity, 
            calc_world_heuristics=calc_world_heuristics, fixed_worlds=fixed_worlds)
        recent_rewards[:-1] = recent_rewards[1:]
        recent_rewards[-1] = rl_stats['episode_reward_mean']

#       if i == 0:
#           mean_path_length = logbook_stats['meanPath']
#           max_mean_reward = (n_sim_steps - mean_path_length) * n_pop

        # End training if within a certain margin of optimal performance
#       done_training = recent_rewards[-1] >= 0.9 * max_mean_reward
        done_training = False

        # if play_phase_len == -1:
            # if i >= staleness_window:
                # running_std = np.std(recent_rewards)
                # print(f'Running reward std dev: {running_std}')
                # done_training = running_std < 1.0 or i >= 100
            # else:
                # done_training = False
        if not fixed_worlds:
            logbook.record(**logbook_stats)
            print(logbook.stream)
        i += 1
        if play_phase_len != -1:
            done_training = done_training or i >= play_phase_len
        net_itr += 1
    # toggle_exploration(trainer, explore=False, n_policies=n_policies)
    trainer.workers.local_worker().set_policies_to_train([])
    return net_itr


def init_particle_trainer(env, num_rllib_remote_workers, n_rllib_envs, evaluate, enjoy, render, save_dir, num_gpus, 
                          oracle_policy, fully_observable, idx_counter, model, env_config):
    """
    Initialize an RLlib trainer object for training neural nets to control (populations of) particles/players in the
    environment.
    #TODO: abstract this to support training generators as well.
    :param env: a dummy environment, created in the main script, from which we will draw environment variables. Will not
    be used in actual training.
    :param num_rllib_remote_workers: how many RLlib remote workers to use for training (1 core per worker). Just a local
    worker if 0.
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

    if oracle_policy:
        policies_dict = {f'policy_{i}': PolicySpec(policy_class=OraclePolicy, observation_space=env.observation_spaces[i],
                                                     action_space=env.action_spaces[i], config={})
                            for i, _ in enumerate(env.swarms)}
        model_config = {}
    else:
        policies_dict = {f'policy_{i}': gen_policy(i, env.observation_spaces[i], env.action_spaces[i], env.fovs[i])
                         for i, swarm in enumerate(env.swarms)}
        # Add the oracle (flood-fill convolutional model) for fast path-length computations. obs/act spaces are dummy
        policies_dict.update({
            'oracle': PolicySpec(policy_class=OraclePolicy, observation_space=env.observation_spaces[0], 
            action_space=env.action_spaces[0], config={})
            })

        model_config = copy.copy(MODEL_DEFAULTS)
        if fully_observable and model != 'flood':
            model_config.update({
            # Fully observable, non-translated map
            "conv_filters": [
                [64, [3, 3], 2], 
                [128, [3, 3], 2], 
                [256, [3, 3], 2]
            ],
            })
        else:
            if model is None:
                model_config.update({
                    "use_lstm": True,
                    "lstm_cell_size": 32,
                    # "fcnet_hiddens": [32, 32],  # Looks like this is unused when use_lstm is True
                    "conv_filters": [
                        [16, [5, 5], 1], 
                        [4, [3, 3], 1]],
                    # "post_fcnet_hiddens": [32, 32],
                })
            elif model == 'paired':
                # ModelCatalog.register_custom_model('paired', MultigridRLlibNetwork)
                ModelCatalog.register_custom_model('paired', CustomRNNModel)
                model_config = {'custom_model': 'paired'}
            elif model == 'flood':
                ModelCatalog.register_custom_model('flood', FloodMemoryModel)
                model_config = {'custom_model': 'flood', 'custom_model_config': {'player_chan': env.player_chan}}
            elif model == 'nca':
                ModelCatalog.register_custom_model('nca', NCA)
                model_config = {'custom_model': 'nca'}

            else:
                raise NotImplementedError

    num_workers = 1 if num_rllib_remote_workers == 0 or enjoy else num_rllib_remote_workers
    num_envs_per_worker = math.ceil(n_rllib_envs / num_workers) if not enjoy else 1
    num_eval_envs = num_envs_per_worker
    if enjoy:
        evaluation_num_episodes = num_eval_envs * 2
    # elif evaluate:
        # evaluation_num_episodes = math.ceil(len(eval_mazes) / num_eval_envs) * 10
    else:
        evaluation_num_episodes = math.ceil(len(eval_mazes) / num_eval_envs)

    trainer_config = {
        "callbacks": RegretCallbacks,
        "multiagent": {
            "policies": policies_dict,
            # the first tuple value is None -> uses default policy
            # "car1": (None, particle_obs_space, particle_act_space, {"gamma": 0.85}),
            # "car2": (None, particle_obs_space, particle_act_space, {"gamma": 0.99}),
            # "traffic_light": (None, tl_obs_space, tl_act_space, {}),
            "policy_mapping_fn":
                lambda agent_id, episode, worker, **kwargs: f'policy_{agent_id[0]}',
        },
        "model": model_config,
        # "model": {
            # "custom_model": "nav_net",
        # },
        # "model": {
            # "custom_model": "flood_model",
        # },
        # {
        # "custom_model": RLlibNN,
        # },
        "env_config": env_config,
        "num_gpus": num_gpus,
        "num_workers": num_rllib_remote_workers if not (enjoy or evaluate) else 0,
        "num_envs_per_worker": num_envs_per_worker,
        "framework": "torch",
        "render_env": render if not enjoy else True,

        # If enjoying, evaluation_interval is nonzero only to ensure eval workers get created for playback.
        "evaluation_interval": 10 if not enjoy else 10,

        "evaluation_num_workers": 0 if not (evaluate) else num_rllib_remote_workers,

        # FIXME: Hack workaround: during evaluation (after training), all but the first call to trainer.evaluate() will 
        # be preceded by calls to env.set_world(), which require an immediate reset to take effect. (And unlike 
        # trainer.train(), evaluate() waits until n episodes are completed, as opposed to proceeding for a fixed number 
        # of steps.)
        "evaluation_num_episodes": evaluation_num_episodes,
            # if not (evaluate or enjoy) else len(eval_mazes) + 1,

        "evaluation_config": {
            "env_config": {
                # "n_pop": 1,

                # If enjoying, then we look at generated levels instead of eval levels. (Because we user trainer.evaluate() when enjoying.)
                "evaluate": True if not enjoy else evaluate,
                "num_eval_envs": num_eval_envs,
            },
            "evaluation_parallel_to_training": True,
            "render_env": render,
            "explore": False if oracle_policy else True,
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
        "sgd_minibatch_size": env.max_steps * n_rllib_envs if (enjoy or evaluate) and render else 128,
    }
    trainer = ppo.PPOTrainer(env=type(env), config=trainer_config)

    # When enjoying, eval envs are set from the evolved world archive in rllib_eval_envs
    if not enjoy:
        # Set evaluation workers to eval_mazes. If more eval mazes then envs, the world_key of each env will be incremented
        # by len(eval_mazes) each reset.
        eval_workers = trainer.evaluation_workers
        hashes = eval_workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
        n_eval_envs = sum([e for we in hashes for e in we])
        idxs = list(eval_mazes.keys())
        idx_counter.set_idxs.remote(idxs)
        hashes = [h for wh in hashes for h in wh]
        idx_counter.set_hashes.remote(hashes, allow_one_to_many=True)
        # FIXME: Sometimes hash-to-idx dict is not set by the above call?
        assert ray.get(idx_counter.scratch.remote())
        # Assign envs to worlds
        eval_workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.set_worlds(worlds=eval_mazes, idx_counter=idx_counter)))

    n_policies = len(env.swarms)
    for i in range(n_policies):
        n_params = 0
        param_dict = trainer.get_weights()[f'policy_{i}']
        for v in param_dict.values():
            n_params += np.prod(v.shape)
        print(f'policy_{i} has {n_params} parameters.')
        print('model overview: \n', trainer.get_policy(f'policy_{i}').model)
    return trainer


def gen_policy(i, observation_space, action_space, fov):
    config = {
        "model": {
            "custom_model_config": {
                "fov": fov,
            }
        }
    }
    return PolicySpec(config=config, observation_space=observation_space, action_space=action_space)


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
    # print("checkpoint saved at", checkpoint)


def toggle_exploration(trainer, explore: bool, n_policies: int):
    for i in range(n_policies):
        trainer.get_policy(f'policy_{i}').config["explore"] = explore
        # Need to update each remote training worker as well (if they exist)
        trainer.workers.foreach_worker(lambda w: w.get_policy(f'policy_{i}').config.update({'explore': explore}))

class MyPrintLogger(Logger):
    """Logs results by simply printing out everything.
    """

    def _init(self):
        # Custom init function.
        print("Initializing...")
        # Setting up our log-line prefix.
        self.prefix = self.config.get("logger_config").get("prefix")

    def on_result(self, result: dict):
        # Define, what should happen on receiving a `result` (dict).
        print(f"{self.prefix}: {result}")

    def close(self):
        # Releases all resources used by this logger.
        print("Closing")

    def flush(self):
        # Flushing all possible disk writes to permanent storage.
        print("Flushing", flush=True)

