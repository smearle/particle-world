# Each policy can have a different configuration (including custom model).
import copy
import math
import os
import shutil
from pathlib import Path
from pdb import set_trace as TT

import numpy as np
import ray
import torch as th
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from timeit import default_timer as timer

from env import ParticleGymRLlib, ParticleEvalEnv, eval_mazes
from model import OraclePolicy
from utils import get_solution


def rllib_evaluate_worlds(trainer, worlds, start_time=None, net_itr=None, idx_counter=None, evaluate_only=False, 
    quality_diversity=False, oracle_policy=False, calc_world_stats=False, calc_agent_stats=False):
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
    if start_time is None:
        start_time = timer()
    idxs = np.random.permutation(list(worlds.keys()))
    if evaluate_only:
        workers = trainer.evaluation_workers
    else:
        workers = trainer.workers
    if not isinstance(list(worlds.values())[0], np.ndarray):
        worlds = {k: np.array(world.discrete) for k, world in worlds.items()}
    # fitnesses = {k: [] for k in worlds}
    rl_stats = []

    # Train/evaluate on all worlds n_trials many times each
    # for i in range(n_trials):
    qd_stats = {}
    world_id = 0

    # Train/evaluate until we have simulated in all worlds
    while world_id < len(worlds):

        # When running parallel envs, if each env is to evaluate a separate world, map envs to worlds
        if idx_counter:
            # n_envs = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: 1))
            envs = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env))
            envs = [e for we in envs for e in we]
            n_envs = len(envs)
            # hashes = [hash(e) for e in envs]

            # Have to get hashes on remote workers. Objects have different hashes in "envs" above.
            hashes = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
            hashes = [h for wh in hashes for h in wh]

            # n_envs = sum([e for we in hashes for e in we])
            sub_idxs = idxs[world_id:min(world_id + n_envs, len(idxs))]
            idx_counter.set_idxs.remote(sub_idxs)
            idx_counter.set_hashes.remote(hashes)

            # FIXME: Sometimes hash-to-idx dict is not set by the above call?
            assert ray.get(idx_counter.scratch.remote())

        # Assign envs to worlds
        workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.set_worlds(worlds=worlds, idx_counter=idx_counter)))

        # If using oracle, manually load the world
        if oracle_policy:
            flood_model = trainer.get_policy('policy_0').model
            workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env.reset()))

            # Hardcoded rendering
            envs = workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env))
            envs = [e for we in envs for e in we]
            envs[0].render()

            new_fitnesses = workers.foreach_worker(
                lambda worker: worker.foreach_env(
                    lambda env: {env.world_idx: ((flood_model.get_solution_length(th.Tensor(env.world).unsqueeze(0)),), (0,0))}))
            rl_stats.append([])

        else:
            # Train/evaluate
            if evaluate_only:
                stats = trainer.evaluate()
            else:
                stats = trainer.train()
            # print(pretty_print(stats))
            rl_stats.append(stats)

            # Collect stats for generator
            new_fitnesses = workers.foreach_worker(
                lambda worker: worker.foreach_env(
                    lambda env: env.get_world_stats(evaluate=evaluate_only, quality_diversity=quality_diversity)))
        assert(len(rl_stats) == 1)
        rl_stats = rl_stats[0]
        logbook_stats = {'iteration': net_itr}
        stat_keys = ['mean', 'min', 'max']  # , 'std]  # Would need to compute std manually
        # if i == 0:
        if calc_world_stats:
            world_stats = get_env_world_stats(trainer)
            logbook_stats.update({f'{k}Path': world_stats[f'{k}_path_length'] for k in stat_keys})
        if calc_agent_stats and not evaluate_only:
            logbook_stats.update({
                f'{k}Rew': rl_stats[f'episode_reward_{k}'] for k in stat_keys})
        if 'evaluation' in rl_stats:
            logbook_stats.update({
                f'{k}EvalRew': rl_stats['evaluation'][f'episode_reward_{k}'] for k in stat_keys})
        # logbook.record(**logbook_stats)
        # print(logbook.stream)
        new_fitnesses = [fit for worker_fits in new_fitnesses for fit in worker_fits]
        new_fits = {}
        [new_fits.update(nf) for nf in new_fitnesses]
        qd_stats.update(new_fits)

        # If we've mapped envs to specific worlds, then we count the number of unique worlds evaluated (assuming worlds are
        # deterministic, so re-evaluation is redundant, and we may sometimes have redundant evaluations because we have too many envs).
        # Otherwise, we count the number of evaluations (e.g. when evaluating on a single fixed world).
        if idx_counter:
            world_id = len(qd_stats)
        else:
            world_id += len(new_fitnesses)

        logbook_stats.update({
            'elapsed': timer() - start_time,
        })

        # [fitnesses[k].append(v) for k, v in trial_fitnesses.items()]
    # fitnesses = {k: ([np.mean([vi[0][fi] for vi in v]) for fi in range(len(v[0][0]))],
    #         [np.mean([vi[1][mi] for vi in v]) for mi in range(len(v[0][1]))]) for k, v in fitnesses.items()}

    return rl_stats, qd_stats, logbook_stats


def train_players(net_itr, play_phase_len, n_policies, n_pop, trainer, landscapes, save_dir, n_rllib_envs, n_sim_steps, 
                  idx_counter=None, logbook=None, quality_diversity=False):
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
        # curr_lands = np.array(landscapes)[world_idxs]
        curr_lands = [landscapes[wid] for wid in world_idxs]
        worlds = {i: l for i, l in enumerate(curr_lands)}

        # Get world stats on first iteration only, since they won't change after this inside training loop
        calc_world_stats = (i == 0)

        rl_stats, qd_stats, logbook_stats = rllib_evaluate_worlds(
            net_itr=net_itr, trainer=trainer, worlds=worlds, idx_counter=idx_counter, calc_agent_stats=True,
            start_time=start_time, evaluate_only=False, quality_diversity=quality_diversity, 
            calc_world_stats=calc_world_stats)
        recent_rewards[:-1] = recent_rewards[1:]
        recent_rewards[-1] = rl_stats['episode_reward_mean']

        if i == 0:
            mean_path_length = logbook_stats['meanPath']
            max_mean_reward = (n_sim_steps - mean_path_length) * n_pop

        # End training if within a certain margin of optimal performance
        done_training = recent_rewards[-1] >= 0.9 * max_mean_reward

        # if play_phase_len == -1:
            # if i >= staleness_window:
                # running_std = np.std(recent_rewards)
                # print(f'Running reward std dev: {running_std}')
                # done_training = running_std < 1.0 or i >= 100
            # else:
                # done_training = False
        logbook.record(**logbook_stats)
        print(logbook.stream)
        i += 1
        if play_phase_len != -1:
            done_training = done_training or i >= play_phase_len
        net_itr += 1
    # toggle_exploration(trainer, explore=False, n_policies=n_policies)
    trainer.workers.local_worker().set_policies_to_train([])
    return net_itr


@ray.remote
class IdxCounter:
    ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
    responsible for providing unique indices to parallel environments.'''

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

    def set_hashes(self, hashes, allow_one_to_many: bool=False):
        if not allow_one_to_many:
            assert len(hashes) >= len(self.idxs)
        idxs = self.idxs

        # If we have more hashes than indices, map many-to-one
        if len(hashes) > len(idxs):
            n_repeats = math.ceil(len(hashes) / len(idxs))
            idxs = np.tile(idxs, n_repeats)
        self.hashes_to_idxs = {hsh: id for hsh, id in zip(hashes, idxs[:len(hashes)])}

    def scratch(self):
        return self.hashes_to_idxs


def init_particle_trainer(env, num_rllib_remote_workers, n_rllib_envs, evaluate, enjoy, render, save_dir, num_gpus, 
                          oracle_policy, fully_observable, idx_counter):
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

        # ModelCatalog.register_custom_model('flood_model', NavNet)
        model_config = copy.copy(MODEL_DEFAULTS)
        if fully_observable:
            model_config.update({
            # Fully observable, non-translated map
            "conv_filters": [
                [64, [3, 3], 1], 
                [64, [3, 3], 2], 
                [64, [3, 3], 2], 
                [64, [3, 3], 2]
            ],
            })
        else:
            model_config.update({
                "use_lstm": True,
                "lstm_cell_size": 32,
                # "fcnet_hiddens": [32, 32],  # Looks like this is unused when use_lstm is True
                "conv_filters": [
                    [16, [5, 5], 1], 
                    [4, [3, 3], 1]],
                # "post_fcnet_hiddens": [32, 32],
            })
    num_workers = 1 if num_rllib_remote_workers == 0 or enjoy else num_rllib_remote_workers
    num_envs_per_worker = math.ceil(n_rllib_envs / num_workers) if not enjoy else 1
    num_eval_envs = num_envs_per_worker
    if enjoy:
        evaluation_num_episodes = num_eval_envs * 2
    elif evaluate:
        evaluation_num_episodes = math.ceil(len(eval_mazes) / num_eval_envs) * num_eval_envs * 5
    else:
        evaluation_num_episodes = math.ceil(len(eval_mazes) / num_eval_envs) * num_eval_envs

    trainer_config = {
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
        "env_config": {
            "swarm_cls": type(env.swarms[0]),  # Assuming env has only one type of swarm
            "width": env.width,
            "n_policies": len(env.swarms),
            "n_pop": env.swarms[0].n_pop,
            "max_steps": env.max_steps,
            # "pg_width": pg_width,
            "evaluate": False,
            "objective_function": env.obj_fn_str,
            "fully_observable": fully_observable,
        },
        "num_gpus": num_gpus,
        "num_workers": num_rllib_remote_workers if not (enjoy or evaluate) else 0,
        "num_envs_per_worker": num_envs_per_worker,
        "framework": "torch",
        "render_env": render if not enjoy else True,

        # If enjoying, evaluation_interval is nonzero only to ensure eval workers get created for playback.
        "evaluation_interval": 50 if not enjoy else 10,

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
        # Set evaluation workers to eval_mazes. If more eval mazes then envs, the world_idx of each env will be incremented
        # by len(eval_mazes) each reset.
        worlds = {i: maze for i, maze in enumerate(eval_mazes)}
        eval_workers = trainer.evaluation_workers
        hashes = eval_workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
        n_eval_envs = sum([e for we in hashes for e in we])
        idxs = list(worlds.keys())
        idx_counter.set_idxs.remote(idxs)
        hashes = [h for wh in hashes for h in wh]
        idx_counter.set_hashes.remote(hashes, allow_one_to_many=True)
        # FIXME: Sometimes hash-to-idx dict is not set by the above call?
        assert ray.get(idx_counter.scratch.remote())
        # Assign envs to worlds
        eval_workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.set_worlds(worlds=worlds, idx_counter=idx_counter)))

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


def get_env_world_stats(trainer):
    flood_model = trainer.get_policy('oracle').model
    path_lengths = trainer.workers.foreach_worker(
        # lambda w: w.foreach_env(lambda e: flood_model.get_solution_length(th.Tensor(e.world[None,...]))))
        lambda w: w.foreach_env(lambda e: len(get_solution(e.world_flat))))
    path_lengths = [p for worker_paths in path_lengths for p in worker_paths]
    mean_path_length = np.mean(path_lengths)
    min_path_length = np.min(path_lengths)
    max_path_length = np.max(path_lengths)
    # std_path_length = np.std(path_lengths)
    return {
        'mean_path_length': mean_path_length,
        'min_path_length': min_path_length,
        'max_path_length': max_path_length,
    }
    # return mean_path_length, min_path_length, max_path_length  #, std_path_length
