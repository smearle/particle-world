# Each policy can have a different configuration (including custom model).
from argparse import Namespace
import concurrent
import copy
from functools import partial
import functools
import math
import os
from pathlib import Path
from pdb import set_trace as TT
import pprint
import shutil
from typing import Iterable
from gym import Env

import numpy as np
import ray
import torch as th
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
# from ray.tune.logger import pretty_print
from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import Logger
from ray.rllib.agents import Trainer
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.typing import Callable, Optional, PartialTrainerConfigDict, TrainerConfigDict, ResultDict
from timeit import default_timer as timer

from envs import eval_mazes, full_obs_test_mazes, partial_obs_test_mazes, partial_obs_test_mazes_2
from evo.evolve import PlayerEvolver, WorldEvolver
from evo.utils import compute_archive_world_heuristics, save
from models import CustomConvRNNModel, CustomFeedForwardModel, FloodMemoryModel, OraclePolicy, CustomRNNModel, NCA
# from paired_models.multigrid_models import MultigridRLlibNetwork
from rl.callbacks import WorldEvoCallbacks
from rl.utils import IdxCounter, get_world_qd_stats, set_worlds
from utils import log, get_solution


stat_keys = ["mean", "min", "max"]  # , 'std]  # Would need to compute std manually


def init_trainer(env: Env, idx_counter: IdxCounter, env_config: dict, cfg: Namespace, gen_only: bool=False, play_only: bool=False):
    """Initialize an RLlib trainer object for training neural nets to control (populations of) particles/players in the
    environment.

    #TODO: abstract this to support training generators as well.

    Args:
        env: a dummy environment, created in the main script, from which we will draw environment variables. Will not
            be used in actual training.
        num_rllib_remote_workers: how many RLlib remote workers to use for training (1 core per worker). Just a local
            worker if 0.
        n_rllib_envs: how many environments to use for training. This determines how many worlds/generators can be
            evaluated with each call to train. When evolving worlds, the determines the batch size.
        evaluate: if True, then we are evaluating only (no training), and will launch n_rllib_workers-many
            evaluation workers.
        enjoy: if True, then the trainer is being initialized only to render and observe trained particles, so we
            will set other rllib parameters accordingly.
        render: whether to render an environment during training.
        save_dir (str): The directory in which experiment logs and checkpoints are to be saved.
        num_gpus (int): How many GPUs to use for training.
        gen_only (bool): Whether the trainer is being initialized strictly for evolving generators (i.e. will never update 
            player policies).
        play_only (bool): Whether the trainer is being initialized strictly for training players. If this and `is_gen` are
            both False, then the trainer is being initialized both for training players and evolving generators.

    Returns: 
        An rllib PPOTrainer object
    """
    # Create multiagent config dict if env is multi-agent
    # Create models specialized for small-scale grid-worlds
    # TODO: separate these tasks?
    model_config = copy.copy(MODEL_DEFAULTS)
    if issubclass(type(env.unwrapped), MultiAgentEnv):
    # if isinstance(env, MultiAgentEnv):
        is_multiagent_env = True

        # TODO: make this general
        n_policies = len(env.swarms)

        if cfg.oracle_policy:
            policies_dict = {f'policy_{i}': PolicySpec(policy_class=OraclePolicy, observation_space=env.observation_spaces[i],
                                                        action_space=env.action_spaces[i], config={})
                                for i, _ in enumerate(env.swarms)}
            model_config = {}
        else:
            policies_dict = {f'policy_{i}': gen_policy(i, env.observation_spaces[i], env.action_spaces[i], env.fields_of_view[i])
                            for i, swarm in enumerate(env.swarms)}
            # If this is a player-only, or generator-and-player trainer, we add the oracle (flood-fill convolutional 
            # model) for fast path-length computations to estimate complexity of worlds in the archive. obs/act spaces 
            # are dummy.
            if not play_only:
                policies_dict.update({
                    'oracle': PolicySpec(policy_class=OraclePolicy, observation_space=env.observation_spaces[0], 
                    action_space=env.action_spaces[0], config={})
                    })
        multiagent_config = {
            "policies": policies_dict,
            "policy_mapping_fn":
                lambda agent_id, episode, worker, **kwargs: f'policy_{agent_id[0]}',
        }
    else:
        is_multiagent_env = False
        multiagent_config = {}

    if cfg.env_is_minerl:
        from envs.minecraft.touchstone import TouchStone
        ModelCatalog.register_custom_model('minerl', CustomConvRNNModel)
        model_config.update({'custom_model': 'minerl'})
    elif cfg.fully_observable and cfg.model == 'strided_feedforward':
        if cfg.rotated_observations:
            model_config.update({
            # Fully observable, translated and padded map
            "conv_filters": [
                [32, [3, 3], 2], 
                [64, [3, 3], 2], 
                [128, [3, 3], 2], 
                [256, [3, 3], 2]
            ],})
        else:
            model_config.update({
            # Fully observable, non-translated map
            "conv_filters": [
                [64, [3, 3], 2], 
                [128, [3, 3], 2], 
                [256, [3, 3], 2]
            ],})
    else:
        if cfg.model is None:
            model_config.update({
                "use_lstm": True,
                "lstm_cell_size": 32,
                # "fcnet_hiddens": [32, 32],  # Looks like this is unused when use_lstm is True
                "conv_filters": [
                    [16, [5, 5], 1], 
                    [4, [3, 3], 1]],
                # "post_fcnet_hiddens": [32, 32],
            })
        elif cfg.model == 'feedforward':
            ModelCatalog.register_custom_model('feedforward', CustomFeedForwardModel)
            model_config = {'custom_model': 'feedforward'}
        elif cfg.model == 'rnn':
            # ModelCatalog.register_custom_model('rnn', MultigridRLlibNetwork)
            ModelCatalog.register_custom_model('rnn', CustomRNNModel)
            model_config = {'custom_model': 'rnn'}
        # TODO: this seems broken
        elif cfg.model == 'flood':
            ModelCatalog.register_custom_model('flood', FloodMemoryModel)
            model_config = {'custom_model': 'flood', 'custom_model_config': {'player_chan': env.player_chan}}
#           elif model == 'nca':
#               ModelCatalog.register_custom_model('nca', NCA)
#               model_config = {'custom_model': 'nca'}

        else:
            raise NotImplementedError


    # Set the number of environments per worker to result in at least n_rllib_envs-many environments.
    n_train_workers = 1 if cfg.n_train_workers == 0 or cfg.enjoy else cfg.n_train_workers
    # num_envs_per_worker = math.ceil(cfg.n_rllib_envs / n_train_workers) if not cfg.enjoy else 1
    num_envs_per_worker = cfg.n_envs_per_worker

    # Because we only ever use a single worker for all evaluation environments.
    num_eval_envs = num_envs_per_worker

    # TODO: testing environments for minerl evaluation.
    # We don't need to evaluate if trainer only used for generators (this will be handled by player-only generator).
    # If gen_adversarial_worlds, we're not updating player policies, so we don't need to evaluate.
    if cfg.env_is_minerl or cfg.gen_adversarial_worlds or gen_only:
        evaluation_interval = 0
    else:
        evaluation_interval = 10 if cfg.fixed_worlds else 1

    if cfg.enjoy:
        # Technically, we have 2 episodes during each call to eval. Imagine we have just called ``trainer.queue_worlds()``.
        # Then when we call ``trainer.evaluate()``, the environment is done on the first step, so that it can load the
        # next world on the subsequent reset. It then iterates through the usual number of steps and is done. The first
        # episode is an exception. 
        evaluation_num_episodes = num_eval_envs * 2
    # elif evaluate:
        # evaluation_num_episodes = math.ceil(len(eval_mazes) / num_eval_envs) * 10
    else:
        # This ensures we get stats for eval eval world with each call to ``trainer.evaluate()``. We may want to 
        # decrease this if we have a large number of eval worlds.
        evaluation_num_episodes = len(eval_mazes)

        # This means we evaluate 1 episode per eval env.
        # evaluation_num_episodes = num_eval_envs

    regret_callbacks = partial(WorldEvoCallbacks, cfg=cfg)
    logger_config = {
            "type": "ray.tune.logger.TBXLogger",
            # Optional: Custom logdir (do not define this here
            # for using ~/ray_results/...).
            "logdir": os.path.abspath(cfg.save_dir),
    }

    trainer_config = {
        "callbacks": regret_callbacks,
        "multiagent": multiagent_config,
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
        "num_gpus": cfg.num_gpus,
#       "num_workers": num_workers if not (cfg.enjoy or cfg.evaluate) else 0,
        "num_workers": cfg.n_train_workers if not (cfg.enjoy or cfg.evaluate or cfg.n_train_workers == 0) else 0,
        "num_envs_per_worker": num_envs_per_worker,
        "framework": "torch",
        "render_env": cfg.render if not cfg.enjoy else True,

        # If enjoying, evaluation_interval is nonzero only to ensure eval workers get created for playback.
        "evaluation_interval": evaluation_interval,

        # We'll only parallelize eval workers when doing evaluation on pre-trained agents.
        "evaluation_num_workers": 0 if not (cfg.evaluate) else cfg.n_evo_workers,

        # FIXME: Hack workaround: during evaluation (after training), all but the first call to trainer.evaluate() will 
        # be preceded by calls to env.set_world(), which require an immediate reset to take effect. (And unlike 
        # trainer.train(), evaluate() waits until n episodes are completed, as opposed to proceeding for a fixed number 
        # of steps.)
        "evaluation_duration": evaluation_num_episodes,
        # "evaluation_duration": "auto",
        "evaluation_duration_unit": "episodes",
        "evaluation_parallel_to_training": True,

        # Evolving players (instead of training them with RLlib)
        "evolve_players": cfg.evolve_players,

#       # TODO: provide more options here?  
        "evo_eval_num_workers": cfg.n_evo_workers,
#       "evo_eval_duration": "auto",
        "evo_eval_config": {
            "batch_mode": "complete_episodes",
            "fixed_worlds": cfg.fixed_worlds,
            # TODO: modify the behavior of `sample()` on the evo_eval worker so that it gets exactly one episode from each environment?
            # "rollout_fragment_length": env.max_episode_steps,
            "rollout_fragment_length": 1,
#           "train_batch_size": env.max_episode_steps * num_envs_per_worker,
            "env_config": {
                **env_config,
                "training_world": False,
            }
        },

        # We *almost* run the right number of episodes s.t. we simulate on each map the same number of times. But there
        # are some garbage resets in there (???).
        "evaluation_config": {
            "env_config": {
                # "n_pop": 1,
                **env_config,

                # If enjoying, then we look at generated levels instead of eval levels. (Because we user trainer.evaluate() when enjoying.)
                "evaluate": True if not cfg.enjoy else cfg.evaluate,
                "num_eval_envs": num_eval_envs,
                "training_world": False,
            },
            "render_env": cfg.render,
            "explore": False if cfg.oracle_policy else True,
        },
        "ignore_worker_failures": True,

        # "lr": 0.1,
        # "log_level": "INFO",
        # "record_env": True,
#       "rollout_fragment_length": env.max_episode_steps,
        # This guarantees that each call to train() simulates 1 episode in each environment/world.

        # TODO: try increasing batch size to ~500k, expect a few minutes update time
        # "train_batch_size": env.max_episode_steps * cfg.n_eps_on_train,
#       "train_batch_size": env.max_episode_steps * num_envs_per_worker,
        # "sgd_minibatch_size": env.max_episode_steps * cfg.n_rllib_envs if (cfg.enjoy or cfg.evaluate) and cfg.render else 128,
        "logger_config": logger_config if not (cfg.enjoy or cfg.evaluate) else {},
    }

    pp = pprint.PrettyPrinter(indent=4)

    # Log the trainer config, excluding overly verbose entries (i.e. Box observation space printouts).
    trainer_config_loggable = trainer_config.copy()
    trainer_config_loggable.pop('multiagent')
    print(f'Loading trainer with config:')
    pp.pprint(trainer_config_loggable)

    # if cfg.parallel_gen_play and not (cfg.fixed_worlds or cfg.enjoy or cfg.evaluate):
    trainer = WorldEvoPPOTrainer(env='world_evolution_env', config=trainer_config)
    # else:
        # trainer = ppo.PPOTrainer(env='world_evolution_env', config=trainer_config)

    # When enjoying, eval envs are set from the evolved world archive in rllib_eval_envs. We do not evaluate when 
    # evolving adversarial worlds.
    if trainer.evaluation_workers:
    # if not cfg.enjoy and not cfg.gen_adversarial_worlds and not cfg.env_is_minerl:  # TODO: eval worlds in minerl
        # Set evaluation workers to eval_mazes. If more eval mazes then envs, the world_key of each env will be incremented
        # by len(eval_mazes) each reset.
        eval_workers = trainer.evaluation_workers
        hashes = eval_workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
        n_eval_envs = sum([e for we in hashes for e in we])
        idxs = list(eval_mazes.keys())

        eval_envs_per_worker = eval_workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: env))
        eval_envs = [env for we in eval_envs_per_worker for env in we]
        n_eval_envs = len(eval_envs)

        # Pad the list of indices with duplicates in case we have more than enough eval environments
        idxs *= math.ceil(n_eval_envs / len(idxs))
        idxs = idxs[:n_eval_envs]

        idx_counter.set_keys.remote(idxs)
        hashes = [h for wh in hashes for h in wh]
        idx_counter.set_hashes.remote(hashes)
        # FIXME: Sometimes hash-to-idx dict is not set by the above call?
        assert ray.get(idx_counter.scratch.remote())
        # Assign envs to worlds
        eval_workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.queue_worlds(worlds=eval_mazes, idx_counter=idx_counter, load_now=True)))

    if is_multiagent_env:
        policy_names = [f'policy_{i}' for i in range(n_policies)]
    else:
        policy_names = ['default_policy']

    for policy_name in policy_names:
        n_params = 0
        param_dict = trainer.get_weights()[policy_name]
        for v in param_dict.values():
            n_params += np.prod(v.shape)
        print(f'{policy_name} has {n_params} parameters.')
        print('model overview: \n', trainer.get_policy(f'{policy_name}').model)
    return trainer


def gen_policy(i, observation_space, action_space, field_of_view):
    config = {
        "model": {
            "custom_model_config": {
                "field_of_view": field_of_view,
            }
        }
    }
    return PolicySpec(config=config, observation_space=observation_space, action_space=action_space)


def save_model(trainer: Trainer, save_dir: str):
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


def toggle_exploration(trainer: Trainer, explore: bool, n_policies: int):
    for i in range(n_policies):
        trainer.get_policy(f'policy_{i}').config["explore"] = explore
        # Need to update each remote training worker as well (if they exist)
        trainer.workers.foreach_worker(lambda w: w.get_policy(f'policy_{i}').config.update({'explore': explore}))


def toggle_train_player(trainer: Trainer, train_player: bool, cfg: Namespace):
    if train_player:
        trainer.workers.local_worker().set_policies_to_train([f'policy_{i}' for i in range(cfg.n_policies)])
    else:
        trainer.workers.local_worker().set_policies_to_train([])


def sync_player_policies(play_trainer: Trainer, gen_trainer: Trainer, cfg: Namespace):
    """Sync player policies from the player-trainer to the generator-trainer."""
    for i in range(cfg.n_policies):
        gen_trainer.get_policy(f'policy_{i}').set_weights(play_trainer.get_weights()[f'policy_{i}'])


def gen_playable_duration_fn(generations_done, num_playable_worlds=0, **kwargs):
    """Evolve worlds until one is playable."""
    return 0 if num_playable_worlds > 0 else 1


algorithm = PPOTrainer
# algorithm = ImpalaTrainer


class WorldEvoPPOTrainer(algorithm):
    """A subclass of PPOTrainer that adds the ability to evolve worlds in parallel to player training."""

    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        cfg = algorithm.get_default_config()
        cfg["evolve_players"] = False
        cfg["evo_eval_num_workers"] = 1
        cfg["evo_eval_duration"] = "auto"
        cfg["evo_eval_duration_unit"] = "episodes"
        cfg["evo_eval_config"] = {
            "fixed_worlds": False,
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 1,
            "evaluation_duration_unit": "episodes",
            "train_batch_size": cfg["train_batch_size"],
            "env_config": cfg["env_config"],
        }

        return cfg

    def set_attrs(self, world_evolver: WorldEvolver, idx_counter, logbook, colearning_config, net_itr, gen_itr, \
            play_itr, player_evolver: Optional[PlayerEvolver] = None):
        if world_evolver:
            self.world_evolver = world_evolver
            self.world_archive = world_evolver.container
        if colearning_config.evolve_players:
            assert player_evolver is not None
        self.player_evolver = player_evolver
        self.player_archive = player_evolver.container if player_evolver else None
        self.idx_counter = idx_counter
        self.logbook = logbook
        self.colearn_cfg = colearning_config
        self.net_itr = net_itr
        self.gen_itr = gen_itr
        self.play_itr = play_itr
        self.net_train_start_time = None
        self._just_loaded = True
        self.last_agent_timesteps_total = 0

    def setup(self, config: PartialTrainerConfigDict):
        super().setup(config)
        self.world_evolver = None
        self.world_archive = None
        self.gen_itr, self.play_itr, self.net_itr = 0, 0, 0
        # Update with evaluation settings:
        user_evo_eval_config = copy.deepcopy(self.config["evo_eval_config"])

        # Merge user-provided eval config with the base config. This makes sure
        # the eval config is always complete, no matter whether we have eval
        # workers or perform evaluation on the (non-eval) local worker.
        evo_eval_config = merge_dicts(self.config, user_evo_eval_config)

        if evo_eval_config["fixed_worlds"]:
            # self.world_archive = full_obs_test_mazes
            self.world_archive = partial_obs_test_mazes
            # self.world_archive = partial_obs_test_mazes_2

        else:
            # Create a separate evolution evaluation worker set for evo eval.
            # If num_workers=0, use the evo eval set's local
            # worker for evaluation, otherwise, use its remote workers
            # (parallelized evaluation).
            self.evo_eval_workers: WorkerSet = self._make_workers(
                env_creator=self.env_creator,
                validate_env=None,
                policy_class=self.get_default_policy_class(self.config),
                config=evo_eval_config,
                num_workers=self.config["evo_eval_num_workers"],
                # Don't even create a local worker if num_workers > 0.
                local_worker=False,
                )
            self.evo_eval_workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.set_mode("evo_eval_world")))

        # FIXME: too many workers getting created when specifying 1 trainer worker and n evo eval workers?

    def evaluate(self, episodes_left_fn=None, duration_fn: Optional[Callable[[int], int]] = None) -> dict:
        return {}
        # FIXME: This is a hack. Should be able to use configs for this?
        self.evaluation_workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.set_mode("evaluation_world")))

        eval_start_time = timer()
        logbook_stats = {}
        eval_results = super().evaluate(episodes_left_fn=episodes_left_fn, duration_fn=duration_fn)
        
        # No logbook when enjoying or doing eval.
        if hasattr(self, 'logbook'):
            logbook_stats.update({
                f'{k}EvalRew': eval_results["evaluation"][f"episode_reward_{k}"] for k in stat_keys})
            logbook_stats.update({'elapsed': timer() - eval_start_time})
            log(self.logbook, logbook_stats, self.net_itr)

        return eval_results

    def step_attempt(self) -> ResultDict:
        """Attempts a single training step, including player evaluation, if required. Performs evo-eval in parallel.

        Override this method in your Trainer sub-classes if you would like to
        keep the n step-attempts logic (catch worker failures) in place or
        override `step()` directly if you would like to handle worker
        failures yourself.

        Returns:
            The results dict with stats/infos on sampling, training,
            and - if required - evaluation.
        """

        def auto_duration_fn(generations_done, num_workers, cfg, **kwargs):
            """Evolve worlds until the concurrent player training step is complete."""
            # Training is done and we already ran at least one
            # evaluation -> Nothing left to run.
            if generations_done > 0 and \
                    train_future.done():
                return 0
            # Count by episodes. -> Run n more
            # (n=num eval workers).
            return num_workers * self.config["num_envs_per_worker"]

        # Preliminary evolution until we have generated a playable world.
        if not self.colearn_cfg.fixed_worlds and self.net_itr == 0:
            # TODO: may need to occupy the GPU with something else if this will take a long time. 
            # E.g. train on repaired worlds.
            logbook_archive_stats = self.evo_eval(duration_fn=gen_playable_duration_fn)

        train_start_time = timer()

        if self.net_train_start_time is None:
            self.net_train_start_time = train_start_time

        if self.colearn_cfg.fixed_worlds:
            # Just training on the train set, which is stored in the world archive (kinda weird).
            training_worlds = self.world_archive

#       elif self.net_itr == 0:
#           # On the first iteration, we'll just train on some random individuals (the initial batch stored in the 
#           # `world_evolver`).
#           training_worlds = self.world_evolver.generate_batch(self.colearn_cfg.world_batch_size)

        else:
            # training_worlds = {k: ind for k, ind in enumerate(
            #     sorted(self.world_evolver.container, key=lambda i: i.fitness.values[0], reverse=True))}
            training_worlds = {k: ind for k, ind in enumerate(self.world_evolver.container)}
            # Randomly sample from the training worlds if we don't have enough.
            if len(training_worlds) < self.colearn_cfg.world_batch_size:
                world_keys = np.random.choice(list(training_worlds.keys()), self.colearn_cfg.world_batch_size, replace=True)
                training_worlds = {k: training_worlds[k] for k in world_keys}

        # else {k: ind for k, ind in enumerate(self.world_evolver.container)}

        if self.colearn_cfg.quality_diversity:
            # Eliminate impossible worlds
            learnable_training_worlds = {k: ind for k, ind in training_worlds.items() if not ind.features == [0, 0]}
            training_worlds = learnable_training_worlds if len(learnable_training_worlds) > 0 else training_worlds

#           # In case all worlds are impossible, do more rounds of evolution until some worlds are feasible.
#           if len(training_worlds) == 0:
#               done_play_phase = True

        # Filter out unplayable worlds if applicable.
        training_worlds = {k: ind for k, ind in training_worlds.items() if isinstance(ind, np.ndarray) or ind.playability_penalty == 0}

        # If evolving a player, take a smaller set of the best worlds for evaluating player agents.
        if self.colearn_cfg.evolve_players and not self.colearn_cfg.evolve_players:
            sorted([ind for ind in training_worlds.values()], key=lambda i: i.fitness.values[0], reverse=True)
            training_worlds[:self.colearn_cfg.world_batch_size]
            training_worlds = {k: ind for k, ind in enumerate(training_worlds)}

        # TODO: Would num_worlds < num_envs be a problem here? Work around this if so (make world-assignment optionally
        #   flexible).
        set_worlds(training_worlds, self.workers, self.idx_counter, self.colearn_cfg, load_now=False)
        step_results = {}
        logbook_stats = {}

        if self.config["evolve_players"]:
            # FIXME: certain envs are both `evolve_player` and `evaluation_world` ???
            self.workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.set_mode("evolve_player")))
        else:
            self.workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.set_mode("training_world")))

        # NOTE: eval is always sequential because meh
        # Sequential training and evolution.
        if self.config["num_workers"] == 0:
            step_results = self._exec_plan_or_training_iteration_fn()
            logbook_archive_stats = {}

            if self.play_itr % self.config["evaluation_interval"] == 0:
                step_results.update(self.evaluate())

            if not self.colearn_cfg.fixed_worlds:
                # Run evo-eval indefinitely
                logbook_archive_stats = self.evo_eval(
                        duration_fn=lambda n_units_done, n_playable: n_units_done > 0)
            
        else:
            # Kick off evaluation-loop (and parallel train() call,
            # if requested).
            # Parallel eval + training.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                train_future = executor.submit(
                    lambda: self._exec_plan_or_training_iteration_fn())
                logbook_archive_stats = {}

                if self.play_itr % self.config["evaluation_interval"] == 0:
                    step_results.update(self.evaluate())

                if not self.colearn_cfg.fixed_worlds:
                    # Run evo-eval indefinitely
                    logbook_archive_stats = self.evo_eval(
                            duration_fn=functools.partial(
                                auto_duration_fn, unit="episodes", num_workers=self.config[
                                    "num_workers"], cfg=self.config[
                                        "evo_eval_config"]))
                
                # Collect the training results from the future.
                step_results.update(train_future.result())
            
#       # FIXME: This gathers and flushes stats before the episode is over. Need to use callbacks instead
#       if self.colearn_cfg.fixed_worlds:
#           qd = self.colearn_cfg.quality_diversity
#           # NOTE: This flushes the stats on all these workers' envs.
#           world_stats = self.workers.foreach_worker(
#               lambda worker: worker.foreach_env(lambda env: env.get_world_stats(quality_diversity=qd)))
#           # Flatten workers
#           world_stats = [s for ws in world_stats for s in ws]
#           pct_wins = []
#           for env_stat_list in world_stats:
#               for world_stat_dict in env_stat_list:
#                   pct_win = np.mean([world_stat_dict[f"policy_{i}"]["pct_win"] for i in range(self.colearn_cfg.n_policies)])
#                   pct_wins.append(pct_win)
#           logbook_stats.update({
#               "pctWin": np.mean(pct_wins),
#           })

        logbook_stats.update({
            f"{k}Rew": step_results[f"episode_reward_{k}"] for k in stat_keys})
        train_end_time = timer()
        logbook_stats.update({
            "fps": (step_results["agent_timesteps_total"] - self.last_agent_timesteps_total) / (train_end_time -train_start_time),
        })
        self.last_agent_timesteps_total = step_results["agent_timesteps_total"]

        # Note that we report stats about the worlds in the archive at step t, and the result of training at step t,
        # though each has been evolved/trained on the policy/worlds at step t-1.
        logbook_stats.update(logbook_archive_stats)
        logbook_stats.update({"elapsed": train_end_time - train_start_time})
        log(self.logbook, logbook_stats, self.net_itr)
        if not self.colearn_cfg.fixed_worlds or self.play_itr % 10 == 0:
            save_model(self, self.colearn_cfg.save_dir)
            save(world_archive=self.world_archive, player_archive=self.player_archive, gen_itr=self.gen_itr, 
                net_itr=self.net_itr, play_itr=self.play_itr, logbook=self.logbook, save_dir=self.colearn_cfg.save_dir)
        self.play_itr += 1
        self.net_itr += 1

        return step_results
            

    def evo_eval(self, duration_fn: Optional[Callable[[int], int]] = None) -> dict:
        """Evaluates current policy under `evaluation_config` settings.

        Note that this default implementation does not do anything beyond
        merging evaluation_config with the normal trainer config.

        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
        # Sync weights to the evo-eval WorkerSet.
        self.evo_eval_workers.sync_weights(
            from_worker=self.workers.local_worker())
        self._sync_filters_if_needed(self.evo_eval_workers)

        # How many episodes/timesteps do we need to run?
        # In "auto" mode (only for parallel eval + training): Run as long
        # as training lasts.
        unit = "episodes"
        # evo_eval_cfg = self.config["evo_eval_config"]
        # rollout = evo_eval_cfg["rollout_fragment_length"]
        # num_envs = self.config["num_envs_per_worker"]
        # duration = self.config["num_workers"] * self.config["num_envs_per_worker"]
        num_ts_run = 0
        metrics = {}

        playable_worlds = []
        # How many episodes have we run (across all eval workers)?
        generations_done = 0
        while True:
            evo_start_time = timer()
            generations_left_to_do = duration_fn(generations_done=generations_done, num_playable_worlds=len(playable_worlds))
            
            if generations_left_to_do <= 0:
                # The player is one update ahead of the worlds, relative the policy the worlds were evolved for.
                self.world_evolver.increment_ages()
                break
                
#           print("Ages of stale individuals: ", [ind.fitness.age for ind in self.world_evolver.stale_individuals])
            world_batch = self.world_evolver.ask(self.colearn_cfg.world_batch_size)
            # unplayable_worlds = [w for w in world_batch if not w.playability_penalty > 0]
            playable_worlds = {wk: ind for wk, ind in world_batch.items() if ind.playability_penalty == 0}
            world_qd_stats = {}

            if len(playable_worlds) > 0:
                # New variable, otherwise we'll try to serialize `self` in the workerset lambda in the below function.
                idx_counter = ray.get_actor("idx_counter")

                set_worlds(playable_worlds, self.evo_eval_workers, idx_counter, self.colearn_cfg, load_now=True)

                # ep_ids = set({})
                batches = []
                generations_done += 1

                # Need to take an extra batch of samples because of the way we queue/load worlds (I think). Results in
                # duplicate evaluations on first iteration. Kind of feels like we're flying by the seat of our pants 
                # here and trusting sample. Maybe it will win us over.
                for i in range(self.colearn_cfg.world_batch_size // max(1, self.colearn_cfg.n_evo_workers) + (0 if self._just_loaded else 1)):
                # while True:

                    # print(f"Sample batch {i}")
                    batch = ray.get([
                        w.sample.remote() for i, w in enumerate(
                            self.evo_eval_workers.remote_workers())
                    ])
                    batches.append(batch)
                    
                    # This is ugly. Better way to count episodes?
                    # [ep_ids.update(set(b.policy_batches['policy_0']['eps_id'])) for b in batch]
                    # if len(ep_ids) >= self.colearn_cfg.world_batch_size * (1 if self._just_loaded else 2):
                        # break

                metrics = collect_metrics(remote_workers=self.evo_eval_workers.remote_workers())
                hist_stats = metrics['hist_stats']
                world_stats = [{k: hist_stats[k][i] for k in hist_stats} for i in range(len(hist_stats['world_key']))]

                assert len(world_stats) == self.colearn_cfg.world_batch_size, f"{len(world_stats)} != {self.colearn_cfg.world_batch_size}"

                # TODO: Assuming we've only evaluated each world once. Not true if episodes can terminate early. 
                # Should bring back some logic from `get_world_qd_stats` to aggregate over these metrics).
                world_stats = {w['world_key']: w for w in world_stats}

                # Get regret / positive value loss metrics from the batches.
                if self.colearn_cfg.objective_function == "regret":
                    assert not self.colearn_cfg.evolve_players
                    pos_val_losses = [bb.policy_batches['policy_0']['pos_val_loss'] for b in batches for bb in b]
                    pos_val_losses = {k: v for pv in pos_val_losses for k, v in pv.items()}

                    assert len(pos_val_losses) == len(world_stats)

                    # [ws.update({'pos_val_loss': pos_val_loss}) for ws, pos_val_loss in zip(world_stats, pos_val_losses)]
                    [world_stats[k].update(
                        {'qd_stats': ((self.colearn_cfg._obj_fn(pos_val_losses[k]),), world_stats[k]['qd_stats'][1])}
                        ) for k in world_stats]

                # world_qd_stats = get_world_qd_stats(world_stats, self.colearn_cfg,
                    # ignore_redundant=self.colearn_cfg.generator_class in ["NCA"] or self._just_loaded)
                world_qd_stats = {wk: w['qd_stats'] for wk, w in world_stats.items()}

                # print(world_qd_stats.keys())
            logbook_stats = self.world_evolver.tell(world_batch, world_qd_stats)
            logbook_stats.update({"elapsed": timer() - evo_start_time})
            log(self.logbook, logbook_stats, self.net_itr)
            self._just_loaded = False
            self.gen_itr += 1
            self.net_itr += 1

        if self.config["evolve_players"]:
            self.player_evolver.increment_ages()

        # The training step is complete, so we are done this phase of generator-evolution.
        logbook_stats = {}
        mean_path_length = compute_archive_world_heuristics(archive=self.world_evolver.container, trainer=self)
        stat_keys = ['mean', 'min', 'max']  # , 'std]
        logbook_stats.update({f'{k}Path': mean_path_length[f'{k}_path_length'] for k in stat_keys})

#       if metrics is None:
#           metrics = collect_metrics(
#               self.evaluation_workers.local_worker(),
#               self.evaluation_workers.remote_workers())
        metrics["timesteps_this_iter"] = num_ts_run

        return logbook_stats

    def _exec_plan_or_training_iteration_fn(self):
        """Evaluate evolved player policies on a fixed set of worlds and update the player archive."""
        if not self.colearn_cfg.evolve_players:
            return super()._exec_plan_or_training_iteration_fn()

        idx_counter = ray.get_actor("play_idx_counter")
        player_batch = self.player_evolver.ask(self.colearn_cfg.world_batch_size)
        idx_counter.set_keys.remote(player_batch.keys())

        # TODO: Parallelize environment simulation on each worker.
        hashes = self.workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
        hashes = [h for wh in hashes for h in wh]
        idx_counter.set_hashes.remote(hashes)

        # Simulate using a batch of player policies on environments, each with the same set of worlds.
        self.workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.set_player_policies(player_batch, idx_counter)))
        rews = self.workers.foreach_worker(lambda w: w.foreach_env(lambda e: e.simulate()))
        player_rews = {}
        [player_rews.update(r) for worker_rews in rews for r in worker_rews]

        logbook_stats = self.player_evolver.tell(player_batch, player_rews)
        log(self.logbook, logbook_stats, self.net_itr)

        step_results = {f'episode_reward_{s}': getattr(np, s)(list(player_rews.values())) for s in ['mean', 'min', 'max']}
        step_results.update({'agent_timesteps_total': 0})

        if self.colearn_cfg.fixed_worlds:
            self.player_evolver.increment_ages()

        return step_results

        