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

import numpy as np
import ray
import torch as th
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.ppo import ppo
# from ray.tune.logger import pretty_print
from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import Logger
from ray.rllib.agents import Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.typing import Callable, Optional, PartialTrainerConfigDict, TrainerConfigDict, ResultDict
from timeit import default_timer as timer

from envs import eval_mazes
from model import CustomConvRNNModel, FloodMemoryModel, OraclePolicy, CustomRNNModel, NCA
# from paired_models.multigrid_models import MultigridRLlibNetwork
from rl.callbacks import RegretCallbacks
from rl.eval_worlds import evaluate_worlds
from utils import get_solution


def train_players(worlds, trainer, cfg, idx_counter=None):
    """Train player-agents using the rllib trainer.

    :param net_itr: (int) Counter for net iterations of algorithm (including both player training and world evolution 
                    steps.) 
    :param play_phase_len: (int) Number of player training steps (calls to train.train()) to perform.
    :param worlds: (dict) Dictionary of worlds to train on, ({world_key: world}).
    :param trainer: (Trainer) RLLib trainer.
    :param cfg: Object with configuration parameters as attributes.
    :param idx_counter: (Actor) ray actor accessed by parallel rllib envs to assign them unique world_keys.
    :param logbook: (Logbook) qdpy logbook to record (evolution and) training stats.
    """
#   trainer.workers.local_worker().set_policies_to_train([f'policy_{i}' for i in range(cfg.n_policies)])
    # toggle_exploration(trainer, explore=True, n_policies=n_policies)
#   i = 0
#   staleness_window = 10
#   done_training = False
#   recent_rewards = np.empty(staleness_window)
#   while not done_training:
    start_time = timer()
    # Saving before training, so that we have a checkpoint of the model after the evolution phase, and before model
    # weights start changing.
#   if i % 10 == 0:
#       save_model(trainer, cfg.save_dir)
    replace = True if cfg.fixed_worlds else False
    if isinstance(worlds, dict):
        world_keys = list(worlds.keys())
    else:
        assert isinstance(worlds, Iterable)
        world_keys = list(range(len(worlds)))
    world_keys = np.random.choice(world_keys, cfg.world_batch_size, replace=replace)
    # curr_lands = np.array(landscapes)[world_keys]
    curr_worlds = [worlds[wk] for wk in world_keys]
    worlds = {i: l for i, l in enumerate(curr_worlds)}

    # Get world heuristics on first iteration only, since they won't change after this inside training loop
    # calc_world_heuristics = (i == 0)

    rl_stats, qd_stats, logbook_stats = evaluate_worlds(
        trainer=trainer, worlds=worlds, cfg=cfg, idx_counter=idx_counter, is_training_player=True,
        start_time=start_time)

    return logbook_stats
#   recent_rewards[:-1] = recent_rewards[1:]
#   recent_rewards[-1] = rl_stats['episode_reward_mean']

#       if i == 0:
#           mean_path_length = logbook_stats['meanPath']
#           max_mean_reward = (n_sim_steps - mean_path_length) * n_pop

    # End training if within a certain margin of optimal performance
#       done_training = recent_rewards[-1] >= 0.9 * max_mean_reward
#   done_training = False

    # if play_phase_len == -1:
        # if i >= staleness_window:
            # running_std = np.std(recent_rewards)
            # print(f'Running reward std dev: {running_std}')
            # done_training = running_std < 1.0 or i >= 100
        # else:
            # done_training = False
#   if not cfg.fixed_worlds:
#       logbook.record(**logbook_stats)
#       print(logbook.stream)
#   i += 1
#   if play_phase_len != -1:
#       done_training = done_training or i >= play_phase_len
#   net_itr += 1
    # toggle_exploration(trainer, explore=False, n_policies=n_policies)
#   trainer.workers.local_worker().set_policies_to_train([])
#   return net_itr


def init_trainer(env, idx_counter, env_config: dict, cfg: Namespace, gen_only: bool=False, play_only: bool=False):
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
        elif cfg.model == 'paired':
            # ModelCatalog.register_custom_model('paired', MultigridRLlibNetwork)
            ModelCatalog.register_custom_model('paired', CustomRNNModel)
            model_config = {'custom_model': 'paired'}
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
    num_workers = 1 if cfg.n_rllib_workers == 0 or cfg.enjoy else cfg.n_rllib_workers
    num_envs_per_worker = math.ceil(cfg.n_rllib_envs / num_workers) if not cfg.enjoy else 1

    # Because we only ever use a single worker for all evaluation environments.
    num_eval_envs = num_envs_per_worker

    # TODO: testing environments for minerl evaluation.
    # We don't need to evaluate if trainer only used for generators (this will be handled by player-only generator).
    # If gen_adversarial_worlds, we're not updating player policies, so we don't need to evaluate.
    if cfg.env_is_minerl or cfg.gen_adversarial_worlds or gen_only:
        evaluation_interval = 0
    else:
        # After how many calls to train do we evaluate?
        # If we're evolving/training until convergence, just
        # evaluate every 10 iterations. If we're "enjoying", just need to set this to any number > 0 to ensure we 
        # initialize the evaluation workers.
        if -1 in [cfg.gen_phase_len, cfg.play_phase_len] or cfg.enjoy:
            evaluation_interval = 10

        # Otherwise, evaluate policies once after every player-training phase. If the trainer is for player-training and
        # evolved world evaluation, we count the phases corresponding to each process, and the phase corresponding to 
        # the re-evaluation of elites. 
        # TODO: might want multiple rounds of elite-re-evaluation, would need to account for that here.
#       elif not cfg.parallel_gen_play:
        evaluation_interval = (cfg.gen_phase_len + 1 + cfg.play_phase_len) * cfg.world_batch_size // cfg.n_eps_on_train 

#       evaluation_interval = 1
#       # If we're parallelizing generation and playing, we're evaluating on the player-only trainer.
#       else:
#           evaluation_interval = cfg.play_phase_len * cfg.world_batch_size // cfg.n_eps_on_train 

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

    regret_callbacks = partial(RegretCallbacks, regret_objective=cfg.objective_function=='regret')
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
        "num_workers": cfg.n_rllib_workers if not (cfg.enjoy or cfg.evaluate) else 0,
        "num_envs_per_worker": num_envs_per_worker,
        "framework": "torch",
        "render_env": cfg.render if not cfg.enjoy else True,
        # "custom_eval_function": evo_evaluate,

        # If enjoying, evaluation_interval is nonzero only to ensure eval workers get created for playback.
        "evaluation_interval": evaluation_interval,

        # We'll only parallelize eval workers when doing evaluation on pre-trained agents.
        "evaluation_num_workers": 1 if not (cfg.evaluate) else cfg.n_rllib_workers,

        # FIXME: Hack workaround: during evaluation (after training), all but the first call to trainer.evaluate() will 
        # be preceded by calls to env.set_world(), which require an immediate reset to take effect. (And unlike 
        # trainer.train(), evaluate() waits until n episodes are completed, as opposed to proceeding for a fixed number 
        # of steps.)
        "evaluation_duration": evaluation_num_episodes,
        # "evaluation_duration": "auto",
        "evaluation_duration_unit": "episodes",
        "evaluation_parallel_to_training": True,

        # We *almost* run the right number of episodes s.t. we simulate on each map the same number of times. But there
        # are some garbage resets in there (???).
        "evaluation_config": {
            "env_config": {
                # "n_pop": 1,

                # If enjoying, then we look at generated levels instead of eval levels. (Because we user trainer.evaluate() when enjoying.)
                "evaluate": True if not cfg.enjoy else cfg.evaluate,
                "num_eval_envs": num_eval_envs,
            },
            "render_env": cfg.render,
            "explore": False if cfg.oracle_policy else True,
        },

#       # TODO: provide more options here?
#       "evo_eval_duration": "auto",
#       "evo_eval_config": {},

        # "lr": 0.1,
        # "log_level": "INFO",
        # "record_env": True,
        "rollout_fragment_length": env.max_episode_steps,
        # This guarantees that each call to train() simulates 1 episode in each environment/world.

        # TODO: try increasing batch size to ~500k, expect a few minutes update time
        "train_batch_size": env.max_episode_steps * cfg.n_eps_on_train,
        # "sgd_minibatch_size": env.max_episode_steps * cfg.n_rllib_envs if (cfg.enjoy or cfg.evaluate) and cfg.render else 128,
        "logger_config": logger_config if not (cfg.enjoy or cfg.evaluate) else {},
    }

    pp = pprint.PrettyPrinter(indent=4)

    # Log the trainer config, excluding overly verbose entries (i.e. Box observation space printouts).
    trainer_config_loggable = trainer_config.copy()
    trainer_config_loggable.pop('multiagent')
    print(f'Loading trainer with config:')
    pp.pprint(trainer_config_loggable)

    if cfg.parallel_gen_play:
        trainer = WorldEvoPPOTrainer(env='world_evolution_env', config=trainer_config)
    else:
        trainer = ppo.PPOTrainer(env='world_evolution_env', config=trainer_config)

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

        idx_counter.set_idxs.remote(idxs)
        hashes = [h for wh in hashes for h in wh]
        idx_counter.set_hashes.remote(hashes)
        # FIXME: Sometimes hash-to-idx dict is not set by the above call?
        assert ray.get(idx_counter.scratch.remote())
        # Assign envs to worlds
        eval_workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.queue_worlds(worlds=eval_mazes, idx_counter=idx_counter)))

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


def evo_evaluate(trainer: Trainer, workers: WorkerSet):
    return trainer.evaluate()


class WorldEvoPPOTrainer(ppo.PPOTrainer):
    """A subclass of PPOTrainer that adds the ability to evolve worlds in parallel to player training."""
#   def evaluate(self, duration_fn=None):
#       # Two pieces of good news:
#       #   - we can do eval in parallel, until the concurrent call to `train()` is complete.
#       #   - calling `evaluate()` takes samples, which leads to the `on_sample_end()` callback being called, where we
#       #     can easily compute positive value loss, if necessary.
#       # This means we can take the following approach to parallel gen & play:
#       # TODO: Use additional eval workers, with eval duration set to `auto`. Use these additional workers to do evo
#       #   evaluation. Unfortunately, this ongoing evolution happens inside a `while Ture` loop inside the evaluate
#       #   function, but each time, duration_fn is called, so maybe we can smuggle in the world-evolution logic there?
#       #   Alternatively, we could implement world-evolution inside each worker's `sample()` function. This would 
#       #   additionally parallelize world-mutation.
#       print('evaluating')

#       def env_evo_duration_fn(*args, **kwargs):
#           # TODO: this function gets called multiple times in quick succession without additional episodes (saving 
#           #   results for the next time around, basically). Quick fix is to make sure number of train workers and 
#           #   eval duration are properly aligned so as to avoid this.
#           units_remaining = duration_fn(*args, **kwargs)

#           if units_remaining > 0:
#               self.world_evo()

#           return units_remaining

#       return super().evaluate(duration_fn=env_evo_duration_fn)

    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        cfg = ppo.PPOTrainer.get_default_config()
        cfg["evo_eval_duration"] = "auto"
        cfg["evo_eval_config"] = {}

        return cfg

    def set_attrs(self, world_evolver, idx_counter, colearning_config):
        self.world_evolver = world_evolver
        self.idx_counter = idx_counter
        self.colearning_config = colearning_config

    def setup(self, config: PartialTrainerConfigDict):
        super().setup(config)
        self.world_evolver = None
        # Update with evaluation settings:
        user_evo_eval_config = copy.deepcopy(self.config["evo_eval_config"])

        # Merge user-provided eval config with the base config. This makes sure
        # the eval config is always complete, no matter whether we have eval
        # workers or perform evaluation on the (non-eval) local worker.
        evo_eval_config = merge_dicts(self.config, user_evo_eval_config)

        # Create a separate evolution evaluation worker set for evo eval.
        # If num_workers=0, use the evo eval set's local
        # worker for evaluation, otherwise, use its remote workers
        # (parallelized evaluation).
        self.evo_eval_workers: WorkerSet = self._make_workers(
            env_creator=self.env_creator,
            validate_env=None,
            policy_class=self.get_default_policy_class(self.config),
            config=evo_eval_config,
            num_workers=self.config["num_workers"],
            # Don't even create a local worker if num_workers > 0.
            local_worker=False,
            )

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

        def auto_duration_fn(unit, num_workers, cfg, num_units_done):
            # Training is done and we already ran at least one
            # evaluation -> Nothing left to run.
            if num_units_done > 0 and \
                    train_future.done():
                return 0
            # Count by episodes. -> Run n more
            # (n=num eval workers).
            return num_workers * self.config["num_envs_per_worker"]

        step_results = {}

        # Kick off evaluation-loop (and parallel train() call,
        # if requested).
        # Parallel eval + training.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            train_future = executor.submit(
                lambda: self._exec_plan_or_training_iteration_fn())
            print('Training')

            step_results.update(self.evaluate())

            # Run evo-eval indefinitely
            step_results.update(
                self.evo_eval(
                    duration_fn=functools.partial(
                        auto_duration_fn, unit="episodes", num_workers=self.config[
                            "num_workers"], cfg=self.config[
                                "evo_eval_config"])))

        return step_results



    def evo_eval(
            self,
            duration_fn: Optional[Callable[[int], int]] = None,
    ) -> dict:
        """Evaluates current policy under `evaluation_config` settings.

        Note that this default implementation does not do anything beyond
        merging evaluation_config with the normal trainer config.

        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
        # Sync weights to the evaluation WorkerSet.
        self.evo_eval_workers.sync_weights(
            from_worker=self.workers.local_worker())
        self._sync_filters_if_needed(self.evo_eval_workers)

        # How many episodes/timesteps do we need to run?
        # In "auto" mode (only for parallel eval + training): Run as long
        # as training lasts.
        unit = "episodes"
        evo_eval_cfg = self.config["evo_eval_config"]
        # rollout = evo_eval_cfg["rollout_fragment_length"]
        num_envs = self.config["num_envs_per_worker"]
        duration = self.config["num_workers"] * self.config["num_envs_per_worker"]
#       duration = self.config["evo_eval_duration"] if \
#           self.config["evo_eval_duration"] != "auto" else \
#           (self.config["num_workers"] or 1) * \
#           (1 if unit == "episodes" else rollout)
        num_ts_run = 0

        # Default done-function returns True, whenever num episodes
        # have been completed.
        if duration_fn is None:

            def duration_fn(num_units_done):
                return duration - num_units_done

        metrics = {}

        # How many episodes have we run (across all eval workers)?
        num_units_done = 0
        round_ = 0
        while True:
            offspring = self.world_evolver.generate_offspring()
            self.set_worlds(offspring, self.evo_eval_workers)

            units_left_to_do = duration_fn(num_units_done=num_units_done)
            if units_left_to_do <= 0:
                break

            round_ += 1
            batches = ray.get([
                w.sample.remote() for i, w in enumerate(
                    self.evo_eval_workers.remote_workers())
                if i * 1 < units_left_to_do
            ])
            print(f"Round {round_}.")
            # 1 episode per returned batch.
            if unit == "episodes":
                num_units_done += len(batches)
            # n timesteps per returned batch.
            else:
                ts = sum(len(b) for b in batches)
                num_ts_run += ts
                num_units_done += ts

#       if metrics is None:
#           metrics = collect_metrics(
#               self.evaluation_workers.local_worker(),
#               self.evaluation_workers.remote_workers())
        metrics["timesteps_this_iter"] = num_ts_run

        # Evaluation does not run for every step.
        # Save evaluation metrics on trainer, so it can be attached to
        # subsequent step results as latest evaluation result.
        # self.evo_eval_metrics = {"evaluation": metrics}

        # Also return the results here for convenience.
        return self.evaluation_metrics

    def set_worlds(self, worlds: dict, workers: WorkerSet):
        """Assign worlds to environments to be loaded at next reset."""
        idxs = np.random.permutation(list(worlds.keys()))
        world_gen_sequences = {k: world.gen_sequence for k, world in worlds.items() if hasattr(world, 'gen_sequence')} \
            if self.config["render_env"] else None

        worlds = {k: np.array(world.discrete) for k, world in worlds.items()}

        # Have to get hashes on remote workers. Objects have different hashes in "envs" above.
        hashes = workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: hash(env)))
        hashes = [h for wh in hashes for h in wh]
        n_envs = len(hashes)
        print(f"{n_envs} environments, {len(idxs)} worlds")

        self.idx_counter.set_idxs.remote(idxs)
        self.idx_counter.set_hashes.remote(hashes)

        # FIXME: Sometimes hash-to-idx dict is not set by the above call?
        assert ray.get(self.idx_counter.scratch.remote())

        # Otherwise we'll try to serialize `self` in the below lambda
        idx_counter = ray.get_actor("idx_counter")

        # Assign envs to worlds
        workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.queue_worlds(worlds=worlds, idx_counter=idx_counter, world_gen_sequences=world_gen_sequences)))

        # If using oracle, manually load the world
        if self.colearning_config.oracle_policy:
            flood_model = self.get_policy('policy_0').model
            workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env.reset()))

            # Hardcoded rendering
            envs = workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env))
            envs = [e for we in envs for e in we]
            envs[0].render()

            new_world_stats = workers.foreach_worker(
                lambda worker: worker.foreach_env(
                    # NOTE: need to match what is returned by env in get_world_stats here
                    #  i.e., [(world_key, ((qd_objective,), (qd_features...)), [swarm_rewards...]), ...]
                    lambda env: [(env.world_key, ((flood_model.get_solution_length(th.Tensor(env.world).unsqueeze(0)),), (0,0)), [])]))