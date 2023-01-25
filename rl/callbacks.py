from enum import unique
from pdb import set_trace as TT
from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID


class WorldEvoCallbacks(DefaultCallbacks):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # print(f"Regret objective: {regret_objective}")
        self.regret_objective = cfg.objective_function == "regret"
        self.n_policies = cfg.n_policies
        self.quality_diversity = cfg.quality_diversity

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                            policies: Dict[str, Policy], episode: Episode,
                            env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"
        # print("episode {} (env-idx={}) started.".format(
            # episode.episode_id, env_index))
        # episode.user_data["pole_angles"] = []
        # episode.hist_data["pole_angles"] = []
        # if env_index == 0:
            # TT()
        env = base_env.envs[env_index]
        env.reset_stats()

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy], episode: Episode,
                        env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        # pole_angle = abs(episode.last_observation_for()[2])
        # raw_angle = abs(episode.last_raw_obs_for()[2])
        # assert pole_angle == raw_angle
        # episode.user_data["pole_angles"].append(pole_angle)

# TODO: get world stats using this callback!
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["policy_0"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        env = base_env.envs[env_index]
        episode_stats = env.stats

        # Check for invalid episode length before preprocessing.
        # assert len(episode_stats) == 1
        # This dirty workaround is necessary for evaluation. Not sure why.
        if len(episode_stats) == 0:
            assert env.evaluate
            return

        episode_stats = episode_stats[0]

        # If you want to add some of these stats, make sure you don't duplicate keys from `world_stats`.
        # episode.hist_data.update({k: [v] for k, v in episode_stats.items()})

        # Here we assume we have forced the env to reload by queuing worlds, though we should remove this hack 
        # altogether eventually.
        if episode_stats['n_steps'] < 1:
            env.stats = []
            return

        world_stats = env.get_world_stats(quality_diversity=self.quality_diversity)
        # TODO: remove flusing behavior from get_world_stats and do it here
        env.stats = []
        assert len(world_stats) == 1
        world_stats = world_stats[0]

        # TODO: flatten QD stats (reshape them outside), so that it will be written to tensorboard automatically.
        episode.hist_data.update({k: [v] for k, v in world_stats.items()})

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: Episode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
        """Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Args:
            worker: Reference to the current rollout worker.
            episode: Episode object.
            agent_id: Id of the current agent.
            policy_id: Id of the current policy for the agent.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default_policy".
            postprocessed_batch: The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches: Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
#       if not self.regret_objective:
#           return

#       advantages = postprocessed_batch["advantages"]
#       pos_val_loss = np.mean(np.abs(advantages))
##      value_targets = postprocessed_batch["value_targets"]
##      vf_preds = postprocessed_batch["vf_preds"]
##      vf_preds
##      td_errors = value_targets - vf_preds
##      policy = worker.get_policy(policy_id)
##      gamma = policy.config["gamma"]  # MDP discount factor
##      gae_lambda = policy.config["lambda"]  # Generalize Advantage Estimation discount factor, lambda
 #      infos = postprocessed_batch["infos"]

 #      # TODO: should be able to remove this check safely
 #      unique_world_keys = np.unique([infos[t]["world_key"] for t in range(len(infos))])
 #      assert len(unique_world_keys) == 1, "Our assumption that each trajectory corresponds to a single world has been violated."

 #      

##      # TODO: vectorize this
##      pos_val_loss_sum = 0
##      for t in range(advantages.shape):
##          pos_val_loss_t = 0
##          for k in range(t, advantages.shape[0]):
##              # TODO: value target is seemingly not the same as TD target here: https://en.wikipedia.org/wiki/Temporal_difference_learning
##              #  , so what is it?
##              (gamma * gae_lambda) ** (k - t) * td_errors[k]
##          pos_val_loss_sum += max(pos_val_loss_t, 0)
##      pos_val_loss = np.mean(pos_val_loss_sum)
 #      pos_val_losses = {unique_world_keys[0]: pos_val_loss}
 #      worker.foreach_env(lambda env: env.set_regret_loss(pos_val_losses))

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        """Compute a proxy for the "regret" of each policy, i.e. its distance from optimal performance.
        
        This is achieved by computing positive value loss, as in ACCEL (https://arxiv.org/pdf/2203.01302.pdf) and PLR 
        (https://arxiv.org/pdf/2010.03934.pdf)"""

        # Skipping training and eval worlds. Note also that mutating the samples as at the end of this function seems
        # to be illegal.
        if not worker.foreach_env(lambda env: env.evo_eval_world)[0]:
            return

        if not self.regret_objective:
            # print(f"Regret objective is not enabled on worker {worker}.")
            return
        # print(f"Regret objective enabled on worker {worker}.")

        if hasattr(samples, 'policy_batches'):
            # TODO: collect regret scores of multiple policies
            pol_batches = [samples.policy_batches[k] for k in samples.policy_batches]
        else:
            pol_batches = [samples]
        
        for pol_batch in pol_batches:

            pos_val_losses = {}
    #       value_targets = pol_batch['value_targets']
    #       vf_preds = pol_batch['vf_preds']
            advantages = pol_batch['advantages']
            world_keys = np.array([info['world_key'] for info in pol_batch['infos']])
            unique_keys = set(world_keys)

            for wk in unique_keys:
                idxs = np.argwhere(world_keys == wk)
    #           w_val_trgs, w_vf_preds = value_targets[idxs], vf_preds[idxs]

    #           # Note that below is wrong: we needed to sum, then clip, then average
    #           pos_val_loss = np.mean(np.clip(w_val_trgs - w_vf_preds, 0, None))

                w_advantages = advantages[idxs]
                pos_val_losses[wk] = np.mean(np.clip(np.sum(w_advantages), 0, None))

                # Wait, no, I think we take the average magnitude of the GAE?
                # pv = np.mean(np.abs(w_advantages))

                # pos_val_losses[wk] = pv

            pol_batch['pos_val_loss'] = pos_val_losses

            # print(f"Regret objective is {pos_val_losses}. Passing to worker environments.")

        # TODO: should be able to pass this info back more naturally somehow, as below
        # FIXME: This breaks training batches.

        # For now, just giving it back to environments to be collected later
        # TT()
        # worker.foreach_env(lambda env: env.set_regret_loss(pos_val_losses))

