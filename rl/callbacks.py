from enum import unique
from pdb import set_trace as TT
from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID


class RegretCallbacks(DefaultCallbacks):
    def __init__(self, *args, regret_objective=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.regret_objective = regret_objective

#   def on_episode_step(
#       self,
#       *,
#       worker: RolloutWorker,
#       base_env: BaseEnv,
#       policies: Dict[str, Policy],
#       episode: Episode,
#       env_index: int,
#       **kwargs
#   ):
#       pass

#   def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
#                      policies: Dict[PolicyID, Policy], episode: Episode,
#                      **kwargs) -> None:
#       """Runs when an episode is done.

#       Args:
#           worker: Reference to the current rollout worker.
#           base_env: BaseEnv running the episode. The underlying
#               sub environment objects can be retrieved by calling
#               `base_env.get_sub_environments()`.
#           policies: Mapping of policy id to policy
#               objects. In single agent mode there will only be a single
#               "default_policy".
#           episode: Episode object which contains episode
#               state. You can use the `episode.user_data` dict to store
#               temporary data, and `episode.custom_metrics` to store custom
#               metrics for the episode.
#           kwargs: Forward compatibility placeholder.
#       """
#       pass

#   def on_postprocess_trajectory(
#           self, *, worker: "RolloutWorker", episode: Episode,
#           agent_id: AgentID, policy_id: PolicyID,
#           policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
#           original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
#       """Called immediately after a policy's postprocess_fn is called.

#       You can use this callback to do additional postprocessing for a policy,
#       including looking at the trajectory data of other agents in multi-agent
#       settings.

#       Args:
#           worker: Reference to the current rollout worker.
#           episode: Episode object.
#           agent_id: Id of the current agent.
#           policy_id: Id of the current policy for the agent.
#           policies: Mapping of policy id to policy objects. In single
#               agent mode there will only be a single "default_policy".
#           postprocessed_batch: The postprocessed sample batch
#               for this agent. You can mutate this object to apply your own
#               trajectory postprocessing.
#           original_batches: Mapping of agents to their unpostprocessed
#               trajectory data. You should not mutate this object.
#           kwargs: Forward compatibility placeholder.
#       """
#       pass

 #      advantages = postprocessed_batch["advantages"]
 #      pos_val_loss = np.mean(np.abs(advantages))
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
        if not self.regret_objective:
            return

        if hasattr(samples, 'policy_batches'):
            # TODO: collect regret scores of multiple policies
            assert len(samples.policy_batches) == 1, "Regret objective only valid for single-policy scenario."
            pol_batch = samples.policy_batches['policy_0']
        else:
            pol_batch = samples

#       value_targets = pol_batch['value_targets']
#       vf_preds = pol_batch['vf_preds']
        advantages = pol_batch['advantages']
        world_keys = np.array([info['world_key'] for info in pol_batch['infos']])
        unique_keys = set(world_keys)
        pos_val_losses = {}


        for wk in unique_keys:
            idxs = np.argwhere(world_keys == wk)
#           w_val_trgs, w_vf_preds = value_targets[idxs], vf_preds[idxs]

#           # Note that below is wrong: we needed to sum, then clip, then average
#           pos_val_loss = np.mean(np.clip(w_val_trgs - w_vf_preds, 0, None))

            w_advantages = advantages[idxs]
            pos_val_losses[wk] = np.mean(np.clip(np.sum(w_advantages), 0, None))

        # TODO: should be able to pass this info back more naturally somehow, as below
        # pol_batch['pos_val_losses'] = pos_val_losses

        # For now, just giving it back to environments to be collected later
        worker.foreach_env(lambda env: env.set_regret_loss(pos_val_losses))

#   def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
#                         result: dict, **kwargs) -> None:
#       """Called at the beginning of Policy.learn_on_batch().

#       Note: This is called before 0-padding via
#       `pad_batch_to_sequences_of_same_size`.

#       Also note, SampleBatch.INFOS column will not be available on
#       train_batch within this callback if framework is tf1, due to
#       the fact that tf1 static graph would mistake it as part of the
#       input dict if present.
#       It is available though, for tf2 and torch frameworks.

#       Args:
#           policy: Reference to the current Policy object.
#           train_batch: SampleBatch to be trained on. You can
#               mutate this object to modify the samples generated.
#           result: A results dict to add custom metrics to.
#           kwargs: Forward compatibility placeholder.
#       """

#       pass

