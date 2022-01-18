# Each policy can have a different configuration (including custom model).
import copy
import math

from ray.rllib.agents.ppo import ppo
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from env import ParticleGymRLlib, gen_policy


def init_particle_trainer(env, num_rllib_workers, num_rllib_envs):
    # env is currently a dummy environment that will not be used in actual training
    MODEL_CONFIG = copy.copy(MODEL_DEFAULTS)
    MODEL_CONFIG.update({
        # "use_lstm": True,
        # "fcnet_hiddens": [32, 32],
    })
    workers = 1 if num_rllib_workers == 0 else num_rllib_workers

    trainer_config = {
        "multiagent": {
            "policies": {f'policy_{i}': gen_policy(i, env.observation_spaces[i], env.action_spaces[i], env.fovs[i])
                         for i, swarm in enumerate(env.swarms)},
            # the first tuple value is None -> uses default policy
            # "car1": (None, particle_obs_space, particle_act_space, {"gamma": 0.85}),
            # "car2": (None, particle_obs_space, particle_act_space, {"gamma": 0.99}),
            # "traffic_light": (None, tl_obs_space, tl_act_space, {}),
            "policy_mapping_fn":
                lambda agent_id: f'policy_{agent_id[0]}',
        },
        "model": MODEL_CONFIG,
        # {
        # "custom_model": RLlibNN,
        # },
        "env_config": {
            "width": env.width,
            "n_policies": len(env.swarms),
            "n_pop": env.swarms[0].n_pop,
            "max_steps": env.max_steps,
            # "pg_width": pg_width,
        },
        "num_gpus": 0,
        "num_workers": num_rllib_workers,
        "num_envs_per_worker": math.ceil(num_rllib_envs / workers),
        "framework": "torch",
        "render_env": True,
        # "evaluation_interval": 10,
        # "explore": False,
        # "lr": 0.1,
        # "log_level": "INFO",
        # "record_env": True,
        "rollout_fragment_length": env.max_steps,
        "train_batch_size": env.max_steps * num_rllib_envs,
    }
    trainer = ppo.PPOTrainer(env=ParticleGymRLlib, config=trainer_config)
    return trainer


