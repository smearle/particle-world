# Each policy can have a different configuration (including custom model).
import copy

from ray.rllib.agents.ppo import ppo
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from env import ParticleGymRLlib


def gen_policy(i, observation_space, action_space, fov):
    config = {
        "model": {
            "custom_model_config": {
                "fov": fov,
            }
        }
    }
    return PolicySpec(config=config, observation_space=observation_space, action_space=action_space)


def init_particle_trainer(env):
    # env is currently a dummy environment that will not be used in actual training
    MODEL_CONFIG = copy.copy(MODEL_DEFAULTS)
    MODEL_CONFIG.update({
        "use_lstm": True,
    })
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
        "num_gpus": 1,
        # "num_workers": 12,
        # "num_envs_per_worker": 2,
        "framework": "torch",
        "render_env": True,
        # "explore": False,
        # "lr": 0.1,
        # "train_batch_size": 500,
        # "log_level": "INFO",
    }
    trainer = ppo.PPOTrainer(env=ParticleGymRLlib, config=trainer_config)
    return trainer


