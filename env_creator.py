import copy
from pdb import set_trace as TT

import gym
from minerl.herobraine.env_spec import EnvSpec
from envs.minecraft.touchstone import TouchStone

from envs.wrappers import MineRLWrapper, WorldEvolutionWrapper


def make_env(environment_class, env_config):
    TT()
    env_config = copy.copy(env_config)

    if isinstance(environment_class, EnvSpec):
        touchstone = TouchStone()
        touchstone.register()
        env = gym.make("TouchStone-v0")
        env =  MineRLWrapper(env)
    else:
        env = environment_class(env_config)

    env = WorldEvolutionWrapper(env)

    return env
    