""" Experimenting with possible co-learning loop using minerl. We evolve some environments (array-based onehot block
representations), evaluating these based on RL player performance, then train the RL player, and repeat.
"""
from pdb import set_trace as TT
from timeit import default_timer as timer

import gym
import minerl.herobraine.hero.handlers as handlers
import numpy as np
from envs.minecraft.touchstone import MLGWB

# In order to use the environment as a gym you need to register it with gym
abs_MLG = MLGWB()
abs_MLG.register()
env = gym.make("MLGWB-v0")
width, depth, height = 7, 7, 7

block_types = ["air", "stone"]

def gen_rand_world():

    return np.random.randint(0, len(block_types), size=(width, depth, height))


def generate_draw_cuboid_string(x1, y1, z1, x2, y2, z2, type_int):
    """ Generates a string that can be used to draw a cuboid of the specified type. 
    """
    type_str = block_types[type_int]

    return f"""<DrawCuboid x1="{x1}" y1="{y1+10}" z1="{z1}" x2="{x2}" y2="{y2+10}" z2="{z2}" type="{type_str}"/>"""


def generate_new_create_server_world_generators_fn(world_arr):

    def create_server_world_generators(self):
        world_generators = [
            # Creating flat layers.
            handlers.FlatWorldGenerator(generatorString="1;7,2x3,2;1"),
            # Add drawing decorators for each block specified in world_arr.
            handlers.DrawingDecorator("""\n""".join(
                generate_draw_cuboid_string(x1, y1, z1, x1, y1, z1, world_arr[x1, y1, z1]) 
                for x1 in world_arr.shape[0] for y1 in world_arr.shape[1] for z1 in world_arr.shape[2]) 
            )
        ]
        print(world_generators)
        return world_generators

    return create_server_world_generators


for i in range(10):

    # TODO: edit the environment without having to reset it. But then again, won't we need to reset it for the player AI?
    #  So it must be that the overhead here is bearable to folks training RL agents, at least.

    # Redefine this env method so that we generate a new environment on reset.
    env.create_server_world_generators = generate_new_create_server_world_generators_fn(gen_rand_world())

    # this line might take a couple minutes to run. I sure hope not though!
    start_time = timer()
    obs  = env.reset()
    print(f'Time to reset: {timer() - start_time}.')

    # Renders the environment with the agent taking noops
    done = False
    while not done:
        env.render()
        # a dictionary of actions. Try indexing it and changing values.
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)