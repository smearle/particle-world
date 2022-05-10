from pdb import set_trace as TT
import time

import numpy as np

from envs.minihack.env import MiniHackEvoLevel


if __name__ == "__main__":

    # level = LevelGenerator(w=5, h=5, lit=True, flags=("premapped",))
    # env = gym.make("MiniHack-River-v0")
    # env.reset() # each reset generates a new environment instance
    # env.step(1)  # move agent '@' north
    # env.render()

    env = MiniHackEvoLevel()
    for _ in range(100):
        world = np.random.randint(0, 2, (env.width, env.height))
        world[2,2] = env.start_chan
        world[-1,-1] = env.goal_chan
        env.set_world(world=world)

        env.reset()
        # for _ in range(10):
            # env.step_adversary(env.adversary_action_space.sample())
        # env.compile_env()

        for _ in range(10000):
            action = env.action_space.sample()
            obs = env.step(action)
            env.render()