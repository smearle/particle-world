from pdb import set_trace as TT
import time

import numpy as np

from envs.minihack.env import MiniHackEvoWorld


if __name__ == "__main__":

    # level = LevelGenerator(w=5, h=5, lit=True, flags=("premapped",))
    # env = gym.make("MiniHack-River-v0")
    # env.reset() # each reset generates a new environment instance
    # env.step(1)  # move agent '@' north
    # env.render()

    env = MiniHackEvoWorld()
    for _ in range(100):
        width, height = env.width, env.height
        # Number of free tile types for placement.
        n_tiles = 2
        # Generate random map with different weights for each tile type.
        weights = np.random.random(n_tiles)
        # Normalize weights.
        weights /= np.sum(weights)
        world = np.random.choice(n_tiles, (width, height), p=weights)
        world[2,2] = env.start_chan
        world[-1,-1] = env.goal_chan
        env.set_world(world=world)

        env.reset()
        # for _ in range(10):
            # env.step_adversary(env.adversary_action_space.sample())
        # env.compile_env()

        done = False
        for _ in range(10000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            print(obs['chars'][21//2-width//2:21//2+width//2, 79//2-height//2:79//2+height//2])
            if done:
                break