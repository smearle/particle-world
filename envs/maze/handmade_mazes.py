import itertools
import numpy as np

from utils import discrete_to_onehot


eval_mazes = {
    # Empty room---easy (goal next to spawn)
    'empty_easy':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    # Empty room---hard (goal and spawn in opposite corners
    'empty_hard':
        np.array([
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        ]),
    # SixteenRooms
    'sixteen_rooms':
        np.array([
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]),
    # Zig-Zag
    'zigzag':
        np.array([
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 3],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        ]),
    # Labyrinth (from REPAIRED paper)
    'labyrinth':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 3, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
            [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]),
}

empty = 0
wall = 1
start = 2
goal = 3


# Test mazes with goals in alternating corners. Will agent learn to check each corner?
full_obs_test_mazes = {}

w = 13
core_layout = np.ones((w, w), dtype=np.int8) * wall
core_layout[w//2,:] = core_layout[:,w//2] = empty
core_layout[-1,1:-1] = core_layout[0,1:-1] = core_layout[1:-1,-1] = core_layout[1:-1,0] = empty
core_layout[w//2, w//2] = start

for x, y in [(0, 1), (0, -2), (-1, 1), (-1, -2)]:
    cross_xy = core_layout.copy()
    cross_xy[x, y] = goal
    full_obs_test_mazes['cross_' + str(x) + '_' + str(y)] = cross_xy
    cross_yx = core_layout.copy()
    cross_yx[y, x] = goal
    full_obs_test_mazes['cross_' + str(y) + '_' + str(x)] = cross_yx

eval_mazes.update(full_obs_test_mazes)


# Convert to 3-channel probability distribution (or agent action)-type representation
eval_mazes_probdists = {}
for k, y in eval_mazes.items():
    y = discrete_to_onehot(y, n_chan=4)
    z = np.empty((y.shape[0] - 1, y.shape[1], y.shape[2]))
    z[:3] = y[:3]
    z[2] -= y[3]
    eval_mazes_probdists[k] = z

eval_mazes_onehots = {}
for k, y in eval_mazes.items():
    eval_mazes_onehots[k] = discrete_to_onehot(y, n_chan=4)


# Simple test mazes with goal on either end of hallway. Can agent learn to check both ends?
partial_obs_test_mazes = {}
core_layout = np.ones((w, w), dtype=np.int8) * wall
core_layout[:, w//2] = empty
core_layout[w//2, w//2] = start

for x, y in [(0, w//2), (w-1, w//2)]:
    partial_obs_test_mazes['partial_' + str(x) + '_' + str(y)] = core_layout.copy()
    partial_obs_test_mazes['partial_' + str(x) + '_' + str(y)][x, y] = goal


partial_obs_test_mazes_2 = {}
core_layout = np.ones((w, w), dtype=np.int8) * wall
core_layout[:, w//2] = empty
core_layout[0, :] = core_layout[w-1, :] = empty
core_layout[w//2, w//2] = start

for x, y in [(0, 0), (w-1, 0), (0, w-1), (w-1, w-1)]:
    partial_obs_test_mazes_2['partial_' + str(x) + '_' + str(y)] = core_layout.copy()
    partial_obs_test_mazes_2['partial_' + str(x) + '_' + str(y)][x, y] = goal



ghost_action_test_maze = {'ghost_action_test_maze':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
}