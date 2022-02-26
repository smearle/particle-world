
exp_names = [
    'test',
    # 0,
]

gen_play_phase_lens = [
    (1, 1),
    # (10, 1),
    (10, 10),
    (-1, 10),
]

qd_objectives = [
    # (False, 'min_solvable'),
    (False, 'contrastive'),
    (True, None),
]

# gen_phase_lens = [
#     1,
#     10,
#     # 50,
#     # 100,
#     # -1,
# ]
# 
# play_phase_lens = [
#     1,
#     10,
#     # 50,
#     # 100,
#     # -1,
# ]
# 
# quality_diversities = [
#     False,
#     # True,
# ]
# 
# objectives = [
#     # 'min_solvable',
#     'contrastive',
# ]