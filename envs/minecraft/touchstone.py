from pdb import set_trace as TT
from typing import List

from gym import spaces
from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.env_specs.obtain_specs import Obtain
from minerl.herobraine.env_specs.navigate_specs import Navigate
import numpy as np


NAVIGATE_STEPS = 6000


def generate_draw_cuboid_string(x1, y1, z1, x2, y2, z2, type_int, block_types):
    """ Generates a string that can be used to draw a cuboid of the specified type. 
    """
    type_str = block_types[type_int]

    return f"""<DrawCuboid x1="{x1}" y1="{y1}" z1="{z1}" x2="{x2}" y2="{y2}" z2="{z2}" type="{type_str}"/>"""


def gen_init_world(width, depth, height, block_types):
    # return np.random.randint(0, len(block_types), size=(width, depth, height))
    return np.zeros((width, depth, height), dtype=np.int32)


#class TouchStone(SimpleEmbodimentEnvSpec):
#    def __init__(self, *args, **kwargs):
#        # suffix = 'Extreme' if extreme else ''
#        # suffix += 'Dense' if dense else ''
#        # name = 'MineRLNavigate{}-v0'.format(suffix)
#        name = 'TouchStone-v0'
#        # self.dense, self.extreme = dense, extreme
#        self.dense, self.extreme = True, False
#
#        # ProcGen 
#        self.width, self.depth, self.height = 14, 14, 14
#        self.block_types = ['air', 'dirt', 'diamond_block']
#        block_type_chan_idxs = {bt: i for i, bt in enumerate(self.block_types)}
#        unique_block_types = ['diamond_block']
#        self.unique_chans = [block_type_chan_idxs[bt] for bt in unique_block_types]
#        self.need_world_reset = False
#
#        super().__init__(name, *args, max_episode_steps=6000, **kwargs)
#
#    def is_from_folder(self, folder: str) -> bool:
#        return folder == 'navigateextreme' if self.extreme else folder == 'navigate'
#
#    def create_observables(self) -> List[Handler]:
#        return super().create_observables() + [
#            handlers.CompassObservation(angle=True, distance=False),
#            handlers.FlatInventoryObservation(['dirt'])]
#
#    def create_actionables(self) -> List[Handler]:
#        return super().create_actionables() + [
#            handlers.PlaceBlock(['none', 'dirt'],
#                                _other='none', _default='none')]
#
#    # john rl nyu microsoft van roy and ian osband
#
#    def create_rewardables(self) -> List[Handler]:
#        return [
#                   handlers.RewardForTouchingBlockType([
#                       {'type': 'diamond_block', 'behaviour': 'onceOnly',
#                        'reward': 100.0},
#                   ])
#               ] + ([handlers.RewardForDistanceTraveledToCompassTarget(
#            reward_per_block=1.0
#        )] if self.dense else [])
#
#    def create_agent_start(self) -> List[Handler]:
#        return [
#            handlers.SimpleInventoryAgentStart([
#                dict(type='compass', quantity='1')
#            ])
#        ]
#
#    def create_agent_handlers(self) -> List[Handler]:
#        return [
#            handlers.AgentQuitFromTouchingBlockType(
#                ["diamond_block"]
#            )
#        ]
#
#    def create_server_world_generators(self) -> List[Handler]:
#        if self.extreme:
#            return [
#                handlers.BiomeGenerator(
#                    biome=3,
#                    force_reset=True
#                )
#            ]
#        else:
#            return [
#                handlers.DefaultWorldGenerator(
#                    force_reset=True
#                )
#            ]
#
#    def create_server_quit_producers(self) -> List[Handler]:
#        return [
#            handlers.ServerQuitFromTimeUp(NAVIGATE_STEPS * MS_PER_STEP),
#            handlers.ServerQuitWhenAnyAgentFinishes()
#        ]
#
#    def create_server_decorators(self) -> List[Handler]:
#        return [handlers.NavigationDecorator(
#            max_randomized_radius=64,
#            min_randomized_radius=64,
#            block='diamond_block',
#            placement='surface',
#            max_radius=8,
#            min_radius=0,
#            max_randomized_distance=8,
#            min_randomized_distance=0,
#            randomize_compass_location=True
#        )]
#
#    def create_server_initial_conditions(self) -> List[Handler]:
#        return [
#            handlers.TimeInitialCondition(
#                allow_passage_of_time=False,
#                start_time=6000
#            ),
#            handlers.WeatherInitialCondition('clear'),
#            handlers.SpawningInitialCondition('false')
#        ]
#
#    def get_docstring(self):
#        return make_navigate_text(
#            top="normal" if not self.extreme else "extreme",
#            dense=self.dense)
#
#    def determine_success_from_rewards(self, rewards: list) -> bool:
#        reward_threshold = 100.0
#        if self.dense:
#            reward_threshold += 60
#        return sum(rewards) >= reward_threshold
#
#
#def make_navigate_text(top, dense):
#    navigate_text = """
#.. image:: ../assets/navigate{}1.mp4.gif
#    :scale: 100 %
#    :alt: 
#
#.. image:: ../assets/navigate{}2.mp4.gif
#    :scale: 100 %
#    :alt: 
#
#.. image:: ../assets/navigate{}3.mp4.gif
#    :scale: 100 %
#    :alt: 
#
#.. image:: ../assets/navigate{}4.mp4.gif
#    :scale: 100 %
#    :alt: 
#
#In this task, the agent must move to a goal location denoted by a diamond block. This represents a basic primitive used in many tasks throughout Minecraft. In addition to standard observations, the agent has access to a “compass” observation, which points near the goal location, 64 meters from the start location. The goal has a small random horizontal offset from the compass location and may be slightly below surface level. On the goal location is a unique block, so the agent must find the final goal by searching based on local visual features.
#
#The agent is given a sparse reward (+100 upon reaching the goal, at which point the episode terminates). """
#    if dense:
#        navigate_text += "**This variant of the environment is dense reward-shaped where the agent is given a reward every tick for how much closer (or negative reward for farther) the agent gets to the target.**\n"
#    else:
#        navigate_text += "**This variant of the environment is sparse.**\n"
#
#    if top == "normal":
#        navigate_text += "\nIn this environment, the agent spawns on a random survival map.\n"
#        navigate_text = navigate_text.format(*["" for _ in range(4)])
#    else:
#        navigate_text += "\nIn this environment, the agent spawns in an extreme hills biome.\n"
#        navigate_text = navigate_text.format(*["extreme" for _ in range(4)])
#    return navigate_text



TOUCHSTONE_DOC = """
In TouchStone, the agent must touch stone.
"""

# TODO: put this in a config or something!
TOUCHSTONE_LENGTH = 128


class TouchStone(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'TouchStone-v0'

        # The agent only observes its pixel-based point-of-view for now.
        # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # ProcGen / World Evolution
        # TODO: more than cubes!
        self.width, self.depth, self.height = 14, 14, 14
        self.block_types = [
            'air',
            'dirt', 
            'stone', 
            ]
        block_type_chan_idxs = {bt: i for i, bt in enumerate(self.block_types)}
        self.n_chan = len(self.block_types)
        # unique_block_types = ['stone']
        unique_block_types = []
        self.unique_chans = [block_type_chan_idxs[bt] for bt in unique_block_types]
        self.goal_chan = block_type_chan_idxs['stone']
        self.empty_chan = block_type_chan_idxs['air']
        
        self.world_arr = gen_init_world(self.width, self.depth, self.height, self.block_types)

        super(TouchStone, self).__init__(*args,
                    max_episode_steps=TOUCHSTONE_LENGTH,
                    reward_threshold=100.0,
                    **kwargs)


    def create_rewardables(self) -> List[Handler]:
        return [
                   handlers.RewardForTouchingBlockType([
                       {'type': 'stone', 'behaviour': 'onceOnly',
                        'reward': 1},
                   ]),
               ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromTouchingBlockType(
                ["stone"]
            )
        ]

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'touchstone'

    def create_server_world_generators(self):
        world_arr = self.world_arr
        world_generators = [

            # Creating flat layers.
            handlers.FlatWorldGenerator(generatorString="1;7,2x3,2;1"),

            # TODO: add a wall around the play area.

            # Add drawing decorators for each block specified in world_arr.
            handlers.DrawingDecorator("""\n""".join(
                generate_draw_cuboid_string(x1, y1+4, z1, x1, y1+4, z1, world_arr[x1, y1, z1], self.block_types) 
                for x1 in range(world_arr.shape[0]) for y1 in range(world_arr.shape[1]) for z1 in range(world_arr.shape[2])) 
            )

        ]
        # print(f'world generators: {world_generators}')
        return world_generators

    def create_agent_start(self) -> List[Handler]:
        return [
            # make the agent start with these items
            handlers.SimpleInventoryAgentStart([
                # dict(type="water_bucket", quantity=1),
                # dict(type="diamond_pickaxe", quantity=1)
            ]),
            # make the agent start 90 blocks high in the air
            handlers.AgentStartPlacement(0, 5, 0, 0, 0)
        ]

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            # allow agent to place water
            # handlers.KeybasedCommandAction("use"),
            # also allow it to equip the pickaxe
            # handlers.EquipAction(["diamond_pickaxe"])
        ]
    
    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            # current location and lifestats are returned as additional
            # observations
            # handlers.ObservationFromCurrentLocation(),
            # handlers.ObservationFromLifeStats()
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            # Sets time to morning and stops passing of time
            handlers.TimeInitialCondition(False, 23000)
        ] 

    # see API reference for use cases of these first two functions

    def create_server_quit_producers(self):
        return []

    def create_server_decorators(self) -> List[Handler]:
        return []

    # the episode can terminate when this is True
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold
#       return False

    def get_docstring(self):
        return TOUCHSTONE_DOC