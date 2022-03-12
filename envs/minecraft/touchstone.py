from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_specs.obtain_specs import Obtain
from pdb import set_trace as TT
import numpy as np
from typing import List

TOUCHSTONE_DOC = """
In TouchStone, the agent must touch stone.
"""

TOUCHSTONE_LENGTH = 8000



def generate_draw_cuboid_string(x1, y1, z1, x2, y2, z2, type_int, block_types):
    """ Generates a string that can be used to draw a cuboid of the specified type. 
    """
    type_str = block_types[type_int]

    return f"""<DrawCuboid x1="{x1}" y1="{y1}" z1="{z1}" x2="{x2}" y2="{y2}" z2="{z2}" type="{type_str}"/>"""


def gen_init_world(width, depth, height, block_types):
    # return np.random.randint(0, len(block_types), size=(width, depth, height))
    return np.ones((width, depth, height), dtype=np.int32)


class TouchStone(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'TouchStone-v0'

        # TODO: more than cubes!
        self.width, self.depth, self.height = 7, 7, 7
        self.block_types = ['stone', 'dirt', 'air']
        self.n_chan = len(self.block_types)
        self.unique_chans = [0]
        self.reward_range = (0, 100)
        self.metadata = None

        self.world_arr = gen_init_world(self.width, self.depth, self.height, self.block_types)

        super(TouchStone, self).__init__(*args,
                    max_episode_steps=TOUCHSTONE_LENGTH,
                    reward_threshold=100.0,
                    **kwargs)


    def create_rewardables(self) -> List[Handler]:
        return [
                   handlers.RewardForTouchingBlockType([
                       {'type': 'stone', 'behaviour': 'onceOnly',
                        'reward': 100.0},
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

#class MLGWB(SimpleEmbodimentEnvSpec):
#    def __init__(self, *args, **kwargs):
#        if 'name' not in kwargs:
#            kwargs['name'] = 'MLGWB-v0'
#
#        self.n_episode = 0
#        super().__init__(*args,
#                    max_episode_steps=MLGWB_LENGTH,
#                    reward_threshold=100.0,
#                    **kwargs)

    def create_server_world_generators(self):
        world_arr = self.world_arr
        world_generators = [
            # Creating flat layers.
            handlers.FlatWorldGenerator(generatorString="1;7,2x3,2;1"),
            # Add drawing decorators for each block specified in world_arr.
            handlers.DrawingDecorator("""\n""".join(
                generate_draw_cuboid_string(x1, y1, z1, x1, y1, z1, world_arr[x1, y1, z1], self.block_types) 
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
            handlers.AgentStartPlacement(-1, 1, -1, 0, 0)
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
            handlers.ObservationFromCurrentLocation(),
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

    def get_docstring(self):
        return TOUCHSTONE_DOC