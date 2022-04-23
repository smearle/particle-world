
import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ec', '--environment_class', type=str, default="ParticleMazeEnv", help="Which environment to "
    "use (one of ParticleMazeEnv, TouchStone).")
    parser.add_argument('-lo', '--load', action='store_true')
    parser.add_argument('-ovr', '--overwrite', action='store_true')
    parser.add_argument('-en', '--enjoy', action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="If reloading an experiment, produce plots, etc. "
                                "to visualize progress.")
    # parser.add_argument('-seq', '--sequential', help='not parallel', action='store_true')
    # parser.add_argument('--max_total_bins', type=int, default=1, help="Maximum number of bins in the grid")
    parser.add_argument('-p', '--parallelismType', type=str, default='None',
                        help="Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default='./runs', help="Path of the output log files")
    # parser.add_argument('-li', '--loadIteration', default=-1, type=int)
    parser.add_argument('-a', '--algo', default='me')
    parser.add_argument('-exp', '--exp_name', default='test')
    parser.add_argument('-fw', '--fixed_worlds', action="store_true", help="When true, train players on fixed worlds, "
                                                                            "skipping the world-generation phase.")
    parser.add_argument('-g', '--generator_class', type=str, default="TileFlipIndividual",
                        help="An evolvable representation of the environment (or environment-generator(?))."
                        )
    parser.add_argument('-r', '--render', action='store_true', help="Render the environment (even during training).")
    parser.add_argument('-ne', '--n_envs_per_worker', type=int, default=40)
    parser.add_argument('-new', '--n_evo_workers', type=int, default=4, 
                        help="Number of RLlib workers. Each uses 1 CPU core. When this is 0, we run a single, local, "
                        "process.")
    parser.add_argument('-ntw', '--n_train_workers', type=int, default=4,)    
    parser.add_argument('-gpus', '--num_gpus', type=int, default=0, help="How many GPUs to use for training.")
    parser.add_argument('-ev', '--evaluate', action='store_true', help="Whether to evaluate trained agents/worlds and"
                                                                        "collect relevant stats.")
    parser.add_argument('-qd', '--quality_diversity', action='store_true',
                        help='Search for a grid of levels with dimensions (measures) given by the fitness of distinct '
                                'policies, and objective score given by the inverse fitness of an additional policy.')
    parser.add_argument('-obj', '--objective_function', type=str, default='min_solvable',
                        help='If not using quality diversity, the name of the fitness function that will compute world'
                                'fitness based on population-wise rewards.')
    parser.add_argument('-n_pol', '--n_policies', type=int, default=1, help="How many distinct policies to train.")
    parser.add_argument('-op', '--oracle_policy', action='store_true', help="Whether to use the oracle (optimal) policy, and"
                                                                        "thereby focus on validating generator-evolution.")
    parser.add_argument('-fo', '--fully_observable', action='store_true',
                        help="Whether to use a fully observable environment.")
    parser.add_argument('-gp', '--gen_phase_len', type=int, default=-1,
                        help="How many generations to evolve worlds (generator). If -1, run until convergence.")
    parser.add_argument('-pp', '--play_phase_len', type=int, default=1, 
                        help="How many iterations to train the player. If -1, run until convergence.")
    parser.add_argument('-m', '--model', type=str, default='rnn')
    parser.add_argument('-fov', '--field_of_view', type=int, default=2, help='How far agents can see in each direction.')
    parser.add_argument('-tr', '--target_reward', type=int, default=0, 
                        help="Target reward the world should elicit from player if using the min_solvable objective "
                        "function.")
    parser.add_argument('-ro', '--rotated_observations', action='store_true',
                        help="Whether to use rotated observations. If so, the agent will have an action space consisting"
                        "of moving forward, staying put, and turning left or right. It will perceive its orientation as a "
                        "discrete space.")
#   parser.add_argument('-to', '--translated_observations', action='store_true', help='Whether to use translated '
#                       ' observations. This will always be the True when fully_observable is False.')
    parser.add_argument("-gaw", "--gen_adversarial_worlds", action="store_true", help="Whether to generate worlds with "
                        "min_solvable objective function until convergence using trained policy, to test the extent of "
                        "the policy's abilities.")
    parser.add_argument('-lc', '--load_config', type=str, default=None, 
                        help="Load a dictionary of (automatically-generated) arguments. "
                        "NOTE: THIS OVERWRITES ALL OTHER ARGUMENTS AVAILABLE IN THE COMMAND LINE.")

    return parser