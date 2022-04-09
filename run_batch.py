"""Run a batch of experiments, either in sequence on a local machine, or in parallel on a SLURM cluster.

The `--batch_config` command will point to a set of hyperparemeters, which are specified in a JSON file in the `configs`
directory.
"""
import argparse
from collections import namedtuple
from itertools import product
import json
import os
import re
import yaml

from cross_eval import vis_cross_eval
from utils import get_experiment_name


def launch_job(sbatch_file, exp_i, job_time, job_cpus, local):
    cmd = f'python main.py --load_config {exp_i}'

    if local:
        print(f'Launching command locally:\n{cmd}')
        os.system(cmd)

    else:
        with open(sbatch_file) as f:
            content = f.read()
            job_name = 'prtcl_'
            # if args.evaluate:
                # job_name += 'eval_'
            job_name += str(exp_i)
            content = re.sub('prtcl_(eval_)?\d+', job_name, content)
            content = re.sub('#SBATCH --time=\d+:', '#SBATCH --time={}:'.format(job_time), content)
            content = re.sub('#SBATCH --cpus-per-task=\d+:', '#SBATCH --cpus-per-task={}:'.format(job_cpus), content)
            cmd = '\n' + cmd
            new_content = re.sub('\n.*python main.py.*', cmd, content)

        with open(sbatch_file, 'w') as f:
            f.write(new_content)

        os.system('sbatch {}'.format(sbatch_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lc', '--local', action='store_true', 
                        help='Run batch of jobs locally, in sequence. (Otherwise submit parallel jobs to SLURM.)')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-gpus', '--num_gpus', type=int, default=1)
    parser.add_argument('-en', '--enjoy', action='store_true')
    parser.add_argument('-ev', '--evaluate', action='store_true')
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-nw', '--n_rllib_workers', type=int, default=12)
    parser.add_argument('-vce', '--vis_cross_eval', action='store_true')
    parser.add_argument('-ovr', '--overwrite', action='store_true')
    parser.add_argument('-lo', '--load', action='store_true')
    parser.add_argument('-bc', '--batch_config', type=str, default='0')
    parser.add_argument('-gaw', '--gen_adversarial_worlds', action='store_true')
    args = parser.parse_args()
    job_time = 48
    n_rllib_workers = 0 if args.visualize else args.n_rllib_workers
    num_gpus = 0 if args.visualize else args.num_gpus
    render = True if args.enjoy else args.render
    load = True if args.visualize or args.enjoy or args.evaluate else args.load
    
    with open(os.path.join('configs', args.batch_config + '.yaml')) as f:
        # batch_config = json.load(f)
        batch_config = yaml.safe_load(f)
    batch_config = namedtuple('batch_config', batch_config.keys())(**batch_config)

    exp_sets = list(product(batch_config.exp_names, batch_config.env_classes, batch_config.ngen_nplay, 
                            batch_config.npol_qd_objectives, batch_config.fullobs_fov_models))
    exp_configs = []

    for exp_i, exp_set in enumerate(exp_sets):
        exp_name, env_cls, (gen_phase_len, play_phase_len), (n_policies, quality_diversity, objective), \
            (fully_observable, field_of_view, model) = exp_set

        # Just for reference in terms of what's currently explicitly supported/expected:
#       if objective in ['min_solvable', 'regret', 'max_reward']:
#           n_policies = 1
#       elif objective == 'contrastive':
#           n_policies = 2
#       elif quality_diversity:
#           n_policies = 3
#       else:
#           raise NotImplementedError
        
        exp_config = {
            'exp_name': exp_name,
            'enjoy': args.enjoy,
            'environment_class': env_cls,
            'evaluate': args.evaluate,
            'field_of_view': field_of_view,
            'fixed_worlds': False,
            'fully_observable': fully_observable,
            'gen_adversarial_worlds': args.gen_adversarial_worlds,
            'gen_phase_len': gen_phase_len,
            'generator_class': "TileFlipIndividual",
            'load': load or args.gen_adversarial_worlds,
            'model': model,
            'n_policies': n_policies,
            'n_rllib_workers': n_rllib_workers,
            'num_gpus': num_gpus,
            'objective_function': objective,
            'overwrite': args.overwrite,
            'play_phase_len': play_phase_len,
            'quality_diversity': quality_diversity,
            'render': render,
            'rotated_observations': True,
            'visualize': args.visualize,
        }
        exp_configs.append(exp_config)
        with open(os.path.join('configs', 'auto', f'{exp_i}.json'), 'w') as f:
            json.dump(exp_config, f, indent=4)

    sbatch_file = os.path.join('slurm', 'run.sh')

    if args.vis_cross_eval:
        vis_cross_eval(exp_configs)

        return 

    for exp_i, exp_set in enumerate(exp_sets):
        launch_job(sbatch_file=sbatch_file, exp_i=exp_i, job_time=job_time, job_cpus=n_rllib_workers, local=args.local)



if __name__ == '__main__':
    main()
