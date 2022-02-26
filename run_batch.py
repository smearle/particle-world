import argparse
from itertools import product
import json
import os
import re

from batch_hyperparams import *
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
    parser.add_argument('-cpus', '--num_cpus', type=int, default=12)
    parser.add_argument('-vce', '--vis_cross_eval', action='store_true')
    parser.add_argument('-ovr', '--overwrite', action='store_true')
    args = parser.parse_args()
    job_time = 48
    num_cpus = 0 if args.visualize else args.num_cpus
    num_gpus = 0 if args.visualize else args.num_gpus
    render = True if args.enjoy else args.render

    exp_sets = list(product(exp_names, gen_play_phase_lens, qd_objectives))
    exp_configs = []

    for exp_i, exp_set in enumerate(exp_sets):
        exp_name, (gen_phase_len, play_phase_len), (quality_diversity, objective) = exp_set
        if objective == 'min_solvable':
            n_policies = 1
        elif objective == 'contrastive':
            n_policies = 2
        elif quality_diversity:
            n_policies = 3
        else:
            raise NotImplementedError
        
        exp_config = {
            'exp_name': exp_name,
            'gen_phase_len': gen_phase_len,
            'play_phase_len': play_phase_len,
            'quality_diversity': quality_diversity,
            'objective_function': objective,
            'n_policies': n_policies,
            'fully_observable': False,
            'model': None,
            'num_proc': num_cpus,
            'num_gpus': num_gpus,
            'visualize': args.visualize,
            'load': not args.overwrite,
            'enjoy': args.enjoy,
            'evaluate': args.evaluate,
            'render': render,
            'num_proc': num_cpus,
        }
        exp_configs.append(exp_config)
        with open(os.path.join('auto_configs', f'{exp_i}.json'), 'w') as f:
            json.dump(exp_config, f)

    sbatch_file = os.path.join('slurm', 'run.sh')

    if args.vis_cross_eval:
        vis_cross_eval(exp_configs)

        return 

    for exp_i, exp_set in enumerate(exp_sets):
        launch_job(sbatch_file=sbatch_file, exp_i=exp_i, job_time=job_time, job_cpus=num_cpus, local=args.local)



if __name__ == '__main__':
    main()
