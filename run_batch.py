import argparse
from itertools import product
import json
import os
import re

from batch_hyperparams import *


def launch_job(exp_i, job_time, job_cpus):
    cmd = f'python main.py --load_config {exp_i}'

    if args.local:
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--local', action='store_true', 
                        help='Run batch of jobs locally, in sequence. (Otherwise submit parallel jobs to SLURM.)')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-gpus', '--num_gpus', type=int, default=1)
    parser.add_argument('-en', '--enjoy', action='store_true')
    parser.add_argument('-ev', '--evaluate', action='store_true')
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-cpus', '--num_cpus', type=int, default=12)
    args = parser.parse_args()
    job_time = 48
    num_cpus = args.num_cpus

    exp_sets = list(product(gen_phase_lens, play_phase_lens, quality_diversities, objectives))

    for exp_i, exp_set in enumerate(exp_sets):
        gen_phase_len, play_phase_len, quality_diversity, objective = exp_set
        load = True
        num_gpus = 0 if args.visualize else args.num_gpus
        render = True if args.enjoy else args.render
        config = {
            'gen_phase_len': gen_phase_len,
            'play_phase_len': play_phase_len,
            'quality_diversity': quality_diversity,
            'objective_function': objective,
            'num_proc': num_cpus,
            'num_gpus': num_gpus,
            'visualize': args.visualize,
            'load': load,
            'enjoy': args.enjoy,
            'evaluate': args.evaluate,
            'render': render,
            'num_proc': num_cpus,
        }
        with open(os.path.join('auto_configs', f'{exp_i}.json'), 'w') as f:
            json.dump(config, f)

    sbatch_file = os.path.join('slurm', 'run.sh')

    for exp_i, exp_set in enumerate(exp_sets):
        launch_job(exp_i=exp_i, job_time=job_time, job_cpus=num_cpus)
