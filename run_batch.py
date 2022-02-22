import argparse
from itertools import product
import json
import os
import re


exp_names = [
    0,
]

gen_phase_lens = [
    1,
    10,
    50,
    # 100,
    -1,
]

play_phase_lens = [
    1,
    10,
    50,
    # 100,
    # -1,
]

quality_diversities = [
    False,
    # True,
]

objectives = [
    'min_solvable',
]


def launch_job(exp_i, job_time, job_cpus):
    cmd = f'python main.py --load_config {exp_i}'

    if args.local:
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
    args = parser.parse_args()
    job_time = 72
    job_cpus = 12
    if args.local:
        num_cpus = 12
    else:
        num_cpus = 12

    exp_sets = list(product(gen_phase_lens, play_phase_lens, quality_diversities, objectives))

    for exp_i, exp_set in enumerate(exp_sets):
        gen_phase_len, play_phase_len, quality_diversity, objective = exp_set
        config = {
            'gen_phase_len': gen_phase_len,
            'play_phase_len': play_phase_len,
            'quality_diversity': quality_diversity,
            'objective_function': objective,
            'num_proc': num_cpus,
            'num_gpus': 1,
        }
        with open(os.path.join('auto_configs', f'{exp_i}.json'), 'w') as f:
            json.dump(config, f)

    sbatch_file = os.path.join('slurm', 'run.sh')

    for exp_i, exp_set in enumerate(exp_sets):
        launch_job(exp_i=exp_i, job_time=job_time, job_cpus=job_cpus)
