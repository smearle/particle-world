#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=prtcl_min_solvable_1-pol_1-gen_1-play_mdl-rnn_debug
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=prtcl_min_solvable_1-pol_1-gen_1-play_mdl-rnn_debug_%j.out

cd /scratch/se2161/particle-world || exit

conda init bash
conda activate particle

export TUNE_RESULT_DIR='./ray_results/'

python main.py --load_config min_solvable_1-pol_1-gen_1-play_mdl-rnn_debug

# leave trailing line
