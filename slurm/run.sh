#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=prtcl_11
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=prtcl_11_%j.out

cd /scratch/se2161/particle-world || exit

conda init bash
conda activate particle

export TUNE_RESULT_DIR='./ray_results/'

python main.py --load_config 11

# leave trailing line
