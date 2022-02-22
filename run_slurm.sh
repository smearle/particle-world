#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=prtcl_0
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=prtcl_0_%j.out

cd /scratch/se2161/particle-world || exit

conda init bash
conda activate particle

export TUNE_RESULT_DIR='./ray_results/'
#python main.py -g TileFlipGenerator -np 12 -gpus 1 -exp 0 --maxTotalBins 961 -qd
python main.py -g TileFlipGenerator -np 12 -gpus 1 -exp min_solvable_1 -obj min_solvable

# gotta leave a trailing line here apparently
