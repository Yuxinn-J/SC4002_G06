#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --output=%j_%N_log.out  # change this line to your output file


source /p/scratch/ccstdl/xu17/miniconda3/etc/profile.d/conda.sh
   conda activate /p/scratch/ccstdl/xu17/miniconda3/envs/jz

# You may need to activate a Python virtual environment if you have one
# source /path/to/your/virtualenv/bin/activate

srun python /p/scratch/ccstdl/xu17/jz/SC4002_G06/Question_Classification/part2_fine_tune.py