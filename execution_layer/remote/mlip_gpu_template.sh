#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=v100m3
#SBATCH --output=out
#SBATCH --error=err
export cuda_path="/data/app/cuda/cuda-12.8"
export PATH=${cuda_path}/bin:$PATH
export LD_LIBRARY_PATH=${cuda_path}/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=${cuda_path}:${CUDA_HOME:-}
source /data/app/anaconda3/2024.10-1/bin/activate
conda activate mace
python3 run_mlip_task.py
