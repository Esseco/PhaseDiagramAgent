#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=%j.out
#SBATCH --error=%j.err
module load gcc/12.2.0
export vasp_path="/data/app/vasp/6.5.1-nvhpc"
module use /data/app/nvhpc/22.5_cuda12.9/modulefiles/
module load nvhpc-hpcx fftw/3.3.10-nvhpc
export PATH=${vasp_path}/bin:$PATH
ulimit -s unlimited
ulimit -l unlimited
mpirun -np $SLURM_NPROCS vasp_std
