#!/bin/bash
#SBATCH --partition biggpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name DROZDA0
#SBATCH --time=24:00:00
srun python train.py > ${SLURM_JOBID}.out
