#!/bin/bash
#SBATCH --partition biggpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name DROZDA0
#SBATCH --time=6:00:00
#SBATCH --array=0-4

srun python create_data.py $SLURM_ARRAY_TASK_ID > ${SLURM_JOBID}.out
