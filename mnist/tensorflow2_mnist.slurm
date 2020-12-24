#!/bin/bash -l

#SBATCH --job-name=tensorflow2_mnist # Job name
#SBATCH --output=tensorflow2_mnist.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=tensorflow2_mnist.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=1 # Total number of tasks
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --nodes=1 # Total number of nodes requested
#SBATCH --ntasks-per-node=1 # Tasks per node
#SBATCH --cpus-per-task=10 # Threads per task
#SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 00:30:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules

module load gnu
module load intel
module load intelmpi
module load cuda

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch the executable
srun python tensorflow2_mnist.horovod.py
