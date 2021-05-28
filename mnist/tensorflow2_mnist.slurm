#!/bin/bash -l

#SBATCH --job-name=tensorflow2_mnist # Job name
#SBATCH --output=tensorflow2_mnist.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=tensorflow2_mnist.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=2 # Total number of tasks
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --nodes=2 # Total number of nodes requested
#SBATCH --ntasks-per-node=1 # Tasks per node
#SBATCH --cpus-per-task=1 # Threads per task
#SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 00:30:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules

module purge
module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
export NUM_NODES=${#NODES[@]}
WORKERS=$(printf '%s-ib:2,' "${NODES[@]}" | sed 's/,$//')


srun python tensorflow2_mnist.py

echo "END"
