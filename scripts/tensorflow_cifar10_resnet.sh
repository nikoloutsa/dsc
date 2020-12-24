#!/bin/bash -l

#SBATCH --job-name=tensorflow_cifar10_resnet 
#SBATCH --output=logs/tensorflow_cifar10_resnet.%j.out 
#SBATCH --error=logs/tensorflow_cifar10_resnet.%j.err 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1 
##SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 00:30:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow/2.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_NODES=${#NODES[@]}

echo "NUM_NODES: $NUM_NODES"

srun -l python train.py configs/tensorflow_cifar10_resnet.yaml -v
