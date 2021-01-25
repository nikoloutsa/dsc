#!/bin/bash -l

#SBATCH --job-name=tensorflow_cifar10_resnet 
#SBATCH --output=logs/tensorflow_cifar10_resnet.N1.G1.B128.%j.out 
#SBATCH --error=logs/tensorflow_cifar10_resnet.N1.G1.B128.%j.err 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
##SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 08:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow/2.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
export NUM_NODES=${#NODES[@]}


echo "Start at `date`"
echo "$CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS_PER_NODE tasks per node"
echo "Job id is $SLURM_JOBID"

#tensorflow 1 GPU baseline
srun -l python train.tensorflow.py configs/tensorflow_cifar10_resnet.B128.yaml 

echo "End at `date`"

#srun -l python train.py configs/tensorflow_cifar10_resnet.yaml -v --distributed
