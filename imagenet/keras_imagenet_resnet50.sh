#!/bin/bash -l

#SBATCH --job-name=keras_imagenet_resnet50
#SBATCH --output=keras_imagenet_resnet50.%j.out 
#SBATCH --error=keras_imagenet_resnet50.%j.err 
#SBATCH --ntasks=32
#SBATCH --gres=gpu:2
#SBATCH --nodes=16 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 12:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project
#SBATCH --export=ALL,HOROVOD_CYCLE_TIME=1,NCCL_DEBUG=INFO,HOROVOD_MPI_THREADS_DISALBE=1

# Load any necessary modules
module purge
#module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow/2.3.0
module load gnu/6.4.0 intel java/1.8.0 cuda/9.2.148 tensorflow/1.12.0gpu
#module load gnu/6.5.0 java/1.8.0 cuda/10.1.168 tensorflow/1.14.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Start at `date`"
START_TIME=$(date +%s)

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_NODES=${#NODES[@]}

echo "NUM_NODES: $NUM_NODES"

srun -l python -u keras_imagenet_resnet50.py --epochs 1 --batch-size 64 --train-dir $WORKDIR/data/train --val-dir $WORKDIR/data/val 

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
