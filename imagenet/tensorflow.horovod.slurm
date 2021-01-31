#!/bin/bash -l

#SBATCH --job-name=horovod_imagenet_resnet 
#SBATCH --output=horovod_imagenet_resnet.%j.out 
#SBATCH --error=horovod_imagenet_resnet.%j.err 
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
##SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 02:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow/2.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Start at `date`"
START_TIME=$(date +%s)

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_NODES=${#NODES[@]}

echo "NUM_NODES: $NUM_NODES"

srun -l python keras_imagenet_resnet50.py --epochs 1 --batch-size 96 --train-dir $WORKDIR/data/train --val-dir $WORKDIR/data/val 

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
