#!/bin/bash -l

#SBATCH --job-name=pytorch_horovod
#SBATCH --output=pytorch_horovod.%j.out 
#SBATCH --error=pytorch_horovod.%j.err 
#SBATCH --ntasks=16
#SBATCH --gres=gpu:2
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
##SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 02:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 intelmpi/2018 cuda/10.1.168 pytorch/1.7.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Start at `date`"
START_TIME=$(date +%s)

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_NODES=${#NODES[@]}

echo "NUM_NODES: $NUM_NODES"

srun -l python pytorch_imagenet_resnet50.py --epochs 1 --batch-size 128 --val-batch-size 128 --train-dir $WORKDIR/data/train --val-dir $WORKDIR/data/val

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
