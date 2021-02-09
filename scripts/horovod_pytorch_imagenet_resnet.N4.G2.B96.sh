#!/bin/bash -l

#SBATCH --job-name=horovod_pytorch_imagenet_resnet 
#SBATCH --output=logs/horovod_pytorch_imagenet_resnet.N4.G2.B96.%j.out 
#SBATCH --error=logs/horovod_pytorch_imagenet_resnet.N4.G2.B96.%j.err 
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 04:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 intelmpi/2018 cuda/10.1.168 pytorch/1.7.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Start at `date`"
START_TIME=$(date +%s)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS_PER_NODE tasks per node"
echo "Job id is $SLURM_JOBID"

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
export NUM_NODES=${#NODES[@]}

echo "NUM_NODES: $NUM_NODES"

srun -l python train.horovod.pytorch.imagenet.py --config=configs/pytorch_imagenet_resnet.B96.yaml

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
