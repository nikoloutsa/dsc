#!/bin/bash -l

#SBATCH --job-name=pytorch_cifar10_resnet 
#SBATCH --output=pytorch_cifar10_resnet.%j.out 
#SBATCH --error=pytorch_cifar10_resnet.%j.err 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 00:30:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 intelmpi/2018 cuda/10.1.168 pytorch/1.7.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_NODES=${#NODES[@]}

echo "NUM_NODES: $NUM_NODES"

WORKERS=$(printf '"%s-ib:5555",' "${NODES[@]}" | sed 's/,$//')
echo "WORKERS: $WORKERS"

srun -l python train.pytorch.py -a resnet50 -b 32 --epochs=1 data 

##INDEX=0
##for node in ${NODES[@]}
##do
##    srun -l -w $node -n 1 python main.py -a resnet50 --dist-url 'tcp://$NODES[0]-ib:5555' --dist-backend 'nccl' --multiprocessing-distributed --world-size $NUM_NODES --rank $INDEX data &
##    INDEX=$(($INDEX+1))
##    sleep 1
##done
##
##wait


#srun -l python main.py -a resnet50 data

