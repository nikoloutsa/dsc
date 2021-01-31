#!/bin/bash -l

#SBATCH --job-name=pytorch_imagenet_resnet 
#SBATCH --output=pytorch_imagenet_resnet.%j.out 
#SBATCH --error=pytorch_imagenet_resnet.%j.err 
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
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

INDEX=0
for node in ${NODES[@]}
do
    echo "srun -l -w $node -N 1 -n 1 python main.py --epochs 1 --batch-size 64 -a resnet50 --dist-url 'tcp://${NODES[0]}-ib:5555' --dist-backend 'nccl' --multiprocessing-distributed --world-size $NUM_NODES --rank $INDEX $WORKDIR/data & " 
    srun -l -w $node -N 1 -n 1 python -u main.py --epochs 1 --batch-size 64 -a resnet50 --dist-url "tcp://${NODES[0]}-ib:5555" --dist-backend 'nccl' --multiprocessing-distributed --world-size $NUM_NODES --rank $INDEX $WORKDIR/data  &
    INDEX=$(($INDEX+1))
done

wait

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
