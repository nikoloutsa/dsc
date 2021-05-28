#!/bin/bash -l

#SBATCH --job-name=horovod_pytorch_imagenet_resnet 
#SBATCH --output=horovod_pytorch_imagenet_resnet.%j.out 
#SBATCH --error=horovod_pytorch_imagenet_resnet.%j.err 
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
##SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 01:30:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 intelmpi/2018 cuda/10.1.168 pytorch/1.7.0
#module load gnu/8.3.0 intel/18.0.5 intelmpi/2018.5 cuda/10.1.168 pytorch/1.2.0
#module load gnu/8 intel/18 intelmpi/2018 cuda/10.1.168 pytorch/1.4.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Start at `date`"
START_TIME=$(date +%s)

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_NODES=${#NODES[@]}
export NUM_NODES=${#NODES[@]}
WORKERS=$(printf '%s-ib:2,' "${NODES[@]}" | sed 's/,$//')

echo "NUM_NODES: $NUM_NODES"



# Launch the executable
#horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
#horovodrun --verbose -np 4 -H $WORKERS python pytorch_mnist.py
#mpirun -np 4 -H $WORKERS python pytorch_mnist.py

srun -l python -u pytorch_mnist.py
#srun -l python -u pytorch_imagenet_resnet50.py.1 --epochs 1 --batch-size 64 --val-batch-size 64 --train-dir $WORKDIR/data/train --val-dir $WORKDIR/data/val
#python pytorch_imagenet_resnet50.py --epochs 1 --batch-size 64 --val-batch-size 64 --train-dir $WORKDIR/data/train --val-dir $WORKDIR/data/val

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
