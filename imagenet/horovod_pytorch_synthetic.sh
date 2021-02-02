#!/bin/bash -l

#SBATCH --job-name=horovod_pytorch_synthetic_benchmark 
#SBATCH --output=horovod_pytorch_synthetic_benchmark.%j.out 
#SBATCH --error=horovod_pytorch_synthetic_benchmark.%j.err 
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
##SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 04:00:00 # Run time (hh:mm:ss) - (max 48h)
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

srun -l python -u pytorch_synthetic_benchmark.py

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
