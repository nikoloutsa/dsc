#!/bin/bash -l

#SBATCH --job-name=horovod_tensorflow_synthetic_benchmark 
#SBATCH --output=logs/horovod_tensorflow_synthetic_benchmark.N1.G1.B32.%j.out 
#SBATCH --error=logs/horovod_tensorflow_synthetic_benchmark.N1.G1.B32.%j.err 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=28000 # Memory per job in MB
#SBATCH -t 01:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project
##SBATCH --export=ALL,HOROVOD_CYCLE_TIME=1,NCCL_DEBUG=INFO,HOROVOD_MPI_THREADS_DISALBE=1

# Load any necessary modules
module purge
module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow/2.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Start at `date`"
START_TIME=$(date +%s)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS_PER_NODE tasks per node"
echo "Job id is $SLURM_JOBID"


NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_NODES=${#NODES[@]}
WORKERS=$(printf '%s-ib:'${SLURM_NTASKS_PER_NODE}',' "${NODES[@]}" | sed 's/,$//')

horovodrun --gloo -np $SLURM_NTASKS -H $WORKERS  --start-timeout 120 --gloo-timeout-seconds 120 \
    python -u train.horovod.tensorflow.synthetic.py --batch-size 32 --num-batches-per-iter ${1:-128} --num-iters 3

#--network-interface ib0
END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
