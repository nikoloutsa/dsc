#!/bin/bash -l

#SBATCH --job-name=horovod_tensorflow_imagenet_resnet 
#SBATCH --output=logs/horovod_tensorflow_imagenet_resnet.N1.G2.B16.%j.out 
#SBATCH --error=logs/horovod_tensorflow_imagenet_resnet.N1.G2.B16.%j.err 
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 12:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

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
export NUM_NODES=${#NODES[@]}

echo "NUM_NODES: $NUM_NODES"
#WORKERS=$(printf '%s-ib:'${SLURM_NTASKS_PER_NODE}',' "${NODES[@]}" | sed 's/,$//')
WORKERS=$(printf '%s-ib:'${SLURM_NTASKS_PER_NODE}',' "${NODES[@]}" | sed 's/,$//')

#echo "horovodrun --gloo -np $SLURM_NTASKS -H $WORKERS --network-interface=ib0 --start-timeout 120 --gloo-timeout-seconds 120 python train.horovod.tensorflow.py --config=configs/tensorflow_imagenet_resnet.B16.yaml"
#horovodrun --gloo -np $SLURM_NTASKS -H $WORKERS --network-interface ib0 --start-timeout 120 --gloo-timeout-seconds 120 python train.horovod.tensorflow.py --config=configs/tensorflow_imagenet_resnet.B16.yaml
srun -l python train.horovod.tensorflow.py --config=configs/tensorflow_imagenet_resnet.B16.yaml

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"
