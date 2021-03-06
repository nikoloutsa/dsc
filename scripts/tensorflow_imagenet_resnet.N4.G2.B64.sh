#!/bin/bash -l

#SBATCH --job-name=tensorflow_imagenet_resnet 
#SBATCH --output=logs/tensorflow_imagenet_resnet.N4.G2.B64.%j.out 
#SBATCH --error=logs/tensorflow_imagenet_resnet.N4.G2.B64.%j.err 
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 24:00:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow/2.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
export NUM_NODES=${#NODES[@]}
WORKERS=$(printf '"%s-ib:5555",' "${NODES[@]}" | sed 's/,$//')


echo "Start at `date`"
START_TIME=$(date +%s)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS_PER_NODE tasks per node"
echo "Job id is $SLURM_JOBID"

INDEX=0
for node in ${NODES[@]}
do
    export TF_CONFIG='{"cluster": {"worker": ['$WORKERS']}, "task": {"type": "worker", "index": '$INDEX'} }'
    echo "srun $node $TF_CONFIG"
    srun -w $node -n 1 -l --export=ALL,TF_CONFIG python train.tensorflow.distributed.py --config=configs/tensorflow_imagenet_resnet.B64.yaml & 
    INDEX=$(($INDEX+1))
done

wait

END_TIME=$(date +%s)
echo "ELAPSED: $(($END_TIME - $START_TIME)) seconds"
 
echo "End at `date`"




