#!/bin/bash -l

#SBATCH --job-name=distributed # Job name
#SBATCH --output=distributed.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=distributed.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=2 # Total number of tasks
#SBATCH --gres=gpu:2 # GPUs per node
#SBATCH --nodes=2 # Total number of nodes requested
#SBATCH --ntasks-per-node=1 # Tasks per node
#SBATCH --cpus-per-task=1 # Threads per task
#SBATCH --mem=28000 # Memory per job in MB
#SBATCH -t 00:30:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
module load gnu/8 cuda/10.1.168 intel/18 java/12.0.2 intelmpi/2018 tensorflow/2.3.0


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_OF_NODES=${#NODES[@]}

WORKERS=$(printf '"%s-ib:5555",' "${NODES[@]}" | sed 's/,$//')

echo "NUM_OF_NODES: $NUM_OF_NODES"
echo "WORKERS: $WORKERS"

INDEX=0
for node in ${NODES[@]}
do
    export TF_CONFIG='{"cluster": {"worker": ['$WORKERS']}, "task": {"type": "worker", "index": '$INDEX'} }'
    echo "srun $node $TF_CONFIG"
    srun -w $node -n 1 --exclusive --export=ALL,TF_CONFIG python resnet_cifar10.py &
    INDEX=$(($INDEX+1))
    sleep 1
done

wait

#TF_CONFIG='{"cluster": {"worker": ["'$NODE1':5555", "'$NODE2':5555"]}, "task": {"type": "worker", "index": 1} }'
#srun -w $NODE2 -n 1 --exclusive --export=ALL,TF_CONFIG python distributed.py &

