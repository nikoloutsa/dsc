#!/bin/bash -l

#SBATCH --job-name=mnist # Job name
#SBATCH --output=out/mnist.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=out/mnist.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=2 # Total number of tasks
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --nodes=2 # Total number of nodes requested
#SBATCH --ntasks-per-node=1 # Tasks per node
#SBATCH --cpus-per-task=1 # Threads per task
##SBATCH --mem=56000 # Memory per job in MB
#SBATCH -t 00:30:00 # Run time (hh:mm:ss) - (max 48h)
#SBATCH --partition=gpu # Run on the GPU nodes queue
#SBATCH -A pa201202 # Accounting project

# Load any necessary modules
module purge
#module load gnu/6.5.0 java/1.8.0 cuda/10.1.168 tensorflow/1.14.0
module load gnu/6.4.0 intel java/1.8.0 cuda/9.2.148 tensorflow/1.12.0gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
NUM_OF_NODES=${#NODES[@]}

WORKERS=$(printf '"%s-ib:5555",' "${NODES[@]}" | sed 's/,$//')

echo "NUM_OF_NODES: $NUM_OF_NODES"
echo "WORKERS: $WORKERS"

#srun -l python -u keras_mnist_advanced.py
srun -l python -u keras_mnist.py

echo "END"

#INDEX=0
#for node in ${NODES[@]}
#do
#    TF_CONFIG='{"cluster": {"worker": ['$WORKERS']}, "task": {"type": "worker", "index": '$INDEX'} }'
#    echo "srun $node $TF_CONFIG"
#    srun -w $node -n 1 --exclusive --export=ALL,TF_CONFIG python mnist.py &
#    INDEX=$(($INDEX+1))
#    sleep 1
#done
#
#wait

#TF_CONFIG='{"cluster": {"worker": ["'$NODE1':5555", "'$NODE2':5555"]}, "task": {"type": "worker", "index": 1} }'
#srun -w $NODE2 -n 1 --exclusive --export=ALL,TF_CONFIG python mnist.py &

