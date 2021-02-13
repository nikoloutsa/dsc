#!/bin/bash

FILE=$1

filename=$(basename -- "$FILE")
FILE_NAME="${filename%%.*}"

d=${FILE%.*.err}
d=${d#*.}
RUN=$(echo $d| sed 's/\./,/g')
NODE=$(echo "$d" | cut -d. -f1)
GPU=$(echo "$d" | cut -d. -f2)
BATCH=$(echo "$d" | cut -d. -f3)

#Validate Epoch  #90: 100%|██████████| 10/10 [00:05<00:00,  1.94it/s, loss=3.85, accuracy=52.4]

ACCURACY=$(grep -E 'Validate Epoch  #90:' $FILE | tail -n1 | egrep -Eo 'accuracy=[-+]?([0-9]*\.[0-9]+|[0-9]+)' | sed 's/accuracy=//g')

#STEPS=$(grep -E 'Steps per epoch:' $FILE | tail -n1 | awk -F: -vORS=, '{ print $3 }' | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')
#EPOCH_TIME=$(grep -E 'Average time per epoch:' $FILE | tail -n1 | awk -F: -vORS=, '{ print $3 }' | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')
#ELAPSED=$(grep -E 'ELAPSED:' $FILE | tail -n1 | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')

#echo "${RUN},${NODE},${GPU},${BATCH},${STEPS},${EPOCH_TIME}"
echo "${FILE_NAME},${RUN},${ACCURACY}"

#B=32; tail logs/horovod_pytorch_cifar10_resnet.N1.G2.B${B}*.out | grep -E 'epoch:|ELAPSED' | sed 's/ELAPSED:/:ELAPSED:/g' |  awk -F: -vORS=, '{ print $3 }' | sed 's/,$/\n/'
