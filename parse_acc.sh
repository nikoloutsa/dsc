#!/bin/bash

FILE=$1

filename=$(basename -- "$FILE")
FILE_NAME="${filename%%.*}"

d=${FILE%.*.out}
FILE_ERR=${FILE%.*}.err

d=${d#*.}
RUN=$(echo $d| sed 's/\./,/g')
NODE=$(echo "$d" | cut -d. -f1)
GPU=$(echo "$d" | cut -d. -f2)
BATCH=$(echo "$d" | cut -d. -f3)

STEPS=$(grep -E 'Steps per epoch:' $FILE | tail -n1 | awk -F: -vORS=, '{ print $3 }' | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')
EPOCH_TIME=$(grep -E 'Average time per epoch:' $FILE | tail -n1 | awk -F: -vORS=, '{ print $3 }' | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')
ELAPSED=$(grep -E 'ELAPSED:' $FILE | tail -n1 | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')

echo "${FILE_NAME},${RUN},${STEPS}" | sed 's/,/./g'
echo "EPOCH_TIME=${EPOCH_TIME}"
echo "ELAPSED=${ELAPSED}"
echo "EPOCH,ACC_TRAIN,ACC_VAL"
for i in {1..90}
do
    ACC_VAL=$(grep -E "Validate Epoch  #${i}:" $FILE_ERR | tail -n1 | egrep -Eo 'accuracy=[-+]?([0-9]*\.[0-9]+|[0-9]+)' | sed 's/accuracy=//g')
    ACC_TRAIN=$(grep -E "Train Epoch     #${i}:" $FILE_ERR | tail -n1 | egrep -Eo 'accuracy=[-+]?([0-9]*\.[0-9]+|[0-9]+)' | sed 's/accuracy=//g')
    echo "${i},${ACC_TRAIN},${ACC_VAL}"
done

