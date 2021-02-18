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

STEPS=$(grep -E 'Steps per epoch:' $FILE | tail -n1 | awk -F: -vORS=, '{ print $NF }' | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')
EPOCH_TIME=$(grep -E 'Average time per epoch:' $FILE | tail -n1 | awk -F: -vORS=, '{ print $NF }' | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')
ELAPSED=$(grep -E 'ELAPSED:' $FILE | tail -n1 | grep -Eo '[-+]?([0-9]*\.[0-9]+|[0-9]+)')
ACCURACY=$(grep -E 'Validate Epoch  #90:' $FILE_ERR | tail -n1 | egrep -Eo 'accuracy=[-+]?([0-9]*\.[0-9]+|[0-9]+)' | sed 's/accuracy=//g')

echo "${FILE_NAME},${RUN},${STEPS},${EPOCH_TIME},${ELAPSED},${ACCURACY}"
