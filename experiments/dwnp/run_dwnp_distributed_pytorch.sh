#!/usr/bin/env bash

MODEL=$1
DATASET=$2
CLIENT_NUM=$3
WORKER_NUM=$4
ROUND=$5
EPOCH=$6
DENSITY=$7
LR=$8

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

command="mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_dwnp.py \
  --gpu_mapping_file gpu_mapping.yaml \
  --gpu_mapping_key mapping_default \
  --model $MODEL \
  --dataset $DATASET \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --lr $LR \
  --target_density $DENSITY"

shift 8

for arg in "$@"; do
  command="$command $arg"
done

eval $command
