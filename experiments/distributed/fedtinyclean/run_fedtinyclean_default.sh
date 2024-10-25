#!/usr/bin/env bash
MODEL=$1
DATASET=$2
CLIENT_NUM=$3
WORKER_NUM=$4
ROUND=$5
EPOCH=$6
CLIENT_OPTIMIZER=$7
DENSITY=$8
LR=$9

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

# Initialize the command with mandatory arguments
command="mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedinitprune.py \
  --model $MODEL \
  --dataset $DATASET \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --optimizer $CLIENT_OPTIMIZER \
  --density $DENSITY \
  --lr $LR"

# Shift the first 9 arguments
shift 9

# Append optional arguments only if they are provided
for arg in "$@"; do
  command="$command $arg"
done

eval $command


