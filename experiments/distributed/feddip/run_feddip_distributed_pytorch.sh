#!/usr/bin/env bash

MODEL=$1
DATASET=$2
CLIENT_NUM=$3
WORKER_NUM=$4
ROUND=$5
EPOCH=$6
A_EPOCHS=$7
INIT_SPARSITY=$8
FINAL_SPARSITY=$9

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

# Initialize command with mandatory arguments
command="mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_feddip.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_default" \
  --model $MODEL \
  --dataset $DATASET \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --A_epochs $A_EPOCHS \
  --init_sparsity $INIT_SPARSITY \
  --final_sparsity $FINAL_SPARSITY"

# Shift the first 9 arguments (changed from 8)
shift 9

# Append optional arguments if provided
for arg in "$@"; do
  command="$command $arg"
done

eval $command