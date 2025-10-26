#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
MODEL=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6
LR=$7
DATASET=$8
PARTITION_ALPHA=$9
FREQ=${10}

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_default" \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --model $MODEL \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --initial_lr $LR \
  --dataset $DATASET \
  --partition_alpha $PARTITION_ALPHA  \
  --frequency_of_the_test $FREQ \
