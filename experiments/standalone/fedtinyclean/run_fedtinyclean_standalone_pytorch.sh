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
DENSITY=${10}
DELTA_T=${11}
T_END=${12}
NUM_EVAL=${13}
FREQ=${14}



python3 ./main_fedtinyclean.py \
  --model $MODEL \
  --dataset $DATASET \
  --partition_alpha $PARTITION_ALPHA  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --target_density $DENSITY \
  --delta_T $DELTA_T \
  --T_end $T_END \
  --num_eval $NUM_EVAL \
  --frequency_of_the_test $FREQ \
