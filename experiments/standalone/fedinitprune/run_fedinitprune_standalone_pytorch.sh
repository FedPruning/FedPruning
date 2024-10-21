#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
BATCH_SIZE=$3
DATASET=$4
MODEL=$5
ROUND=$6
EPOCH=$7
LR=$8
DENSITY=$9
PARTITION_ALPHA=${10}
FREQ=${11}

python3 ./main_fedinitprune.py \
--dataset $DATASET \
--model $MODEL \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--lr $LR \
--target_density $DENSITY \
--partition_alpha $PARTITION_ALPHA  \
--frequency_of_the_test $FREQ \