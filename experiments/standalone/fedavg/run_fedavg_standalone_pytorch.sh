#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
BATCH_SIZE=$3
DATASET=$4
MODEL=$5
ROUND=$6
EPOCH=$7
ILR=$8
FLR=$9
PARTITION_ALPHA=${10}
FREQ=${11}

python3 ./main_fedavg.py \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--batch_size $BATCH_SIZE \
--dataset $DATASET \
--model $MODEL \
--comm_round $ROUND \
--epochs $EPOCH \
--initial_lr $ILR \
--final_lr   $FLR \
--partition_alpha $PARTITION_ALPHA  \
--frequency_of_the_test $FREQ \
