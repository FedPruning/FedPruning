
## Quick Start
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedtiny_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --progressive_pruning on --ABNS on --ABNS_num_of_candidates 5
```

## Usage
[$\cdot$] is mandatory arguments, {$\cdot$} is the optional arguments.
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedtiny_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round]  [comm_round] [epochs] [target_density] [initial_lr]   {--delta_T , --T_end, --partition_alpha, --num_eval, --frequency_of_the_test, --progressive_pruning, --ABNS, --ABNS_num_of_candidates}
```

where

```
[gpus] specifies which GPUs to use.
[model] is the name of the model.
[dataset] is the name of the dataset.
[client_num_in_total] is the total number of clients.
[client_num_per_round] is the number of clients selected per round.
[comm_round] is the number of communication rounds.
[epochs] is the number of local epochs.
[target_density] is the target density for the sparse model.
[initial_lr] is the initial learning rate.
{--delta_T} is the interval rounds between two adjustment round.
{--T_end} is the end round number for adjustment round.
{--partition_alpha} refers to the partition alpha, higher partition alpha makes lower degree of data heterogeneity
{--num_eval}is the number data samples for validation, -1 means using the whole testing dataset.
{--frequency_of_the_test} the frequency to test/validate the performance during the training, using num_eval data samples
{--progressive_pruning} the switch of progressive pruning module, on for open and off for close
{--ABNS} the switch of Adaptive BN Selection module, on for open and off for close
{--ABNS_num_of_candidates} the num_of_candidates in Adaptive BN Selection module
```