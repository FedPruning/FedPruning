
## Quick Start
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_feddip_distributed_pytorch.sh resnet18 cifar10 100 10 200 5 2 0.05 0.9 --delta_T 10 --T_end 100 --num_eval 128 --frequency_of_the_test 10 --lambda_shrink 5e-03 --mu 0.1

CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_feddip_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 2 0.05 0.9 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --lambda_shrink 5e-03 --mu 0.1
```

## Usage
[$\cdot$] is mandatory arguments, {$\cdot$} is the optional arguments.
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_feddip_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round] [comm_round] [epochs] [A_epochs] [init_sparsity] [final_sparsity] {--delta_T --T_end --partition_alpha --num_eval --frequency_of_the_test --adjustment_epochs --lambda_shrink --mu}
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
[A_epochs] the number of epochs to train before incremental training.
[init_sparsity] is the initial sparsity ratio for the sparse model.
[final_sparsity] is the final sparsity ratio for the sparse model.
{--delta_T} is the interval rounds between two adjustment round.
{--T_end} is the end round number for adjustment round.
{--partition_alpha} refers to the partition alpha, higher partition alpha makes lower degree of data heterogeneity.
{--num_eval} is the number data samples for validation, -1 means using the whole testing dataset.
{--frequency_of_the_test} the frequency to test/validate the performance during the training, using num_eval data samples.
{--adjustment_epochs} the number of local epochs used in model adjustment round, if not set, it equals to the number of epochs for training round.
{--lambda_shrink} penalty affecting the shrinkage rate.
{--mu} regularization parameter for proximal term.
```