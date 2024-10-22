
## Quick Start
```
CUDA_VISIBLE_DEVICES=0 sh run_fedinitprune_standalone_pytorch.sh 100 10 64 cifar10 resnet18 200 10 0.001 0.1 0.5 5
```

## Usage
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedinitprune_standalone_pytorch.sh [client_num_in_total] [client_num_per_round] [batch_size] [dataset] [model] [comm_round] [epochs] [lr] [target_density] [partition_alpha] [frequency_of_the_test]
```

where

```
[gpus] specifies which GPUs to use.
[client_num_in_total] is the total number of clients.
[client_num_per_round] is the number of clients selected per round.
[batch_size] is the batch size.
[dataset] is the name of the dataset.
[model] is the name of the model.
[comm_round] is the number of communication rounds.
[epochs] is the number of local epochs.
[lr] is the learning rate.
[target_density] is the pruning target density.
[partition_alpha] refers to the partition alpha, higher partition alpha makes lower degree of data heterogeneity.
[frequency_of_the_test] the frequency to test/validate the performance during the training.
```