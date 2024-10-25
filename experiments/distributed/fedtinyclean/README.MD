
## Quick Start
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedtinyclean_default.sh resnet18  cifar10 10 10 100 1 sgd 0.1 1 0.01
```

## Usage
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedtinyclean_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round]  [comm_round] [epochs] [ooptimizer] [density] [initial_lr] 
```

where

```
[gpus] specifies which GPUs to use.
[client_num_in_total] is the total number of clients.
[client_num_per_round] is the number of clients selected per round.
[model] is the name of the model.
[comm_round] is the number of communication rounds.
[epochs] is the number of local epochs.
[batch_size] is the batch size.
[initial_lr] is the initial learning rate.
[dataset] is the name of the dataset.
[partition_alpha] refers to the partition alpha, higher partition alpha makes lower degree of data heterogeneity
[delta_T] is the interval rounds between two adjustment round.
[T_end] is the end round number for adjustment round.
[num_eval] is the number data samples for validation, -1 means using the whole testing dataset.
[frequency_of_the_test] the frequency to test/validate the performance during the training, using num_eval data samples
```