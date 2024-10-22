
## Quick Start
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 resnet18 200 10 64 0.2 cifar10 0.5 5
```

## Usage
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedavg_distributed_pytorch.sh [client_num_in_total] [client_num_per_round] [model] [comm_round] [epochs] [batch_size] [initial_lr] [dataset] [partition_alpha] [frequency_of_the_test]
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
[partition_alpha] refers to the partition alpha, higher partition alpha makes lower degree of data heterogeneity.
[frequency_of_the_test] the frequency to test/validate the performance during the training.
```

export WANDB_DIR='./../../../../wandb/'
export WANDB_CACHE_DIR='./../../../../wandb/.cache/'
export WANDB_CONFIG_DIR='./../../../../wandb/.config/'
wandb login a43329fc912eff90a92976498088770f5745bb57
