
## Quick Start
```
# general test 
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedmef_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --client_optimizer adam
# --adjustment_epochs None --growth_data_mode batch --client_optimizer adam

# quick test 
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedmef_distributed_pytorch.sh resnet18 cifar10 100 10 500 1 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 
# --adjustment_epochs None --growth_data_mode batch --client_optimizer sgd

CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedmef_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --strategy mink --gamma 0.5 --lambda_l2 0.01 --psi_of_lr 1.0 --max_lr 0.1 --enable_dynamic_lowest_k 0
```

### NLP task
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedmef_distributed_pytorch.sh gpt2 tinystories 100 10 200 1 0.1 0.1 --delta_T 10 --T_end 100 --num_eval 128 --frequency_of_the_test 10 --partition_alpha 5 --batch_size 16

CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedmef_distributed_pytorch.sh gpt2 tinystories 100 10 500 5 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --partition_alpha 5 --batch_size 16
```

## Usage
[$\cdot$] is mandatory arguments, {$\cdot$} is the optional arguments.
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedmef_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round]  [comm_round] [epochs] [target_density] [initial_lr]   {--delta_T , --T_end, --partition_alpha, --num_eval, --frequency_of_the_test, --strategy, --gamma, --lambda_l2, --psi_of_lr, --max_lr --enable_dynamic_lowest_k}
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
{--strategy} strategy for sap, only mink and vrandom is supported
{--gamma} a sap rate gamma to train
{--lambda_l2} lambda of l2 loss in BaE
{--psi_of_lr} weight for adjusted learning rate in BaE
{--max_lr} max learning rate when adjust in BaE
{--enable_dynamic_lowest_k} is a switch for finding lowest k in training or marking lowest k before training. 1 means find lowest k in training.
```