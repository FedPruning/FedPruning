
### CV task
```
# CV task: regular run (11 processes with mapping_default)


CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_fedsparsy_distributed_pytorch.sh resnet18 cifar10 100 10 1000 2 0.5 0.1 --gpu_mapping_key mapping_default --target_sparsity 0.8 --reparam_beta 1.2 --reparam_per_step --server_lr 0.8 --server_clip_norm -1 --num_eval 512 --frequency_of_the_test 5 --batch_size 64 --client_optimizer sgd

# CV task: medium-scale test
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_fedsparsy_distributed_pytorch.sh resnet18 cifar10 20 4 20 1 0.1 0.01 --gpu_mapping_key mapping_5_processes --target_sparsity 0.9 --reparam_beta 1.25 --reparam_per_step --server_lr 0.1 --num_eval 128 --frequency_of_the_test 2 --batch_size 64 --client_optimizer sgd



# CV task: fast sanity check

CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_fedsparsy_distributed_pytorch.sh resnet18 cifar10 10 2 3 1 0.1 0.01 --gpu_mapping_key mapping_3_processes --target_sparsity 0.9 --reparam_beta 1.25 --reparam_per_step --server_lr 0.1 --num_eval 64 --frequency_of_the_test 1 --batch_size 32 --server_clip_norm -1


```

### NLP task
```
# NLP task: fast sanity check
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_fedsparsy_distributed_pytorch.sh gpt2 tinystories 10 2 2 1 0.1 0.001 --gpu_mapping_key mapping_3_processes --dataset_ratio 0.01 --partition_alpha 5 --batch_size 4 --num_eval 32 --target_sparsity 0.9 --reparam_beta 1.25 --server_lr 0.1

# NLP task: medium-scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_fedsparsy_distributed_pytorch.sh gpt2 tinystories 20 4 10 1 0.1 0.001 --gpu_mapping_key mapping_5_processes --dataset_ratio 0.02 --partition_alpha 5 --batch_size 8 --num_eval 64 --frequency_of_the_test 2 --target_sparsity 0.9 --reparam_beta 1.25 --server_lr 0.1

# NLP task: regular run
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_fedsparsy_distributed_pytorch.sh gpt2 tinystories 100 10 200 1 0.1 0.001 --dataset_ratio 0.05 --partition_alpha 5 --batch_size 16 --num_eval 128 --frequency_of_the_test 10 --target_sparsity 0.9 --reparam_beta 1.25 --server_lr 0.1

```

## SparsyFed Logic
SparsyFed uses two different sparsification stages:

1. **Reference weight density (controlled by `[target_density]`)*
*
   - `[target_density]` controls the **reference sparse weight structure** (via `layer_density_dict` / `reference_mask_dict`).
   - Example: `[target_density]=0.9` means about 90% weights are kept in sparse layers (about 10% weight sparsity).
   - This is **not communication density**.
   - This is also **not directly activation tensor density**.
   - During local training, activation-gradient Top-K sparsity is computed from layer-wise weight density: `activation_sparsity(layer) ≈ 1 - layer_density(layer)`.

2. **Sparse update upload before communication (controlled by `--target_sparsity`)**
   - After local training, each client computes the model update `Δω = ω_current - ω_initial`.
   - A global Top-K operator is applied to `Δω` with target sparsity `--target_sparsity`.
   - Only the sparse update is uploaded to the server.

In addition, SparsyFed uses the re-parameterization
`θ = sign(ω) * |ω|^β`
before local optimization, where `β` is controlled by `--reparam_beta`.

## Usage
`[$\cdot$]` is mandatory arguments, `{$\cdot$}` is optional arguments.
```
CUDA_VISIBLE_DEVICES=[gpus] bash run_fedsparsy_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round] [comm_round] [epochs] [target_density] [lr] {--target_sparsity, --reparam_beta, --reparam_per_step, --server_lr, --partition_alpha, --dataset_ratio, --num_eval, --frequency_of_the_test, --batch_size, --client_optimizer}

```

where

```
[gpus] specifies which GPUs to use.
[model] is the name of model.
[dataset] is the name of dataset.
[client_num_in_total] is the total number of clients.
[client_num_per_round] is the number of clients selected per round.
[comm_round] is the number of communication rounds.
[epochs] is the number of local epochs.
[target_density] is the target density of the reference sparse weight structure (not communication density). For example, 0.9 means ~90% sparse-layer weights are kept.
[lr] is the client-side learning rate.

{--target_sparsity} is the target sparsity ratio for uploaded model updates (e.g., `0.95` keeps top 5% updates).

{--reparam_beta} is the re-parameterization beta in `θ = sign(ω) * |ω|^β`.
{--reparam_per_step} applies re-parameterization as temporary forward-time weights before every batch.
{--reparam_clip} clamps temporary re-parameterized weights into `[-clip, clip]` (0 disables clamping).
{--batch_metric_log_dir} sets the output directory for per-batch logs. Note: when launching via `run_fedsparsy_distributed_pytorch.sh`, this will be overwritten to the current experiment directory's `batch_metric_logs/`.
{--train_loss_log_interval} controls how often train loss is written to file.
{--act_sparsity_log_interval} controls how often activation sparsity is written to file.
{--server_lr} is the server learning rate used to apply aggregated sparse updates.
{--server_clip_norm} is global norm clipping threshold on server update; `<=0` disables clipping.


{--partition_alpha} controls data heterogeneity. Larger values mean more homogeneous client data.
{--dataset_ratio} is only used for `tinystories` to subsample the dataset for quicker experiments.
{--num_eval} is the number of samples used for validation. `-1` means full test set.
{--frequency_of_the_test} is the evaluation frequency.
{--batch_size} is the local batch size.
{--client_optimizer} is the client optimizer, options: `sgd`, `adam`.
{--loss_based_lr_schedule} enables loss-bucket based client LR schedule.
{--loss_lr_0_1} is the client LR when batch loss is in `[0, 1)`.
{--loss_lr_1_2} is the client LR when batch loss is in `[1, 2)`.
{--loss_lr_2_3} is the client LR when batch loss is in `[2, 3)`.
{--loss_lr_ge_3} is the client LR when batch loss is in `[3, +∞)`.

```

## How to tune the main FL parameters for quick testing
For a fast sanity check, the following settings are recommended:

```
[client_num_in_total] = 10
[client_num_per_round] = 2
[comm_round] = 2 ~ 5
[epochs] = 1
```

Guidelines:

- **`[client_num_in_total]`**
  - Recommended for quick testing: `10`
  - Meaning: total logical client pool size.
  - A smaller value makes debugging easier and reduces data partition complexity.

- **`[client_num_per_round]`**
  - Recommended for quick testing: `2`
  - Meaning: number of clients sampled in each communication round.
  - Smaller values reduce per-round runtime and communication cost.
  - In the current MPI script, this value also determines how many client worker processes are launched.

- **`[comm_round]`**
  - Recommended for quick testing: `2`, `3`, or `5`
  - Meaning: number of federated communication rounds.
  - Use `2` or `3` to quickly verify whether the training / aggregation pipeline runs correctly.
  - Use `5` if you want to observe whether metrics start to stabilize.

- **`[epochs]`**
  - Recommended for quick testing: `1`
  - Meaning: number of local training epochs per selected client in one round.
  - Setting it to `1` is usually enough to verify correctness while keeping runtime short.

A practical CV sanity-check command is:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline bash run_fedsparsy_distributed_pytorch.sh resnet18 cifar10 10 2 3 1 0.1 0.01 --gpu_mapping_key mapping_3_processes --target_sparsity 0.9 --reparam_beta 1.25 --server_lr 1.0 --num_eval 64 --frequency_of_the_test 1 --batch_size 32
```

A practical NLP sanity-check command is:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=offline bash run_fedsparsy_distributed_pytorch.sh gpt2 tinystories 10 2 2 1 0.1 0.001 --gpu_mapping_key mapping_3_processes --dataset_ratio 0.01 --partition_alpha 5 --batch_size 4 --num_eval 32 --target_sparsity 0.9 --reparam_beta 1.25 --server_lr 1.0
```

If you launch only 3 MPI processes in total (1 server + 2 clients), make sure the GPU mapping key also corresponds to 3 processes. In this folder, `mapping_3_processes` is prepared for quick tests.
If you launch 5 MPI processes in total (1 server + 4 clients), use `mapping_5_processes` for medium-scale tests.



## Differences from FedDST

| Feature | FedDST | SparsyFed |
|---------|---------|-----------|
| Forward activations | Dense | Dense |
| Backward activations | Dense | Layer-wise Top-K pruned |
| Activation sparsity target | N/A | Aligned with per-layer weight sparsity |
| Re-parameterization | No | Yes (`β`) |
| Upload content | Weights / masks | Sparse model update `Δω` |
| Upload pruning | Not global Top-K update pruning | Global Top-K update pruning |
| Server update | Weight averaging + mask logic | Weighted update aggregation + server optimizer |
| Mode switching | Yes | No |
| Mask transmission | Yes | No |
