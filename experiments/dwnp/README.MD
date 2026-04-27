# DWNP Experiment

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_dwnp_distributed_pytorch.sh resnet18 cifar10 100 10 200 5 0.5 0.001 \
  --r_w 1 --r_hn 5 --r_theta 2 \
  --server_flops_retention 0.5 --device_flops_retention 0.9 --lambda_reg 1.0 \
  --dwnp_warmup_rounds 5 --hn_min_retention 0.6 --hn_max_retention 1.0 \
  --hn_smooth_coeff 0.1 --task_loss_alpha 0.0 --hn_temperature 0.5 --hn_stochastic_gate 0 \
  --dwnp_enable_final_prune 1 \
  --partition_alpha 0.5 --frequency_of_the_test 5
```

Example command used in this repo (wandb disabled):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash
run_dwnp_distributed_pytorch.sh resnet18 cifar10 100 10 200 5 1.0 0.05
```

## Usage

```bash
CUDA_VISIBLE_DEVICES=[gpus] sh run_dwnp_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round] [comm_round] [epochs] [target_density] [lr] {optional args}
```

Important optional args:

- `--r_w`: model communication interval.
- `--r_hn`: hypernetwork local training frequency.
- `--r_theta`: hypernetwork communication interval.
- `--server_flops_retention`: target FLOPs retention for server mask.
- `--device_flops_retention`: target FLOPs retention for device mask.
- `--lambda_reg`: coefficient for FLOPs regularization.
- `--hn_smooth_coeff`: optional smoothness penalty for neighboring HN outputs.
- `--task_loss_alpha`: optional weight for the detached task loss term in HN training.
- `--dwnp_warmup_rounds`: warmup rounds before enabling device-wise extra pruning.
- `--hn_min_retention`, `--hn_max_retention`: lower and upper bounds for the predicted retention ratios.
- `--hn_lr`: hypernetwork learning rate.
- `--hn_local_epochs`: local epochs when training hypernetwork.
- `--hn_embedding_dim`, `--hn_hidden_dim`: hypernetwork architecture.
- `--hn_temperature`: ST Gumbel-Sigmoid temperature.
- `--hn_stochastic_gate`: whether to keep the CNN-side mask application stochastic during training.
- `--use_hn_server_mask`: whether the server-side mask is updated from the HN during the main training rounds.
- `--dwnp_enable_final_prune`: whether to run one final server-side prune and broadcast the pruned model before the last fine-tuning round.

Notes:

- The current implementation keeps the original distributed structure used by the other FedPruning methods: server aggregation, client-local training, and periodic synchronization remain unchanged.
- The final pruning round is still part of the same distributed loop; it does not add a new standalone script or message schema.
- The final server-side pruning pass is exported with deterministic gates (non-stochastic) for stable reproduction and deployment.
- `hn_stochastic_gate` controls stochasticity during CNN-side joint-mask preparation, while HN training uses ST Gumbel-Sigmoid sampling.
