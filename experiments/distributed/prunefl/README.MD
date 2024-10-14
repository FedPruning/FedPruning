export WANDB_DIR='./../../../../wandb/'
export WANDB_CACHE_DIR='./../../../../wandb/.cache/'
export WANDB_CONFIG_DIR='./../../../../wandb/.config/'
wandb login a43329fc912eff90a92976498088770f5745bb57
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_prunefl_distributed_pytorch.sh 10 10 resnet18 hetero 200 1 64 0.001 cifar10 ./../../../data/cifar10 adam 0 0.5 10 100
