CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedinitprune_distributed_pytorch.sh 10 10 resnet18 hetero 200 1 64 0.001 cifar10 ./../../../data/cifar10 adam 0 0.5
