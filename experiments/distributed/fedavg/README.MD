
## Usage
```
mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_default" \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --initial_lr $LR \
  --ci $CI
```


## ResNet18 Federated Training

#### CIFAR10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 resnet18 hetero 500 5 64 0.001 cifar10 ./../../../data/cifar10 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/m6m5l1kp?nw=nwuserzhihaoyu6
****
Testing Accuracy:0.718
****



#### CIFAR100

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 resnet18 hetero 500 5 64 0.001 cifar100 ./../../../data/cifar100 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/dz7ublqn?nw=nwuserzhihaoyu6
****
Testing Accuracy:0.445
****

#### CINIC10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 resnet18 hetero 500 5 64 0.001 cinic10 ./../../../data/cinic10 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/swds9tfd?nw=nwuserzhihaoyu6
****
Testing Accuracy:0.601
****

## ResNet56 Federated Training

#### CIFAR10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 resnet56 hetero 500 5 64 0.001 cifar10 ./../../../data/cifar10 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/p55t75wd?nw=nwuserzhihaoyu6
****
Testing Accuracy: 0.804
****



#### CIFAR100
train on IID dataset 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 resnet56 hetero 500 5 64 0.001 cifar100 ./../../../data/cifar100 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/6zq0d35p?nw=nwuserzhihaoyu6
****
Testing Accuracy: 0.539
****

#### CINIC10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 resnet56 hetero 500 5 64 0.001 cinic10 ./../../../data/cinic10 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/16jfqepj?nw=nwuserzhihaoyu6
****
Testing Accuracy: 0.664
****



## MobileNet Federated Training


#### CIFAR10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 mobilenet hetero 500 5 64 0.001 cifar10 ./../../../data/cifar10 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/z04old5k?nw=nwuserzhihaoyu6
****
Testing Accuracy: 0.898
****


#### CIFAR100

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 mobilenet hetero 500 5 64 0.001 cifar100 ./../../../data/cifar100 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/yznynjub?nw=nwuserzhihaoyu6
****
Testing Accuracy: 0.445
****
#### CINIC10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedavg_distributed_pytorch.sh 100 10 mobilenet hetero 500 5 64 0.001 cinic10 ./../../../data/cinic10 adam 

```
reference experimental result: https://wandb.ai/zhyu9/fedprune/runs/tmpg7f1a?nw=nwuserzhihaoyu6
****
Testing Accuracy: 0.664
****
