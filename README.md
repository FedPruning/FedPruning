# FedPruning: A Library and Benchmark for Efficient Federated Pruning.

## Introduction
Federated Learning enables multiple clients to collaboratively train a deep learning model without sharing data, though it often suffers from resource constraints on local devices. Neural network pruning facilitates on-device training by removing redundant parameters from dense networks, significantly reducing computational and storage costs. Recent state-of-the-art Federated Pruning techniques have achieved performance comparable to full-size models.

Our repository, **FedPruning**, serves as an open research library for efficient federated pruning methods. It supports multi-GPU training with multiprocessing capabilities. Moreover, it includes comprehensive datasets and models to facilitate fair comparisons in evaluations. Detailed documentation is available [here](https://honghuangs-organization.gitbook.io/fedpruning-documents).

## Installation 
```python
conda create -n fedpruning python=3.10
conda activate fedpruning
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda mpi4py
pip install -r requirements.txt
```

### Quick Start
To run the simple dynamic pruning baseline methods, FedTiny-Clean, with 100 clients on CIFAR-10 dataset, use the following:
```
cd experiments/distributed/fedtinyclean
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedtinyclean_distributed_pytorch.sh 100 10 resnet18 500 5 64 0.001 cifar10 0.5 0.1 10 300 128 10
```
