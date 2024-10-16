# FedPruning: A Library and Benchmark for Efficient Federated Pruning.

Federated Learning enables multiple clients to collaboratively train a deep learning model without sharing data, though it often suffers from resource constraints on local devices. Neural network pruning facilitates on-device training by removing redundant parameters from dense networks, significantly reducing computational and storage costs. Recent state-of-the-art Federated Pruning techniques have achieved performance comparable to full-sized models

## Installation 
```python
conda create -n fedprune python=3.10
conda activate fedprune
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda mpi4py
pip install -r requirements.txt
```
