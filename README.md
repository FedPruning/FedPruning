<div align="center">


# FedPruning

**A Library and Benchmark for Efficient Federated Learning with Dynamic Pruning**

Federated Learning enables multiple clients to collaboratively train a deep learning model without sharing data, though it often suffers from resource constraints on local devices. Neural network pruning facilitates on-device training by removing redundant parameters from dense networks, significantly reducing computational and storage costs. Recent state-of-the-art Federated Pruning techniques have achieved performance comparable to full-size models.

Our repository, **FedPruning**, serves as an open research library for efficient federated pruning methods. It supports multi-GPU training with multiprocessing capabilities. Moreover, it includes comprehensive datasets and models to facilitate fair comparisons in evaluations. Detailed documentation is available [here](https://honghuangs-organization.gitbook.io/fedpruning-documents).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[**Installation**](#installation) | [**Quick Start**](#quick-start) | [**Documentation**](#documentation) | [**Supported Models**](#supported-models) | [**Citation**](#citation) | [**Contact**](#contact)

</div>

---

## News

- **[2026-01]** 🎉 Initial release of FedPruning framework
- **[2025-09]** Paper [FedRTS](https://arxiv.org/abs/2501.19122) accepted at NeurIPS



## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.0 (for GPU support)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/FedPruning/FedPruning.git
cd FedPruning

# Create a virtual environment (recommended)
conda create -n fedpruning python=3.10 -y
conda activate fedpruning

# Install PyTorch (CUDA 11.8)
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install MPI dependency
conda install -y -c anaconda mpi4py

# Install other Python dependencies
pip install -r requirements.txt
```



## Quick Start

### Command Line Interface

```bash
cd experiments/fedtinyclean
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedtinyclean_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10
```



## Documentation

### Project Structure

```
FedPruning/
├── api/
│   ├── data_preprocessing/
│   ├── distributed/
│   ├── model/
│   │   ├── cv/
│   │   ├── finance/
│   │   ├── linear/
│   │   ├── mobile/
│   │   ├── nlp/
│   ├── pruning/
│   └── utils/
├── core/
│   ├── distributed/
│   ├── non_iid_partition/
│   ├── robustness/
│   ├── trainer/
├── data/
├── experiments/
```



## Supported Datasets

- **CIFAR-10**
- **CIFAR-100**
- **CINIC-10**
- **SVHN**



## Supported Models

### **Computer Vision (`model/cv/`)** 

- ResNet
- VGG (`vgg.py`)
- MobileNet (`mobilenet.py`, `mobilenet_v3.py`)
- EfficientNet (`efficientnet.py`)
- Basic CNN (`cnn.py`)
- MNIST GAN (`mnist_gan.py`) 

### **Linear (`model/linear/`)** 

- Logistic Regression (`lr.py`) 

### **Mobile (`model/mobile/`)** 

- LeNet variants. 

### **NLP (`model/nlp/`)** 

- GPT-2 (`gpt2.py`) 
- RNN (`rnn.py`)



## Experiments

### Reproducing Paper Results

```bash
# Take FedRTS as an example
# CV Task 
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedrts_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.5 0.001 --delta_T 10 --partition_alpha 0.5 --adjust_alpha 0.2

CUDA_VISIBLE_DEVICES=4,5,6,7 sh run_fedrts_distributed_pytorch.sh resnet18 cifar10 100 10 500 5 0.1 0.001 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --aggregated_gamma 0.5 --initial_distribution_ratio 1.0 --client_optimizer adam

# NLP task
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedrts_distributed_pytorch.sh gpt2 tinystories 100 10 200 1 0.1 0.1 --delta_T 10 --T_end 100 --num_eval 128 --frequency_of_the_test 10 --partition_alpha 5 --batch_size 16

CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedrts_distributed_pytorch.sh gpt2 tinystories 100 10 500 5 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 --partition_alpha 5 --batch_size 16
```



## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{huang2025fedrts,
  title={Fedrts: Federated robust pruning via combinatorial thompson sampling},
  author={Huang, Hong and Yang, Hai and Chen, Yuan and Ye, Jiaxun and Wu, Dapeng},
  journal={arXiv preprint arXiv:2501.19122},
  year={2025}
}
```



## Contact

- **Maintainer**: [Hong Huang: hohuang-c@my.cityu.edu.hk]
- **Issues**: Please report bugs and feature requests via [GitHub Issues](https://github.com/FedPruning/FedPruning/issues)

<div align="center">
<b> Star ⭐ this repository if you find it helpful! </b>

</div>
