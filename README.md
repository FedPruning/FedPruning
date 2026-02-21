
# FedPruning

**A Library and Benchmark for Efficient Federated Learning with Dynamic Pruning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[**Installation**](#installation) • [**Quick Start**](#quick-start) • [**Documentation**](https://honghuangs-organization.gitbook.io/fedpruning-documents) • [**Supported Methods**](#supported-methods) • [**Citation**](#citation)

---

## 📢 News

- **[2026-02]** 🎉 Our survey [FedPruning](https://www.techrxiv.org/doi/full/10.36227/techrxiv.177074303.30781623/v1), the **first comprehensive survey on federated pruning**, is now available on TechRxiv!
- **[2026-01]** 🎉 FedPruning framework is officially released!
- **[2025-09]** 🎉 Our work [FedRTS](https://arxiv.org/abs/2501.19122), built on FedPruning, has been accepted to **NeurIPS 2025**!

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

## Supported Methods

### ✅ Completed
- [x] FedAVG
- [x] FedTiny
- [x] FedMef
- [x] FedDST
- [x] FedRTS
- [x] PruneFL
- [x] FedSGC

### ⏳ To Do
- [ ] DWNP
- [ ] FedDIP


## Supported Datasets

- [x] **CIFAR-10**
- [x] **CIFAR-100**
- [x] **CINIC-10**
- [x] **SVHN**
- [x] **Tiny-ImageNet**

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

@article{huang2026survey,
  title={A Survey on Efficient Federated Pruning: Progress, Challenges, and Opportunities},
  author={Huang, Hong and Yang, Zhengjie and Chen, Ning and Hu, Juntao and Yang, Jinhai and Liu, Xue and Wu, Dapeng},
  journal={Authorea Preprints},
  year={2026},
  publisher={Authorea}
}

@inproceedings{huangfedrts,
  title={FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling},
  author={Huang, Hong and Yang, Jinhai and Chen, Yuan and Ye, Jiaxun and Wu, Dapeng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}

@inproceedings{huang2024fedmef,
  title={Fedmef: Towards memory-efficient federated dynamic pruning},
  author={Huang, Hong and Zhuang, Weiming and Chen, Chen and Lyu, Lingjuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27548--27557},
  year={2024}
}

@inproceedings{huang2023distributed,
  title={Distributed pruning towards tiny neural networks in federated learning},
  author={Huang, Hong and Zhang, Lan and Sun, Chaoyue and Fang, Ruogu and Yuan, Xiaoyong and Wu, Dapeng},
  booktitle={2023 IEEE 43rd International Conference on Distributed Computing Systems (ICDCS)},
  pages={190--201},
  year={2023},
  organization={IEEE}
}
```
