import argparse
import logging
import os
import random
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from api.data_preprocessing.svhn.data_loader import load_partition_data_svhn
from api.data_preprocessing.tinystories.data_loader import load_partition_data_tinystories
from api.data_preprocessing.tinyimagenet.data_loader import load_partition_data_tiny

from api.model.cv.resnet_gn import resnet18 as resnet18_gn
from api.model.cv.mobilenet import mobilenet
from api.model.cv.resnet import resnet18, resnet56
from api.model.nlp.gpt2 import GPT2Model, GPT2Config
from torchvision.models import mobilenet_v3_small as MobileNetV3
from torchvision.models import efficientnet_v2_s as EfficientNetV2
from torchvision.models import mnasnet0_75 as MNASNet
from torchvision.models import shufflenet_v2_x0_5 as ShuffleNet
from torchvision.models import squeezenet1_1 as SqueezeNet
from torchvision.models import swin_t as SwinT
from torchvision.models import vit_b_16 as ViT

from api.distributed.dwnp.DWNPAPI import FedML_DWNP_distributed, FedML_init
from api.pruning.model_pruning import SparseModel


def add_args(parser):
    parser.add_argument("--model", type=str, default="resnet56", metavar="N", help="neural network used in training")
    parser.add_argument("--dataset", type=str, default="cifar10", metavar="N", help="dataset used for training")

    parser.add_argument("--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha")
    parser.add_argument("--partition_method", type=str, default="hetero", metavar="N", help="partition method")

    parser.add_argument("--client_num_in_total", type=int, default=10, metavar="NN", help="number of clients")
    parser.add_argument("--client_num_per_round", type=int, default=10, metavar="NN", help="clients per round")

    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size")
    parser.add_argument("--epochs", type=int, default=5, metavar="EP", help="local epochs")
    parser.add_argument("--comm_round", type=int, default=100, help="communication rounds")

    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate")
    parser.add_argument("--wd", type=float, default=0.001, help="weight decay")
    parser.add_argument("--client_optimizer", type=str, default="sgd", help="sgd or adam")

    parser.add_argument("--frequency_of_the_test", type=int, default=5, help="test frequency")
    parser.add_argument("--num_eval", type=int, default=128, help="validation sample count, -1 uses full test set")

    parser.add_argument("--pruning_strategy", type=str, default="ERK_magnitude", help="sparsity strategy")
    parser.add_argument("--target_density", type=float, default=0.5, help="base sparse model density")

    parser.add_argument("--r_w", type=int, default=1, help="model communication interval")
    parser.add_argument("--r_hn", type=int, default=5, help="hypernetwork local update frequency")
    parser.add_argument("--r_theta", type=int, default=1, help="hypernetwork communication interval")

    parser.add_argument("--server_flops_retention", type=float, default=0.5, help="server-side FLOPs retention target")
    parser.add_argument("--device_flops_retention", type=float, default=0.9, help="device-side FLOPs retention target")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="FLOPs regularization coefficient")
    parser.add_argument("--hn_smooth_coeff", type=float, default=0.1, help="smoothness coefficient for HN output")
    parser.add_argument("--dwnp_warmup_rounds", type=int, default=0, help="warmup rounds before enabling device-wise extra pruning")
    parser.add_argument("--use_hn_server_mask", type=int, default=0, help="whether to let HN update server mask")

    parser.add_argument("--hn_lr", type=float, default=0.0005, help="hypernetwork learning rate")
    parser.add_argument("--hn_local_epochs", type=int, default=1, help="local epochs for hypernetwork update")
    parser.add_argument("--hn_subset_batches", type=int, default=2, help="subset batches for HN local objective")
    parser.add_argument("--hn_embedding_dim", type=int, default=32, help="embedding dim in hypernetwork")
    parser.add_argument("--hn_hidden_dim", type=int, default=64, help="hidden dim in hypernetwork")
    parser.add_argument("--hn_temperature", type=float, default=0.5, help="ST gumbel-sigmoid temperature")
    parser.add_argument("--hn_stochastic_gate", type=int, default=0, help="whether to sample stochastic gates during CNN updates")
    parser.add_argument("--hn_min_retention", type=float, default=0.6, help="minimum per-layer retention predicted by HN")
    parser.add_argument("--hn_max_retention", type=float, default=1.0, help="maximum per-layer retention predicted by HN")
    parser.add_argument("--task_loss_alpha", type=float, default=0.0, help="weight of detached task loss in HN objective")
    parser.add_argument("--dwnp_enable_final_prune", type=int, default=1, help="whether to run a final server-side prune and finetune round")

    parser.add_argument("--dataset_ratio", type=float, default=0.05, metavar="PA", help="subset ratio for tinystories")
    parser.add_argument("--nlp_hidden_size", type=int, default=256, metavar="N", help="hidden size for GPT2")

    parser.add_argument("--gpu_mapping_key", type=str, default="mapping_default", help="gpu mapping key")
    parser.add_argument("--gpu_mapping_file", type=str, default="gpu_mapping.yaml", help="gpu mapping file")
    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu server num")
    parser.add_argument("--gpu_num_per_server", type=int, default=4, help="gpu num per server")

    parser.add_argument("--is_mobile", type=int, default=1, help="whether running on FedML-Mobile server")
    parser.add_argument("--backend", type=str, default="MPI", help="backend")
    parser.add_argument("--data_dir", type=str, default=None, help="data directory")
    parser.add_argument("--ci", type=int, default=0, help="CI")

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if args.data_dir is None:
        if dataset_name == "tinyimagenet":
            args.data_dir = "./../../data/Tiny-ImageNet"
        else:
            args.data_dir = f"./../../data/{dataset_name}"

    if dataset_name == "tinystories":
        dataset_tuple = load_partition_data_tinystories(
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            args.dataset_ratio,
        )
    elif dataset_name == "tinyimagenet":
        dataset_tuple = load_partition_data_tiny(
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
        )
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        elif dataset_name == "svhn":
            data_loader = load_partition_data_svhn
        else:
            data_loader = load_partition_data_cifar10

        dataset_tuple = data_loader(
            args.dataset,
            args.data_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
        )

    return dataset_tuple


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s", model_name, output_dim)
    model = None
    if model_name == "resnet18_gn":
        model = resnet18_gn(num_classes=output_dim)
    if model_name == "resnet18":
        model = resnet18(class_num=output_dim)
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenetv3":
        model = MobileNetV3(num_classes=output_dim)
    elif model_name == "efficientnet":
        model = EfficientNetV2(num_classes=output_dim)
    elif model_name == "shufflenet":
        model = ShuffleNet(num_classes=output_dim)
    elif model_name == "squeezenet":
        model = SqueezeNet(num_classes=output_dim)
    elif model_name == "swint":
        model = SwinT(num_classes=output_dim)
    elif model_name == "vit":
        model = ViT(image_size=32, num_classes=output_dim)
    elif model_name == "mnasnet":
        model = MNASNet(num_classes=output_dim)
    elif model_name == "gpt2":
        GPT2Config["hidden_size"] = args.nlp_hidden_size
        model = GPT2Model(GPT2Config)
        logging.info("number of parameters: %.2fM", model.get_num_params() / 1e6)
    else:
        raise Exception(f"{model_name} is not found !")
    return model


if __name__ == "__main__":
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    comm, process_id, worker_number = FedML_init()

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    str_process_name = "DWNP (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    logging.basicConfig(
        level=logging.DEBUG,
        format=str(process_id) + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )

    hostname = socket.gethostname()
    logging.info(
        "#############process ID = "
        + str(process_id)
        + ", host name = "
        + hostname
        + "########"
        + ", process ID = "
        + str(os.getpid())
        + ", process Name = "
        + str(psutil.Process(os.getpid()))
    )

    if process_id == 0:
        wandb.init(
            project="FedPruning",
            name="DWNP_" + args.dataset + "_" + args.model,
            config=args,
        )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    logging.info("process_id = %d, size = %d", process_id, worker_number)
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id,
        worker_number,
        args.gpu_mapping_file,
        args.gpu_mapping_key,
    )

    dataset = load_data(args, args.dataset)

    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    args.num_classes = class_num

    inner_model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model = SparseModel(inner_model, target_density=args.target_density, strategy=args.pruning_strategy)

    FedML_DWNP_distributed(
        process_id,
        worker_number,
        device,
        comm,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        args,
    )
