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

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10_ust
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10

from api.model.cv.resnet_gn import resnet18 as resnet18_gn
from api.model.cv.mobilenet import mobilenet
from api.model.cv.resnet import resnet18, resnet56

from api.distributed.fedust.FedUSTAPI import FedML_init, FedML_FedUST_distributed
from api.pruning.model_pruning import SparseModel


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--model", type=str, default="resnet56", metavar="N", help="neural network used in training")

    parser.add_argument("--dataset", type=str, default="cifar10", metavar="N", help="dataset used for training")

    parser.add_argument(
        "--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha (default: 0.5)"
    )

    parser.add_argument(
        "--client_num_in_total", type=int, default=10, metavar="NN", help="number of workers in a distributed cluster"
    )

    parser.add_argument("--client_num_per_round", type=int, default=10, metavar="NN", help="number of workers")

    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument(
        "--num_eval", type=int, default=128, help="the number of the data samples used for eval, -1 is the total testing dataset."
    )
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument("--epochs", type=int, default=5, metavar="EP", help="how many epochs will be trained locally")

    parser.add_argument("--A_epochs", type=int, default=2, metavar="EP", help="how many epochs will be trained before pruning and growing ")

    parser.add_argument("--comm_round", type=int, default=10, help="how many round of communications we shoud use")

    parser.add_argument("--frequency_of_the_test", type=int, default=5, help="the frequency of the algorithms")

    parser.add_argument('--target_density', type=float, default=0.5,
                        help='pruning target density')

    parser.add_argument('--delta_T', type=int, default=10, help='delta t for update')

    parser.add_argument('--T_end', type=int, default=100, help='end of time for update')

    parser.add_argument("--adjust_alpha", type=float, default=0.2, help='the ratio of num elements for adjustments')

    parser.add_argument("--forgotten_sigma", type=float, default=0.2, help='sigma for forgotten set')

    # Following arguments are seldom changed
    parser.add_argument(
        "--gpu_mapping_key", type=str, default="mapping_default", help="the key in gpu utilization file"
    )
    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument(
            "--gpu_mapping_file",
            type=str,
            default="gpu_mapping.yaml",
            help="the gpu utilization file for servers and clients. If there is no \
                            gpu_util_file, gpu will not be used.",
        )

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")

    parser.add_argument("--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server")

    parser.add_argument(
        "--is_mobile", type=int, default=1, help="whether the program is running on the FedML-Mobile server side"
    )

    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0.001)

    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument("--data_dir", type=str, default=None, help="data directory")

    parser.add_argument("--client_optimizer", type=str, default="sgd", help="SGD with momentum; adam")

    parser.add_argument("--growth_data_mode", type=str, default="batch", help=" the number of data samples used for parameter growth, option are [ 'random', 'single', 'batch', 'entire']" )

    args = parser.parse_args()  
    return args


def load_data(args, dataset_name):

    if args.data_dir is None:
        args.data_dir = f"./../../../data/{dataset_name}"

    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10_ust
    elif dataset_name == "cifar100":
        data_loader = load_partition_data_cifar100
    elif dataset_name == "cinic10":
        data_loader = load_partition_data_cinic10
    else:
        data_loader = load_partition_data_cifar10_ust

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = data_loader(
        args.dataset,
        args.data_dir,
        args.partition_method,
        args.partition_alpha,
        args.client_num_in_total,
        args.batch_size,
        )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "resnet18_gn":
        model = resnet18_gn(num_classes=output_dim)
    if model_name == "resnet18":
        model = resnet18(class_num=output_dim)
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    return model

if __name__ == "__main__":
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedUST (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
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

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            project="FedPruning",
            name="FedUST_"
            + args.dataset
            + "_"
            + args.model
            ,
            config=args,
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key
    )

    # load data
    dataset = load_data(args, args.dataset)

    [
        train_data_num,
        test_data_num,
        train_data_global, # None here 
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict, 
        test_data_local_dict, 
        class_num,
    ] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./experiments/distributed_fedprune)
    inner_model = create_model(args, model_name=args.model, output_dim=dataset[7])
    # create the sparse model
    model = SparseModel(inner_model, target_density=args.target_density, )

    # start distributed training
    FedML_FedUST_distributed(
        process_id,
        worker_number,
        device,
        comm,
        model,
        train_data_num,
        None, # We do net need train_data_global, so we set it as None
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        None, # We do net need test_data_local_dict, so we set it as None
        args,
    )