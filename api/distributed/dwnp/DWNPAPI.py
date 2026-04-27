from mpi4py import MPI

from .DWNPAggregator import DWNPAggregator
from .DWNPTrainer import DWNPTrainer
from .DWNPClientManager import DWNPClientManager
from .DWNPServerManager import DWNPServerManager

from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_language_model import MyModelTrainer as MyModelTrainerLM


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_DWNP_distributed(
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
    model_trainer=None,
    preprocessed_sampling_lists=None,
):
    if process_id == 0:
        init_server(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            model_trainer,
            preprocessed_sampling_lists,
        )
    else:
        init_client(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )


def init_server(
    args,
    device,
    comm,
    rank,
    size,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        if args.dataset in ["tinystories"]:
            model_trainer = MyModelTrainerLM(model, args)
        else:
            model_trainer = MyModelTrainerCLS(model, args)
    model_trainer.set_id(-1)

    worker_num = size - 1
    aggregator = DWNPAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
    )

    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = DWNPServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = DWNPServerManager(
            args,
            aggregator,
            comm,
            rank,
            size,
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    server_manager.send_init_msg()
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    process_id,
    size,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer=None,
):
    client_index = process_id - 1
    if model_trainer is None:
        if args.dataset in ["tinystories"]:
            model_trainer = MyModelTrainerLM(model, args)
        else:
            model_trainer = MyModelTrainerCLS(model, args)
    model_trainer.set_id(client_index)

    trainer = DWNPTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = DWNPClientManager(args, trainer, comm, process_id, size, args.backend)
    client_manager.run()
