from mpi4py import MPI

from .FedTinyAggregator import FedTinyAggregator
from .FedTinyTrainer import FedTinyTrainer
from .FedTinyClientManager import FedTinyClientManager
from .FedTinyServerManager import FedTinyServerManager

from api.standalone.fedtiny.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from api.standalone.fedtiny.my_model_trainer_language_model import MyModelTrainer as MyModelTrainerLM

# from ...standalone.fedinitprune.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
# from ...standalone.fedinitprune.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedTiny_distributed(
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
    output_dim_global,
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
            output_dim_global,
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
            output_dim_global,
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
    output_dim_global,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        if args.dataset in ["tinystories",]:
            model_trainer = MyModelTrainerLM(model, args.dataset)
        else:
        #Sparse model generates trainer
            model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = FedTinyAggregator(
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

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = FedTinyServerManager(args, model, aggregator,output_dim_global, comm, rank, size, backend)
    else:
        server_manager = FedTinyServerManager(
            args,
            model,
            aggregator,
            output_dim_global,
            comm,
            rank,
            size,
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    if args.ABNS == 1:
        #print('ABNS is on.')
        server_manager.send_ABNS_msg()
        server_manager.send_ABNS_info()
        server_manager.receive_BN_loss()
        #if server_manager.flag_ABNS_complete == True:
        server_manager.run()
    else:
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
    output_dim_global,
    model_trainer=None,
):
    client_index = process_id - 1
    if model_trainer is None:
        if args.dataset in ["tinystories",]:
            model_trainer = MyModelTrainerLM(model, args.dataset)
        # elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        #     model_trainer = MyModelTrainerNWP(model)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedTinyTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedTinyClientManager(args,model, trainer,  output_dim_global, comm, process_id, size, backend)
    # if args.ABNS == True:
    #     #print('ABNS is on.')
    #     client_manager.C2Ssend_ABNS_msg()
    client_manager.run()
