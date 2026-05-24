import logging
import os
import sys
import numpy as np
import random
import torch
from api.model.cv.resnet_gn import resnet18 as resnet18_gn
from api.model.cv.mobilenet import mobilenet
from api.model.cv.resnet import resnet18, resnet56
#from experiments.distributed.fedtiny.main_fedtiny import create_model
from api.pruning.model_pruning import SparseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

try:
    from core.distributed.client.client_manager import ClientManager
    from core.distributed.communication.message import Message
except ImportError:
    from FedPruning.core.distributed.client.client_manager import ClientManager
    from FedPruning.core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process


class FedTinyClientManager(ClientManager):
    def __init__(self, args, model, trainer, output_dim_global,comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.model = model
        self.ABNS_seed_list = []
        self.trainer = trainer #FedTinyTrainer
        self.output_dim_global = output_dim_global
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.mode = 0

    # mode 0: the client receieve both weight and mask to clients;  config the new weight and mask, train; send the weight
    # mode 1: the client receieve weights;  config the new weight, train;  send the weight,
    # mode 2: the client receieve weights;  config the new weight; train, compute the gradients; send the weight and gradients
    # TODO (special mode, only for \delta t == 1) mode 3: receieve both weight and mask to clients; config the new weight and mask, train, compute the gradients; send the weight and gradients. 

    def run(self):
        super().run()
#    def C2Ssend_ABNS_msg(self):
        
        
    def register_message_receive_handlers(self):
        if self.args.ABNS == 1:
            self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_ABNS,
                                              self.handle_message_ABNS_seed)
            self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_BN_PARAMS_TO_CLIENT,
                                              self.handle_message_BN_params)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)
        
    def handle_message_ABNS_seed(self, msg_params):
        ABNS_seed_list = msg_params.get(MyMessage.MSG_ARG_KEY_ABNS_SEED)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE_CODE)
        self.round_idx =  msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)
        self.ABNS_seed_list = ABNS_seed_list
        bn_parameters_list = []
        local_sample_num = None
        # if self.args.is_mobile == 1:
        #     global_model_params = transform_list_to_tensor(global_model_params)
        for i in range(len(ABNS_seed_list)):
            print("ABNS_seed_list in client is",ABNS_seed_list[i])
            model = self.model
            model.init_weights(seed = ABNS_seed_list[i], init_method = "kaiming_normal")
            model_params = model.cpu().state_dict()
            # 
            #logging.info(f"The type is {type(model_params)}, the params of model {i} is {model_params}")
            #print(f"The type is {type(model_params)}, the params of model {i} is {model_params}")
            self.trainer.update_model(model_params)
            self.trainer.update_dataset(int(client_index))
            bn_parameters, local_sample_num = self.__train_BN()
            bn_parameters_list.append(bn_parameters)

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_BN_PARAMS_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_BN_PARAMS, bn_parameters_list)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)    


    def handle_message_BN_params(self, msg_params):
        logging.info("handle_message_BN_params_from_server.")
        global_BN_params = msg_params.get(MyMessage.MSG_ARG_KEY_BN_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE_CODE)
        self.round_idx =  msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)

        loss_BN = []
        for c in range(len(global_BN_params)):
            model = self.model
            model.init_weights(seed = self.ABNS_seed_list[c], init_method = "kaiming_normal")
            model_params = model.cpu().state_dict()
            self.trainer.update_model(model_params)
            self.trainer.update_model(global_BN_params[c])
            self.trainer.update_dataset(int(client_index))
            _, loss_tmp, _ =self.trainer.test()
            loss_BN.append(loss_tmp)
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_BN_LOSS_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_BN_LOSS_LIST, loss_BN)
        self.send_message(message)


    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE_CODE)
        self.round_idx =  msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)
        
        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE_CODE)
        self.round_idx =  msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        if self.mode in [0, 3]:
            mask_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_MASKS)
            self.trainer.trainer.model.mask_dict = mask_dict
            self.trainer.trainer.model.apply_mask()
            
        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.__train()
        if self.round_idx == self.num_rounds:
            # post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, gradient=None):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADIENT, gradient)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)


    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, gradient, local_sample_num = self.trainer.train(mode = self.mode, round_idx=self.round_idx)
        if self.mode in [2, 3]:
            self.send_model_to_server(0, weights, local_sample_num, gradient)
        else:
            self.send_model_to_server(0, weights, local_sample_num)

    def __train_BN(self):
        logging.info("#######training with BN########### round_id = %d" % self.round_idx)
        bn_parameters, local_sample_num = self.trainer.train_BN()#FedTinyTrainer.train()
        return bn_parameters, local_sample_num
    