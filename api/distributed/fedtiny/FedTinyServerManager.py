import logging
import os, signal
import sys
import numpy as np
import torch
import random
from api.pruning.model_pruning import SparseModel

from api.model.cv.resnet_gn import resnet18 as resnet18_gn
from api.model.cv.mobilenet import mobilenet
from api.model.cv.resnet import resnet18, resnet56
from api.standalone.fedtiny.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from api.pruning.init_scheme import f_decay
from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
try:
    from core.distributed.communication.message import Message
    from core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedPruning.core.distributed.communication.message import Message
    from FedPruning.core.distributed.server.server_manager import ServerManager

class FedTinyServerManager(ServerManager):
    def __init__(self, args, model, aggregator, output_dim_global, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.model = model
        self.aggregator = aggregator #FedTinyAggregator
        self.output_dim_global = output_dim_global
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.mode = 0 
        self.flag_ABNS_complete = False
        self.ABNS_seed_list = None

        # mode 0, the server send both weight and mask to clients, received the weight, perform weight aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 1, the server send weights, received the weights, perform weights aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 2, the server send weights, received the weights and gradients, perform weights and gradients aggregation, pruning and growing to produce new mask,  go to mode 0
        # TODO (special mode, only for \delta t == 1) mode 3, the server send both weight and mask to clients, received the weight, perform weights and gradients aggregation, pruning and growing to produce new mask,  if t < t_end, go to mode 3 ,else , go to mode 0. 

    def run(self):
        super().run()
    def send_ABNS_msg(self):
        logging.info(f"Server start sending ABNS models")
        
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        #ABNS_seed_list = np.random.randint(10,size=self.args.ABNS_num_of_candidates)
        ABNS_seed_list = [i for i in range(self.args.ABNS_num_of_candidates)]
        self.ABNS_seed_list = ABNS_seed_list
        print(f"ABNS_seed_list is{ABNS_seed_list}")
        # global_model_params = self.aggregator.get_global_model_params()
        # if self.args.is_mobile == 1:
        #     global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_ABNS(process_id, ABNS_seed_list, client_indexes[process_id - 1], self.mode, self.round_idx)
    
    def send_init_msg(self):
        # sampling clients

        # if (self.args.ABNS == True and self.flag_ABNS_complete == True)|self.args.ABNS == False:
        #     print(f"Starting sending init message ")
        # else:
        #     print(f"self.args.ABNS is {self.args.ABNS}, self.flag_ABNS_complete is {self.flag_ABNS_complete}")
        logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                        self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()  #MyModelTrainerCLS(model).get_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):   
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1], self.mode, self.round_idx)
    def send_ABNS_info(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_BN_PARAMS_TO_SERVER,
                self.handle_message_receive_BN_params_from_client)
    def receive_BN_loss(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_BN_LOSS_TO_SERVER,
                self.handle_message_receive_BN_loss_from_client)
        
    def register_message_receive_handlers(self):
        # if self.args.ABNS == True:
        #     self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_BN_PARAMS_TO_SERVER,
        #             self.handle_message_receive_BN_params_from_client)
                    
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                self.handle_message_receive_model_from_client)

    def mode_convert(self,):
        if self.mode == 0:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 1:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 2:
            self.mode = 0
        elif self.mode == 3:
            if self.round_idx < self.args.T_end :
                self.mode = 3
            else:
                self.mode = 0

        return self.mode
    def handle_message_receive_BN_params_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        bn_params_list = msg_params.get(MyMessage.MSG_ARG_KEY_BN_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        print(f"The sender_id is {sender_id}. The local_sample_number is {local_sample_number}. Receiving bn_params_list.")
        #print(f"The sender_id is {sender_id}. The local_sample_number is {local_sample_number}. The bn_params_list is {bn_params_list}.")
        self.aggregator.add_local_trained_BN_params(sender_id - 1, bn_params_list, local_sample_number)

        b_BN_params_all_received = self.aggregator.check_whether_BN_params_all_receive()
        logging.info("b_BN_params_all_received = " + str(b_BN_params_all_received))
        if b_BN_params_all_received:
            global_BN_params = self.aggregator.aggregate_BN_params()
            #print(f"Len of global_BN_params is {len(global_BN_params)}, global_BN_params is {global_BN_params} ")
            # send global BN params to client
            #self.send_S2C_BN_params(self,global_BN_params)

            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                    self.args.client_num_per_round)

            for receiver_id in range(1, self.size):
                self.send_message_BN_params_to_client(receiver_id, global_BN_params,
                            client_indexes[receiver_id - 1], self.mode, self.round_idx)
                
    def handle_message_receive_BN_loss_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        bn_loss_list = msg_params.get(MyMessage.MSG_ARG_KEY_BN_LOSS_LIST)
        print(f"The sender_id is {sender_id}. The bn_loss_list is {bn_loss_list}.")
        self.aggregator.add_local_trained_BN_loss(sender_id - 1, bn_loss_list)

        b_BN_loss_all_received = self.aggregator.check_whether_BN_loss_all_receive()
        logging.info("b_BN_loss_all_received = " + str(b_BN_loss_all_received))
        if b_BN_loss_all_received:
            global_loss = self.aggregator.aggregate_BN_loss()
            print(f"Len of global_BN_loss is {len(global_loss)}, global_BN_loss is {global_loss} ")
            min_loss = global_loss[0]
            min_loss_index = 0
            for i in range(len(global_loss)):
                if global_loss[i] < min_loss:
                    min_loss = global_loss[i]
                    min_loss_index = i
            
            #-----------------------------------------------------------
            print(f"ABNS_seed_list index is {min_loss_index}, value is {self.ABNS_seed_list[min_loss_index]}")


            #----------------------------TEST for aggregator
            model = self.model
            model.init_weights(seed = self.ABNS_seed_list[min_loss_index], init_method = "kaiming_normal")
            model_trainer = MyModelTrainerCLS(model)
            model_trainer.set_id(-1)
            self.aggregator.trainer = model_trainer

            #------------------------------
            model_params = model.cpu().state_dict()
            self.aggregator.set_global_model_params(model_params)
            self.flag_ABNS_complete = True
            print(f"self.flag_ABNS_complete is {self.flag_ABNS_complete}")
            self.send_init_msg()


    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)#5
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)#
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)#264
        if self.mode in [2, 3]:
            gradients = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADIENT)
            #print(f"gradients is {gradients}")
            # if aaa == 0:
            #     print(aaa)
            # select topk gradients
            if self.args.progressive_pruning == 1: 
                #print('Progressive pruning is on.')
                gradients = self.get_topk_gradients(gradients)
            # add gradient
            self.aggregator.add_local_trained_gradient(sender_id - 1, gradients)
            
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            if self.mode in [2, 3]:
                global_gradient = self.aggregator.aggregate_gradient()
                # update the global model which is cached at the server side
                self.aggregator.trainer.model.adjust_mask_dict(global_gradient, t=self.round_idx, T_end=self.args.T_end, alpha=self.args.adjust_alpha)
                self.aggregator.trainer.model.to(self.aggregator.device)
                self.aggregator.trainer.model.apply_mask()
                 
            # logging.info("mask_dict after pruning and growing = " +str(mask_dict))
            self.aggregator.test_on_server_for_all_clients(self.round_idx)
            
            # start the next round
            self.round_idx += 1

            # convert the mode 
            self.mode = self.mode_convert()

            if self.round_idx == self.round_num + 1:
                # post_complete_message_to_sweep_process(self.args)
                self.finish()
                print('here')
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                    self.args.client_num_per_round)

            print('indexes of clients: ' + str(client_indexes))
            print("size = %d" % self.size)
            logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)
            
            if self.mode in [0, 3]:
                mask_dict = self.aggregator.trainer.model.mask_dict #MyModelTrainer(model).model.mask_dict/SparseModel.mask_dict
                for k in mask_dict:
                    mask_dict[k] = mask_dict[k].cpu()
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx, mask_dict)
            else:
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx)

    def send_message_ABNS(self, receive_id, ABNS_seed_list, client_index, mode_code, round_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_ABNS, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ABNS_SEED, ABNS_seed_list)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        self.send_message(message)

    def send_message_init_config(self, receive_id, global_model_params, client_index, mode_code, round_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        self.send_message(message)

    def send_message_BN_params_to_client(self, receive_id, global_BN_params, client_index, mode_code, round_idx, mask_dict=None):
        logging.info("send_message_sync_BN_params_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_BN_PARAMS_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_BN_PARAMS, global_BN_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_MASKS, mask_dict)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, mode_code, round_idx,
        mask_dict=None):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_MASKS, mask_dict)
        self.send_message(message)

    def get_topk_gradients(self, gradients):
        for name, param in self.aggregator.trainer.model.model.named_parameters():
                mask_dict = self.aggregator.trainer.model.mask_dict
                if name in mask_dict:  
                    active_num = (mask_dict[name] == 1).int().sum().item()
                    k = int(f_decay(t=self.round_idx, T_end=self.args.T_end, alpha=self.args.adjust_alpha) * active_num)
                    print('Attention: acitive_num: ', active_num,'k: ',k)
                    # Find the k  largest gradients connections among the currently inactive connections
                    inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1).cpu()
                    
                    grad_inactive = gradients[name].abs().view(-1)[inactive_indices].cpu()

                    print(f"Attention: Length of grad_inactive for parameter {name}: {len(grad_inactive)}")
                    if len(grad_inactive) >= k:
                        _, topk_indices = torch.topk(grad_inactive, k, sorted=False)
                    else:
                        print("k is smaller than number of grad_inactive, skip topk gradients")
                        _, topk_indices = torch.topk(grad_inactive, len(grad_inactive), sorted=False)
                    mask_gradients = torch.zeros(gradients[name].view(-1).shape, dtype=torch.bool)
                    for idx in topk_indices:
                        mask_gradients[idx] = True 
                    # zero_indices = [idx for idx in all_indices if idx not in grow_indices]
                    # gradients[name].view(-1)[:]
                    #print("original gradient:",gradients[name],"non_zero_num: ",torch.count_nonzero(gradients[name]))
                    gradients[name].view(-1)[~mask_gradients.cpu()] = 0
                    #print("topk gradient:",gradients[name],"non_zero_num: ",torch.count_nonzero(gradients[name]))
        return gradients

