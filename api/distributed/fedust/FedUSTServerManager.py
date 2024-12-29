import logging
import os, signal
import sys

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process
from api.pruning.init_scheme import pruning
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
try:
    from core.distributed.communication.message import Message
    from core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedPruning.core.distributed.communication.message import Message
    from FedPruning.core.distributed.server.server_manager import ServerManager

class FedUSTServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.mode = 0 

        # mode 0, the server send both weight and mask to clients, received the weight, perform weight aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 1, the server send weights, received the weights, perform weights aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 2, the server send weights, received the weights and masks, perform weights and masks aggregation, pruning and growing to produce new mask,  go to mode 0
        # TODO (special mode, only for \delta t == 1) mode 3, the server send both weight and mask to clients, received the weights and masks, perform weights and masks aggregation, pruning and growing to produce new mask,  if t < t_end, go to mode 3 ,else , go to mode 0.

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients

        logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1], self.mode, self.round_idx)

    def register_message_receive_handlers(self):
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
    
    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        if self.mode in [2, 3]:
            masks = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_MASKS)
            self.aggregator.add_local_trained_mask(sender_id - 1, masks)
            
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            logging.info(f"current mode for server is {self.mode}, the round is {self.round_idx}")
            if self.mode in [2, 3]:
                model = self.aggregator.trainer.model
                global_mask = self.aggregator.aggregate_mask()
                # # prune to reach density
                layer_density_strategy, pruning_strategy = model.strategy.split("_")
                new_global_mask = pruning(model, model.layer_density_dict, pruning_strategy, mask_dict=global_mask)
                model.mask_dict = new_global_mask
                model.to(self.aggregator.device)
                model.apply_mask()
                
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
                mask_dict = self.aggregator.trainer.model.mask_dict
                for k in mask_dict:
                    mask_dict[k] = mask_dict[k].cpu()
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx, mask_dict)
            else:
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx)


    def send_message_init_config(self, receive_id, global_model_params, client_index, mode_code, round_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
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
