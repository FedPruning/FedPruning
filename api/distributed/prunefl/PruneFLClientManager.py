import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

try:
    from core.distributed.client.client_manager import ClientManager
    from core.distributed.communication.message import Message
except ImportError:
    from FedPruning.core.distributed.client.client_manager import ClientManager
    from FedPruning.core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process


class PruneFLClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.mode = 0
        self.current_client_index = -1  # current client index assigned to this worker

    # mode 0: the client receieve both weight and mask to clients;  config the new weight and mask, train; send the weight
    # mode 1: the client receieve weights;  config the new weight, train;  send the weight,
    # mode 2: the client receieve weights;  config the new weight; train, compute the gradients; send the weight and gradients
    # TODO (special mode, only for \delta t == 1) mode 3: receieve both weight and mask to clients; config the new weight and mask, train, compute the gradients; send the weight and gradients. 

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        """
        Add to self.message_handler_dict,
            and call the corresponding callback function when receiving a specific MSG_TYPE
        Pass the msg content to the callback function
        """
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.current_client_index = client_index

        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE_CODE)
        self.round_idx =  msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)

        if self.args.is_mobile == 1: # true
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.current_client_index = client_index

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
        if self.round_idx == self.num_rounds - 1:
            # post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, gradient_squared=None):
        """
        Call handle_message_receive_model_from_client in ServerManager
        """
        # sender_id is the id of worker  not  equal to current_client_id
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, self.current_client_index)
        if gradient_squared:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADIENT_SQUARED, gradient_squared)
        # message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADIENT, gradient)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)


    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, gradient_squared, local_sample_num = self.trainer.train(mode = self.mode, round_idx=self.round_idx)

        if self.mode in [2,3]:
            self.send_model_to_server(0, weights, local_sample_num, gradient_squared)
        else:
            self.send_model_to_server(0, weights, local_sample_num)
        """
        if self.mode in [2, 3]:
            gradient_squared_divided = {name: value / self.train_rounds for name, value in self.acc_gradient_squared.items()}

            self.send_model_to_server(0, weights, local_sample_num, gradient_squared_divided)
            self.acc_gradient_squared = None
            self.train_rounds = 0
        else:
            self.send_model_to_server(0, weights, local_sample_num)
        """