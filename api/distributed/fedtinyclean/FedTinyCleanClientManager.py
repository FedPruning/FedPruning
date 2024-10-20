import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from core.distributed.client.client_manager import ClientManager
    from core.distributed.communication.message import Message
except ImportError:
    from FedPruning.core.distributed.client.client_manager import ClientManager
    from FedPruning.core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process


class FedTinyCleanClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.mode = 0

    # mode 0: the client receieve both weight and mask to clients;  config the new weight and mask, train; send the weight
    # mode 1: the client receieve weights;  config the new weight, train;  send the weight,
    # mode 2: the client receieve weights;  config the new weight; train, compute the gradients; send the weight and gradients
    # TODO (special mode, only for \delta t == 1) mode 3: receieve both weight and mask to clients; config the new weight and mask, train, compute the gradients; send the weight and gradients. 

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

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
        if self.round_idx == self.num_rounds - 1:
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
        weights, gradient, local_sample_num = self.trainer.train(self.round_idx, self.mode)
        if self.mode in [2, 3]:
            # logging.info("########## the client send gradients is ##########")
            # logging.info(gradient)
            self.send_model_to_server(0, weights, local_sample_num, gradient)
        else:
            self.send_model_to_server(0, weights, local_sample_num)