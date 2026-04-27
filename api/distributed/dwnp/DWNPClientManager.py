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
from .utils import transform_list_to_tensor


class DWNPClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.current_client_index = -1

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
            self.handle_message_init,
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        self._handle_server_message(msg_params, init=True)

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server")
        self._handle_server_message(msg_params, init=False)

    def _handle_server_message(self, msg_params, init=False):
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        hn_params = msg_params.get(MyMessage.MSG_ARG_KEY_HN_PARAMS)
        emb_params = msg_params.get(MyMessage.MSG_ARG_KEY_EMB_PARAMS)

        client_index = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX))
        round_idx = int(msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX))
        sync_w = bool(msg_params.get(MyMessage.MSG_ARG_KEY_SYNC_W))
        train_hn = bool(msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_HN))
        sync_hn = bool(msg_params.get(MyMessage.MSG_ARG_KEY_SYNC_HN))

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)
            hn_params = transform_list_to_tensor(hn_params)
            emb_params = transform_list_to_tensor(emb_params)

        client_changed = client_index != self.current_client_index
        self.current_client_index = client_index
        self.round_idx = round_idx

        if init or sync_w or client_changed:
            self.trainer.update_model(model_params)

        if init or sync_hn:
            self.trainer.update_hypernet(hn_params)
            self.trainer.update_dynamic_emb(emb_params)

        self.trainer.update_dataset(client_index)
        self.__train(train_hn)

        if self.round_idx == self.num_rounds:
            self.finish()

    def send_result_to_server(
        self,
        receive_id,
        weights,
        local_sample_num,
        hn_params,
        emb_params,
        server_density,
        device_density,
    ):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_RESULT_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, self.current_client_index)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_SERVER_DENSITY, server_density)
        message.add_params(MyMessage.MSG_ARG_KEY_DEVICE_DENSITY, device_density)
        message.add_params(MyMessage.MSG_ARG_KEY_HN_PARAMS, hn_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EMB_PARAMS, emb_params)
        self.send_message(message)

    def __train(self, train_hn):
        logging.info("#######training########### round_id = %d", self.round_idx)
        weights, hn_params, emb_params, local_sample_num, server_density, device_density = self.trainer.train(
            round_idx=self.round_idx,
            train_hn=train_hn,
        )
        self.send_result_to_server(
            0,
            weights,
            local_sample_num,
            hn_params,
            emb_params,
            server_density,
            device_density,
        )
