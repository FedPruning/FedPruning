import logging
import os
import sys

from .message_define import MyMessage
from .utils import transform_tensor_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
try:
    from core.distributed.communication.message import Message
    from core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedPruning.core.distributed.communication.message import Message
    from FedPruning.core.distributed.server.server_manager import ServerManager


class FedSparsyServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def _sample_clients_for_current_round(self):
        if self.is_preprocessed and self.preprocessed_client_lists is not None:
            client_indexes = self.preprocessed_client_lists[self.round_idx]
        else:
            client_indexes = self.aggregator.client_sampling(
                self.round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

        client_indexes = list(client_indexes)
        available_worker_num = self.size - 1
        if len(client_indexes) < available_worker_num:
            raise ValueError(
                f"Sampled {len(client_indexes)} clients, but {available_worker_num} client processes are running."
            )

        assigned_client_indexes = client_indexes[:available_worker_num]
        self.aggregator.set_active_clients(list(range(available_worker_num)), assigned_client_indexes)
        return assigned_client_indexes

    def send_init_msg(self):
        client_indexes = self._sample_clients_for_current_round()
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)

        for process_id, client_index in zip(range(1, self.size), client_indexes):
            self.send_message_init_config(process_id, global_model_params, client_index, self.round_idx)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def _broadcast_finish_to_clients(self):
        for receiver_id in range(1, self.size):
            message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receiver_id)
            self.send_message(message)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_update = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, model_update, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = %s", b_all_received)

        if b_all_received:
            global_update = self.aggregator.aggregate()
            global_model_params = self.aggregator.apply_server_optimizer(
                global_update,
                getattr(self.args, "server_lr", 1.0),
            )

            self.aggregator.test_on_server_for_all_clients(self.round_idx)
            self.round_idx += 1

            if self.round_idx >= self.round_num:
                logging.info("=======Training finished, notifying clients to shut down=======")
                self._broadcast_finish_to_clients()
                import time
                time.sleep(0.5)  # Give the MPI buffer time to flush messages
                self.com_manager.stop_receive_message()
                self.finish()
                return

            client_indexes = self._sample_clients_for_current_round()
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)

            for receiver_id, client_index in zip(range(1, self.size), client_indexes):
                self.send_message_sync_model_to_client(
                    receiver_id,
                    global_model_params,
                    client_index,
                    self.round_idx,
                )

    def send_message_init_config(self, receive_id, global_model_params, client_index, round_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, round_idx):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        self.send_message(message)
