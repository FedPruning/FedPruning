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


class DWNPServerManager(ServerManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0

        self.hn_update_steps = 0
        self.final_prune_applied = False

        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def _flags_for_round(self, round_idx):
        if getattr(self.args, "dwnp_enable_final_prune", 1) and round_idx == self.args.comm_round:
            return True, False, False

        sync_w = (round_idx % self.args.r_w == 0)
        train_hn = (round_idx % self.args.r_hn == 0)
        if train_hn:
            next_hn_step = self.hn_update_steps + 1
            sync_hn = (next_hn_step % self.args.r_theta == 0)
        else:
            sync_hn = False
        return sync_w, train_hn, sync_hn

    def send_init_msg(self):
        client_indexes = self.aggregator.client_sampling(
            self.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )

        global_model_params = self.aggregator.get_global_model_params()
        global_hn_params = self.aggregator.get_global_hn_params()
        global_emb_params = self.aggregator.get_global_emb_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
            global_hn_params = transform_tensor_to_list(global_hn_params)
            global_emb_params = transform_tensor_to_list(global_emb_params)

        sync_w, train_hn, sync_hn = self._flags_for_round(self.round_idx)

        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id,
                global_model_params,
                global_hn_params,
                global_emb_params,
                client_indexes[process_id - 1],
                self.round_idx,
                sync_w,
                train_hn,
                sync_hn,
            )

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_RESULT_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        client_index = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX))

        hn_params = msg_params.get(MyMessage.MSG_ARG_KEY_HN_PARAMS)
        emb_params = msg_params.get(MyMessage.MSG_ARG_KEY_EMB_PARAMS)

        self.aggregator.add_local_trained_result(
            sender_id - 1,
            client_index,
            model_params,
            local_sample_number,
            hn_params,
            emb_params,
        )

        if not self.aggregator.check_whether_all_receive():
            return

        sync_w, train_hn, sync_hn = self._flags_for_round(self.round_idx)

        if sync_w:
            global_model_params = self.aggregator.aggregate_models()
        else:
            self.aggregator.model_dict = dict()
            global_model_params = self.aggregator.get_global_model_params()

        if train_hn:
            self.hn_update_steps += 1
            if sync_hn:
                global_hn_params = self.aggregator.aggregate_hypernet()
                global_emb_params = self.aggregator.aggregate_dynamic_emb()
            else:
                self.aggregator.hn_params_dict = dict()
                self.aggregator.emb_params_dict = dict()
                global_hn_params = self.aggregator.get_global_hn_params()
                global_emb_params = self.aggregator.get_global_emb_params()
        else:
            self.aggregator.hn_params_dict = dict()
            self.aggregator.emb_params_dict = dict()
            global_hn_params = self.aggregator.get_global_hn_params()
            global_emb_params = self.aggregator.get_global_emb_params()

        self.aggregator.test_on_server_for_all_clients(self.round_idx)

        self.round_idx += 1
        if self.round_idx == self.round_num + 1:
            self.finish()
            return

        if getattr(self.args, "dwnp_enable_final_prune", 1) and self.round_idx == self.args.comm_round and not self.final_prune_applied:
            global_model_params = self.aggregator.prepare_final_pruned_model(self.round_idx)
            self.final_prune_applied = True

        if self.is_preprocessed:
            if self.preprocessed_client_lists is None:
                client_indexes = [self.round_idx] * self.args.client_num_per_round
            else:
                client_indexes = self.preprocessed_client_lists[self.round_idx]
        else:
            client_indexes = self.aggregator.client_sampling(
                self.round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

        sync_w, train_hn, sync_hn = self._flags_for_round(self.round_idx)

        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
            global_hn_params = transform_tensor_to_list(global_hn_params)
            global_emb_params = transform_tensor_to_list(global_emb_params)

        for receiver_id in range(1, self.size):
            self.send_message_sync_model_to_client(
                receiver_id,
                global_model_params,
                global_hn_params,
                global_emb_params,
                client_indexes[receiver_id - 1],
                self.round_idx,
                sync_w,
                train_hn,
                sync_hn,
            )

    def send_message_init_config(
        self,
        receive_id,
        global_model_params,
        global_hn_params,
        global_emb_params,
        client_index,
        round_idx,
        sync_w,
        train_hn,
        sync_hn,
    ):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_HN_PARAMS, global_hn_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EMB_PARAMS, global_emb_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_SYNC_W, sync_w)
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_HN, train_hn)
        message.add_params(MyMessage.MSG_ARG_KEY_SYNC_HN, sync_hn)
        self.send_message(message)

    def send_message_sync_model_to_client(
        self,
        receive_id,
        global_model_params,
        global_hn_params,
        global_emb_params,
        client_index,
        round_idx,
        sync_w,
        train_hn,
        sync_hn,
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d", receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_HN_PARAMS, global_hn_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EMB_PARAMS, global_emb_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_SYNC_W, sync_w)
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_HN, train_hn)
        message.add_params(MyMessage.MSG_ARG_KEY_SYNC_HN, sync_hn)
        self.send_message(message)
