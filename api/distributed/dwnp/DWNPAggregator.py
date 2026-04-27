import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor


class DWNPAggregator(object):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set(self.args.num_eval)
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device

        self.model_dict = dict()
        self.hn_params_dict = dict()
        self.emb_params_dict = dict()
        self.sample_num_dict = dict()

        self.flag_client_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def get_global_hn_params(self):
        return self.trainer.get_hypernet_params()

    def set_global_hn_params(self, hn_parameters):
        self.trainer.set_hypernet_params(hn_parameters)

    def get_global_emb_params(self):
        return self.trainer.get_dynamic_emb_params()

    def set_global_emb_params(self, emb_parameters):
        self.trainer.set_dynamic_emb_params(emb_parameters)

    def prepare_final_pruned_model(self, round_idx=None):
        server_mask = self.trainer.apply_server_pruning(self.device, round_idx=round_idx)
        self.set_global_model_params(self.trainer.get_model_params())
        return self.get_global_model_params()

    def add_local_trained_result(
        self,
        worker_index,
        client_index,
        model_params,
        sample_num,
        hn_params=None,
        emb_params=None,
    ):
        logging.info("add_model worker=%d client=%d", worker_index, client_index)
        self.model_dict[client_index] = model_params
        self.sample_num_dict[client_index] = sample_num
        if hn_params is not None:
            self.hn_params_dict[client_index] = hn_params
        if emb_params is not None:
            self.emb_params_dict[client_index] = emb_params
        self.flag_client_uploaded_dict[worker_index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_uploaded_dict[idx] = False
        return True

    def aggregate_models(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in self.model_dict.keys():
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        self.set_global_model_params(averaged_params)
        self.model_dict = dict()

        logging.info("aggregate model time cost: %.3f", time.time() - start_time)
        return averaged_params

    def aggregate_hypernet(self):
        if len(self.hn_params_dict) == 0:
            return self.get_global_hn_params()

        start_time = time.time()
        hn_list = []
        training_num = 0

        for idx in self.hn_params_dict.keys():
            local_hn = self.hn_params_dict[idx]
            if self.args.is_mobile == 1:
                local_hn = transform_list_to_tensor(local_hn)
            hn_list.append((self.sample_num_dict[idx], local_hn))
            training_num += self.sample_num_dict[idx]

        (_, averaged_hn) = hn_list[0]
        for k in averaged_hn.keys():
            for i in range(len(hn_list)):
                local_sample_number, local_hn = hn_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_hn[k] = local_hn[k] * w
                else:
                    averaged_hn[k] += local_hn[k] * w

        self.set_global_hn_params(averaged_hn)
        self.hn_params_dict = dict()

        logging.info("aggregate hypernet time cost: %.3f", time.time() - start_time)
        return averaged_hn

    def aggregate_dynamic_emb(self):
        if len(self.emb_params_dict) == 0:
            return self.get_global_emb_params()

        start_time = time.time()
        emb_list = []
        training_num = 0

        for idx in self.emb_params_dict.keys():
            local_emb = self.emb_params_dict[idx]
            if self.args.is_mobile == 1:
                local_emb = transform_list_to_tensor(local_emb)
            emb_list.append((self.sample_num_dict[idx], local_emb))
            training_num += self.sample_num_dict[idx]

        (_, averaged_emb) = emb_list[0]
        for k in averaged_emb.keys():
            for i in range(len(emb_list)):
                local_sample_number, local_emb = emb_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_emb[k] = local_emb[k] * w
                else:
                    averaged_emb[k] += local_emb[k] * w

        self.set_global_emb_params(averaged_emb)
        self.emb_params_dict = dict()

        logging.info("aggregate dynamic emb time cost: %.3f", time.time() - start_time)
        return averaged_emb

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s", str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if num_samples != -1:
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx >= self.args.comm_round - 10:
            logging.info("################test_on_server_for_all_clients : %s", str(round_idx))
            if round_idx >= self.args.comm_round - 10 or self.args.num_eval == -1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

            for key in metrics:
                if key != "test_total":
                    wandb.log({f"Test/{key}": metrics[key], "round": round_idx})
            logging.info(metrics)
