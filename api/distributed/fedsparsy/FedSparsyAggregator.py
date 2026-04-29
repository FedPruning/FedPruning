import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor


class FedSparsyAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
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
        self.model_dict = {}
        self.mask_dict = {}
        self.sample_num_dict = {}
        self.flag_client_model_uploaded_dict = {idx: False for idx in range(self.worker_num)}
        self.active_worker_ids = list(range(self.worker_num))
        self.active_client_indexes = list(range(self.worker_num))

        self.server_momentum_buffer = {}
        self.server_adam_m = {}
        self.server_adam_v = {}
        self.server_optimizer_step = 0

    def set_active_clients(self, worker_ids, client_indexes):
        self.active_worker_ids = list(worker_ids)
        self.active_client_indexes = list(client_indexes)
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def _is_float_tensor(self, value):
        return isinstance(value, torch.Tensor) and torch.is_floating_point(value)

    def _check_finite_tensor_dict(self, tensor_dict, dict_name):
        for key, value in tensor_dict.items():
            if not self._is_float_tensor(value):
                continue
            if torch.isfinite(value).all():
                continue
            raise FloatingPointError(f"Non-finite tensor found in {dict_name}: {key}")

    def _sanitize_tensor_dict(self, tensor_dict, dict_name):
        sanitized = {}
        for key, value in tensor_dict.items():
            if self._is_float_tensor(value):
                if not torch.isfinite(value).all():
                    logging.warning("Non-finite tensor found in %s: %s, replacing nan/inf with 0", dict_name, key)
                    sanitized[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized

    def _tensor_dict_global_norm(self, tensor_dict):
        total = 0.0
        for value in tensor_dict.values():
            if not self._is_float_tensor(value):
                continue
            total += torch.sum(value.float() * value.float()).item()
        return total ** 0.5

    def _clip_tensor_dict_by_global_norm(self, tensor_dict, max_norm):
        if max_norm is None or max_norm <= 0:
            return tensor_dict

        total_norm = self._tensor_dict_global_norm(tensor_dict)
        if total_norm == 0 or total_norm <= max_norm:
            return tensor_dict

        clip_coef = max_norm / (total_norm + 1e-12)
        logging.warning(
            "Clip aggregated update by global norm: original_norm=%.6f, max_norm=%.6f, clip_coef=%.6f",
            total_norm,
            max_norm,
            clip_coef,
        )

        clipped_tensor_dict = {}
        for key, value in tensor_dict.items():
            if self._is_float_tensor(value):
                clipped_tensor_dict[key] = value * clip_coef
            else:
                clipped_tensor_dict[key] = value
        return clipped_tensor_dict

    def apply_server_optimizer(self, global_update, learning_rate=1.0):
        global_params = self.get_global_model_params()
        optimizer_name = getattr(self.args, "server_optimizer", "sgd").lower()

        global_params = self._sanitize_tensor_dict(global_params, "global_params_before_update")
        global_update = self._sanitize_tensor_dict(global_update, "global_update_before_clipping")
        self._check_finite_tensor_dict(global_params, "global_params_before_update")
        self._check_finite_tensor_dict(global_update, "global_update_before_clipping")
        total_norm_before_clip = self._tensor_dict_global_norm(global_update)
        logging.info("server_update_norm_before_clip=%.6f", total_norm_before_clip)
        global_update = self._clip_tensor_dict_by_global_norm(
            global_update,
            getattr(self.args, "server_clip_norm", -1.0),
        )
        total_norm_after_clip = self._tensor_dict_global_norm(global_update)
        logging.info("server_update_norm_after_clip=%.6f", total_norm_after_clip)
        self._check_finite_tensor_dict(global_update, "global_update_after_clipping")

        if optimizer_name == "adam":
            self.server_optimizer_step += 1
            beta1 = getattr(self.args, "server_beta1", 0.9)
            beta2 = getattr(self.args, "server_beta2", 0.999)
            eps = getattr(self.args, "server_eps", 1e-8)

        momentum = getattr(self.args, "server_momentum", 0.0)

        for key in global_params.keys():
            if key not in global_update:
                continue
            if not self._is_float_tensor(global_params[key]) or not self._is_float_tensor(global_update[key]):
                continue

            update_tensor = global_update[key]
            if optimizer_name == "adam":
                if key not in self.server_adam_m:
                    self.server_adam_m[key] = torch.zeros_like(update_tensor)
                    self.server_adam_v[key] = torch.zeros_like(update_tensor)

                self.server_adam_m[key] = beta1 * self.server_adam_m[key] + (1 - beta1) * update_tensor
                self.server_adam_v[key] = beta2 * self.server_adam_v[key] + (1 - beta2) * update_tensor.pow(2)

                m_hat = self.server_adam_m[key] / (1 - beta1 ** self.server_optimizer_step)
                v_hat = self.server_adam_v[key] / (1 - beta2 ** self.server_optimizer_step)
                global_params[key] = global_params[key] + learning_rate * m_hat / (torch.sqrt(v_hat) + eps)
            else:
                if momentum > 0:
                    if key not in self.server_momentum_buffer:
                        self.server_momentum_buffer[key] = torch.zeros_like(update_tensor)
                    self.server_momentum_buffer[key] = momentum * self.server_momentum_buffer[key] + update_tensor
                    update_tensor = self.server_momentum_buffer[key]

                global_params[key] = global_params[key] + learning_rate * update_tensor

        self._check_finite_tensor_dict(global_params, "global_params_after_update")
        self.set_global_model_params(global_params)
        return global_params

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_trained_mask(self, index, mask):
        logging.info("add_mask. index = %d" % index)
        self.mask_dict[index] = mask
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("active_worker_ids = %s", self.active_worker_ids)
        if not self.active_worker_ids:
            return False

        for idx in self.active_worker_ids:
            if not self.flag_client_model_uploaded_dict.get(idx, False):
                return False

        for idx in self.active_worker_ids:
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        update_list = []
        training_num = 0

        for idx in self.active_worker_ids:
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            update_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        if training_num == 0 or not update_list:
            raise ValueError("No client updates were collected for aggregation.")

        averaged_update = {}
        for key, value in update_list[0][1].items():
            if self._is_float_tensor(value):
                averaged_update[key] = torch.zeros_like(value)
            elif isinstance(value, torch.Tensor):
                averaged_update[key] = value.clone()
            else:
                averaged_update[key] = value

        for local_sample_number, local_update in update_list:
            local_update = self._sanitize_tensor_dict(local_update, "local_update")
            weight = local_sample_number / training_num
            for key in averaged_update.keys():
                if self._is_float_tensor(local_update[key]):
                    averaged_update[key] += local_update[key] * weight

        averaged_update = self._sanitize_tensor_dict(averaged_update, "aggregated_update")
        self._check_finite_tensor_dict(averaged_update, "aggregated_update")
        end_time = time.time()
        logging.info("aggregate time cost: %s", end_time - start_time)
        return averaged_update

    def aggregate_mask(self):
        aggr_mask = self.mask_dict[0]
        for idx in range(1, self.worker_num):
            for k, v in self.mask_dict[idx].items():
                aggr_mask[k] = torch.logical_or(aggr_mask[k].to(self.device), v.to(self.device)).float()
        return aggr_mask

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        client_indexes = list(client_indexes)
        logging.info("client_indexes = %s" % str(client_indexes))
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
        should_test = round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1
        if should_test:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))

            use_full_test = self.args.num_eval == -1 or (
                self.args.comm_round > 10 and round_idx >= self.args.comm_round - 10
            )
            eval_loader = self.test_global if use_full_test else self.val_global
            logging.info("Evaluation split = %s", "full_test" if use_full_test else "validation_subset")
            metrics = self.trainer.test(eval_loader, self.device, self.args)

            for key in metrics:
                if key != "test_total":
                    wandb.log({f"Test/{key}": metrics[key], "round": round_idx})
            logging.info(metrics)
