import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor
from api.pruning.init_scheme import f_decay

class FedRTSAggregator(object):

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
        self.model_dict = dict()
        self.gradients_idx_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.alphas = None
        self.betas = None

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_trained_gradient(self, index, gradients_idx):
        logging.info("add_gradient. index = %d" % index)
        self.gradients_idx_dict[index] = gradients_idx
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self, t, T_end, alpha):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        need_initialize = False
        if self.alphas is None or self.betas is None:
            self.alphas = dict()
            self.betas = dict()
            need_initialize = True

        torch.manual_seed(t)
        gamma = self.args.aggregated_gamma
        ratio = self.args.initial_distribution_ratio
        mask_dict = self.trainer.model.mask_dict
        # global_model_dict = self.trainer.model.model.named_parameters()
        global_model_dict = self.get_global_model_params()
        for name, param in global_model_dict.items():
            if name in mask_dict:
                K = (mask_dict[name] == 1).int().sum().item()  # active_num
                k = int(f_decay(t, alpha, T_end) * K)  # pruned links, K - kappa
                kappa = K - k  # core links
                model_name = f'model.{name}' # layer name

                weight_outcomes = torch.zeros_like(self.model_dict[0][model_name]).cpu()
                active_indices = (mask_dict[name].view(-1) == 1).nonzero(as_tuple=False).view(-1).cpu()
                for idx in range(self.worker_num):
                    local_dict = self.model_dict[idx]
                    client_w = self.sample_num_dict[idx] / training_num  # p_k
                    if self.args.is_mobile == 1:
                        local_dict = transform_list_to_tensor(local_dict)

                    # select indices corresponding to h(i, x, kappa) = 1
                    _, local_largest_indices = torch.topk(torch.abs(local_dict[model_name].view(-1)[active_indices]), kappa, largest=True)
                    # update a individual outcomes for client
                    local_weight_outcomes = torch.zeros_like(weight_outcomes).cpu()
                    local_weight_outcomes.view(-1)[active_indices[local_largest_indices.cpu()]] = 1.0

                    weight_outcomes += local_weight_outcomes * client_w
                # update global voting probabilities
                _, global_largest_indices = torch.topk(torch.abs(param.view(-1)[active_indices]), kappa, largest=True)
                global_weight_outcomes = torch.zeros_like(weight_outcomes).cpu()
                global_weight_outcomes.view(-1)[active_indices[global_largest_indices.cpu()]] = 1.0
                weight_outcomes = (1 - gamma) * global_weight_outcomes + gamma * weight_outcomes

                semi_outcomes = weight_outcomes.view(-1)[active_indices]

                # Initialize beta distribution of thompson sampling.
                if need_initialize:
                    self.alphas[name] = torch.ones_like(self.model_dict[0][model_name])
                    self.betas[name] = torch.ones_like(self.model_dict[0][model_name])

                # update alphas
                self.alphas[name].view(-1)[active_indices] += ratio * semi_outcomes
                # update betas
                self.betas[name].view(-1)[active_indices] += ratio * (1 - semi_outcomes)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if  num_samples != -1:
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def ts_adj(self, t, T_end, alpha):
        # set independent random seed
        torch.manual_seed(t)
        gamma = self.args.aggregated_gamma
        ratio = self.args.initial_distribution_ratio
        mask_dict = self.trainer.model.mask_dict
        training_num = 0
        for idx in range(self.worker_num):
            training_num += self.sample_num_dict[idx]

        # global_model_dict = self.trainer.model.model.named_parameters()
        global_model_dict = self.get_global_model_params()
        for name, param in global_model_dict.items():
            if name in mask_dict:
                K = (mask_dict[name] == 1).int().sum().item()  # active_nums
                k = int(f_decay(t, alpha, T_end) * K)  # K - kappa
                # add protection
                inactive_num = int((mask_dict[name] == 0).int().sum().item())
                k = min(k, inactive_num)

                gradient_outcomes = torch.zeros_like(mask_dict[name]).cpu()
                inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1).cpu()
                for idx in range(self.worker_num):
                    gradients_idx = self.gradients_idx_dict[idx]
                    client_w = self.sample_num_dict[idx] / training_num  # p_k

                    # update a individual outcomes for client
                    local_gradient_outcomes = torch.zeros_like(gradient_outcomes).cpu()
                    local_gradient_outcomes.view(-1)[inactive_indices[gradients_idx[name]]] = 1
                    # voting based on sample num(importance related to training data size)
                    gradient_outcomes += local_gradient_outcomes * client_w
                gradient_outcomes = (1 - gamma) * 0.5 + gamma * gradient_outcomes
                semi_outcomes = gradient_outcomes.view(-1)[inactive_indices]

                # update alphas
                self.alphas[name].view(-1)[inactive_indices] += ratio * semi_outcomes
                # update betas
                self.betas[name].view(-1)[inactive_indices] += ratio * (1 - semi_outcomes)

                # sample based on beta distribution
                samples = torch.distributions.Beta(self.alphas[name].view(-1),
                                                   self.betas[name].view(-1)).sample()
                _, remaining_indices = torch.topk(samples, K, largest=True) # check number

                mask_dict[name].view(-1)[:] = 0
                mask_dict[name].view(-1)[remaining_indices] = 1

        self.trainer.model.mask_dict = mask_dict
        # self.alphas = None
        # self.betas = None

    def test_on_server_for_all_clients(self, round_idx):
        # if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
        #     return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx >= self.args.comm_round - 10:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))

            # last seven testing should be tested with full testing dataset
            if round_idx >= self.args.comm_round - 10 or self.args.num_eval == -1 :
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
            
            for key in metrics:
                if key != "test_total":
                    wandb.log({f"Test/{key}": metrics[key], "round": round_idx})
            logging.info(metrics)