import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor
from api.pruning.init_scheme import f_decay

class FedAdaPruningAggregator(object):

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
        self.gradient_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.weight_alphas = None
        self.weight_betas = None

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_trained_gradient(self, index, gradient):
        logging.info("add_gradient. index = %d" % index)
        self.gradient_dict[index] = gradient
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
        if self.weight_alphas is None or self.weight_betas is None:
            self.weight_alphas = dict()
            self.weight_betas = dict()
            need_initialize = True

        torch.manual_seed(t)
        gamma = self.args.aggregated_gamma
        ratio = self.args.initial_distribution_ratio
        mask_dict = self.trainer.model.mask_dict
        global_model_dict = self.trainer.model.model.named_parameters()
        for name, param in global_model_dict:
            if name in mask_dict:
                active_num = (mask_dict[name] == 1).int().sum().item()
                k = int(f_decay(t, alpha, T_end) * active_num)
                model_name = f'model.{name}'
                
                weight_voting_dict = torch.zeros_like(self.model_dict[0][model_name]).cpu()
                active_indices = (mask_dict[name].view(-1) == 1).nonzero(as_tuple=False).view(-1).cpu()
                for idx in range(self.worker_num):
                    model_dict = self.model_dict[idx]
                    # p_k
                    client_w = self.sample_num_dict[idx] / training_num
                    if self.args.is_mobile == 1:
                        model_dict = transform_list_to_tensor(model_dict)
                    # select lowest_k weights' indices
                    # !!! NOTE: Need to update lowest in activate index
                    _, local_lowest_k_indices = torch.topk(torch.abs(model_dict[model_name].view(-1)[active_indices]), k, largest=False)
                    # update a tmp voting dict and voting
                    local_weight_voting_dict = torch.zeros_like(weight_voting_dict).cpu()
                    local_weight_voting_dict.view(-1)[active_indices[local_lowest_k_indices.cpu()]] = ratio
                    # voting based on sample num(importance related to training data size)
                    weight_voting_dict += local_weight_voting_dict * client_w
                # update global voting probabilities
                _, global_lowest_k_indices = torch.topk(torch.abs(param.view(-1)[active_indices]), k, largest=False)
                global_weight_voting_dict = torch.zeros_like(weight_voting_dict).cpu()
                global_weight_voting_dict.view(-1)[active_indices[global_lowest_k_indices.cpu()]] = ratio
                weight_voting_dict = (1 - gamma) * weight_voting_dict + gamma * global_weight_voting_dict

                active_votes = weight_voting_dict.view(-1)[active_indices]
                avg_active_prob = ratio * k / float(active_votes.size()[0])

                # Initialize beta distribution of thompson sampling on weights' history information.
                if need_initialize:
                    self.weight_alphas[name] = torch.full_like(self.model_dict[0][model_name], avg_active_prob)
                    self.weight_betas[name] = torch.full_like(self.model_dict[0][model_name], avg_active_prob)

                # update alphas
                self.weight_alphas[name].view(-1)[active_indices] += active_votes
                # update betas
                if self.args.ts_beta_update == 1:
                    self.weight_betas[name].view(-1)[active_indices] += (ratio - active_votes)
                    # self.weight_betas[name] = torch.clamp(self.weight_betas[name], min=1e-6) # add protection
                else:
                    active_zero_indices = (active_votes == 0) # find indices in voting which value equals to 0
                    active_avg_failed_prob = ratio * (float(active_votes.size()[0]) - k) / float(active_zero_indices.size()[0])
                    self.weight_betas[name].view(-1)[active_indices][active_zero_indices] += active_avg_failed_prob

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def aggregate_gradient(self):
        start_time = time.time()
        gradient_list = []
        training_num = 0

        for idx in range(self.worker_num):
            gradient_list.append((self.sample_num_dict[idx], self.gradient_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(gradient_list))
        (num0, averaged_grad) = gradient_list[0]
        # logging.info(averaged_grad.keys())
        for k in averaged_grad.keys():
            for i in range(0, len(gradient_list)):
                local_sample_number, local_grad = gradient_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_grad[k] = local_grad[k].to(self.device) * w
                else:
                    averaged_grad[k] += local_grad[k].to(self.device) * w

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_grad

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

    def ts_pruning_growing(self, gradients, t, T_end, alpha):
        # set independent random seed
        torch.manual_seed(t)
        gamma = self.args.aggregated_gamma
        ratio = self.args.initial_distribution_ratio
        mask_dict = self.trainer.model.mask_dict
        training_num = 0
        for idx in range(self.worker_num):
            training_num += self.sample_num_dict[idx]
        # ......绷不住了，model套model，没发现这个问题
        global_model_dict = self.trainer.model.model.named_parameters()

        for name, param in global_model_dict:
            if name in mask_dict:
                active_num = (mask_dict[name] == 1).int().sum().item()
                k = int(f_decay(t, alpha, T_end) * active_num)
                # add protection for pruned num
                inactive_num = int((mask_dict[name] == 0).int().sum().item())
                k = min(k, inactive_num)
                if k == 0:
                    continue

                # Pruning voting, based on weights
                # update clients' voting probabilities
                # !!!! NOTE: key in model_dict is not equals to key in mask_dict and gradient
                model_name = f'model.{name}'
                weight_voting_dict = torch.zeros_like(self.model_dict[0][model_name]).cpu()
                active_indices = (mask_dict[name].view(-1) == 1).nonzero(as_tuple=False).view(-1).cpu()
                for idx in range(self.worker_num):
                    model_dict = self.model_dict[idx]
                    # p_k
                    client_w = self.sample_num_dict[idx] / training_num
                    if self.args.is_mobile == 1:
                        model_dict = transform_list_to_tensor(model_dict)
                    # select lowest_k weights' indices
                    # !!! NOTE: Need to update lowest in activate index
                    _, local_lowest_k_indices = torch.topk(torch.abs(model_dict[model_name].view(-1)[active_indices]), k, largest=False)
                    # update a tmp voting dict and voting
                    local_weight_voting_dict = torch.zeros_like(weight_voting_dict).cpu()
                    local_weight_voting_dict.view(-1)[active_indices[local_lowest_k_indices.cpu()]] = ratio
                    # voting based on sample num(importance related to training data size)
                    weight_voting_dict += local_weight_voting_dict * client_w
                # update global voting probabilities
                _, global_lowest_k_indices = torch.topk(torch.abs(param.view(-1)[active_indices]), k, largest=False)
                global_weight_voting_dict = torch.zeros_like(weight_voting_dict).cpu()
                global_weight_voting_dict.view(-1)[active_indices[global_lowest_k_indices.cpu()]] = ratio
                weight_voting_dict = (1 - gamma) * weight_voting_dict + gamma * global_weight_voting_dict

                # Growing voting, based on gradients
                # update clients' voting probabilities
                # print("key: ", self.gradient_dict[0].keys())
                gradient_voting_dict = torch.zeros_like(self.gradient_dict[0][name]).cpu()
                inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1).cpu()
                for idx in range(self.worker_num):
                    gradient_dict = self.gradient_dict[idx]
                    # p_k
                    client_w = self.sample_num_dict[idx] / training_num
                    _, local_largest_k_indices = torch.topk(torch.abs(gradient_dict[name].view(-1)[inactive_indices]), k, largest=True)
                    # update a tmp voting dict and voting
                    local_gradient_voting_dict = torch.zeros_like(gradient_voting_dict).cpu()
                    local_gradient_voting_dict.view(-1)[inactive_indices[local_largest_k_indices.cpu()]] = 1
                    # voting based on sample num(importance related to training data size)
                    gradient_voting_dict += local_gradient_voting_dict * client_w
                # update global voting probabilities
                _, global_largest_k_indices = torch.topk(torch.abs(gradients[name].view(-1)[inactive_indices]), k, largest=True)
                global_gradient_voting_dict = torch.zeros_like(gradient_voting_dict).cpu()
                global_gradient_voting_dict.view(-1)[inactive_indices[global_largest_k_indices.cpu()]] = 1
                gradient_voting_dict = (1 - gamma) * gradient_voting_dict + gamma * global_gradient_voting_dict
                # pruning, update probabilities based on beta distribution
                active_indices = (mask_dict[name].view(-1) == 1).nonzero(as_tuple=False).view(-1).cpu()
                active_votes = weight_voting_dict.view(-1)[active_indices]

                # update alphas
                self.weight_alphas[name].view(-1)[active_indices] += active_votes
                # update betas
                if self.args.ts_beta_update == 1:
                    # avg_active_prob = ratio * k / float(active_votes.size()[0])
                    self.weight_betas[name].view(-1)[active_indices] += (ratio - active_votes)
                    # self.weight_betas[name] = torch.clamp(self.weight_betas[name], min=1e-6) # add protection
                else:
                    active_zero_indices = (active_votes == 0) # find indices which value equals to 0
                    active_avg_failed_prob = ratio * (float(active_votes.size()[0]) - k) / float(active_zero_indices.size()[0])
                    self.weight_betas[name].view(-1)[active_indices][active_zero_indices] += active_avg_failed_prob
                # sample based on beta distribution
                pruning_samples = torch.distributions.Beta(self.weight_alphas[name].view(-1)[active_indices], self.weight_betas[name].view(-1)[active_indices]).sample()
                _, prune_indices = torch.topk(pruning_samples, k, largest=True)
                # _, prune_indices = torch.topk(active_votes, k, largest=True) # 因为前面已经选出最低的k个并投票，因此仍然选prob最大的k个即可
                mask_dict[name].view(-1)[active_indices[prune_indices.cpu()]] = 0
                # growing
                inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1).cpu()
                inactive_votes = gradient_voting_dict.view(-1)[inactive_indices].cpu()
                # update alphas
                avg_inactive_prob = ratio * k / float(inactive_votes.size()[0])
                growing_alphas = inactive_votes + avg_inactive_prob
                # update betas
                growing_betas = torch.full_like(growing_alphas, avg_inactive_prob)
                inactive_zero_indices = (inactive_votes == 0) # find indices which value equals to 0
                inactive_avg_failed_prob = ratio * (float(inactive_votes.size()[0]) - k) / float(inactive_zero_indices.size()[0])
                growing_betas[inactive_zero_indices] += inactive_avg_failed_prob
                # sample based on beta distribution
                growing_samples = torch.distributions.Beta(growing_alphas, growing_betas).sample()
                _, grow_indices = torch.topk(growing_samples, k, largest=True)
                # _, grow_indices = torch.topk(inactive_votes, k, largest=True)
                mask_dict[name].view(-1)[inactive_indices[grow_indices.cpu()]] = 1
        self.trainer.model.mask_dict = mask_dict
        self.weight_alphas = None
        self.weight_betas = None

    def test_on_server_for_all_clients(self, round_idx):
        # if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
        #     return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx >= self.args.comm_round - 10:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            # train_num_samples = []
            # train_tot_corrects = []
            # train_losses = []
            # for client_idx in range(self.args.client_num_in_total):
            #     # train data
            #     metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
            #     train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
            #     train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            #     train_num_samples.append(copy.deepcopy(train_num_sample))
            #     train_losses.append(copy.deepcopy(train_loss))

            #     """
            #     Note: CI environment is CPU-based computing. 
            #     The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            #     """
            #     if self.args.ci == 1:
            #         break

            # test on training dataset
            # train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            # train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            # stats = {'training_acc': train_acc, 'training_loss': train_loss}
            # logging.info(stats)

            # last seven testing should be tested with full testing dataset
            if round_idx >= self.args.comm_round - 10 or self.args.num_eval == -1 :
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
            
            for key in metrics:
                if key != "test_total":
                    wandb.log({f"Test/{key}": metrics[key], "round": round_idx})
            logging.info(metrics)