import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor

class FedTinyAggregator(object):

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
        self.BN_params_dict = dict()
        self.BN_loss_dict = dict()
        self.global_BN_params = list()
        self.flag_client_model_uploaded_dict = dict()
        self.flag_client_BN_params_uploaded_dict = dict()
        self.flag_client_BN_loss_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
            self.flag_client_BN_params_uploaded_dict[idx] = False
            self.flag_client_BN_loss_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)
    def set_global_BN_params(self, BN_parameters):
        self.global_BN_params = BN_parameters

    def add_local_trained_BN_params(self, index, bn_params_list, sample_num):
        logging.info("add_BN_params. index = %d" % index)
        self.BN_params_dict[index] = bn_params_list
        self.sample_num_dict[index] = sample_num
        self.flag_client_BN_params_uploaded_dict[index] = True

    def add_local_trained_BN_loss(self, index, bn_loss_list):
        logging.info("add_BN_loss. index = %d" % index)
        self.BN_loss_dict[index] = bn_loss_list
        self.flag_client_BN_loss_uploaded_dict[index] = True

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
    def check_whether_BN_params_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_BN_params_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_BN_params_uploaded_dict[idx] = False
        return True
    
    def check_whether_BN_loss_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_BN_loss_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_BN_loss_uploaded_dict[idx] = False
        return True
    def aggregate_BN_params(self):
        start_time = time.time()
        BN_params_list = []
        training_num = 0

        for idx in range(self.worker_num):
            # if self.args.is_mobile == 1:
            #     self.BN_params_dict[idx] = transform_list_to_tensor(self.BN_params_dict[idx])
            BN_params_list.append((self.sample_num_dict[idx], self.BN_params_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.BN_params_dict[idx] = " + str(len(self.BN_params_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = BN_params_list[0]
        print(f"averaged_params is {averaged_params}")
        for c in range(0, len(averaged_params)):
            for k in averaged_params[c].keys():
                for i in range(0, len(BN_params_list)):
                    local_sample_number, local_BN_params = BN_params_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[c][k] = local_BN_params[c][k] * w
                    else:
                        averaged_params[c][k] += local_BN_params[c][k] * w


        # update the global model which is cached at the server side
        self.set_global_BN_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate BN params time cost: %d" % (end_time - start_time))
        return averaged_params
    def aggregate_BN_loss(self):
        BN_loss_list = []
        #for c in 
        for idx in range(self.worker_num):
            BN_loss_list.append((self.BN_loss_dict[idx]))
        averaged_loss = BN_loss_list[0]
        for c in range(len(averaged_loss)):
            for i in range(0, len(BN_loss_list)):
                if i == 0:
                    averaged_loss[c] = BN_loss_list[i][c]
                else:
                    averaged_loss[c] += BN_loss_list[i][c]
        Averaged_loss = [i/len(BN_loss_list) for i in averaged_loss]
        return Averaged_loss

    def aggregate(self):
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

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            # last seven testing should be tested with full testing dataset
            if round_idx >= self.args.comm_round - 10 or self.args.num_eval == -1 :
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
