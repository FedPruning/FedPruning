import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor

class PruneFLAggregator(object):
    """
    Responsible for aggregating model parameters and gradients from different clients
    """
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

        self.worker_num = worker_num  # total number of client, not per round client
        self.device = device

        """
        key is client index
        """
        self.model_dict = dict()  # Store the model uploaded by each client
        # self.acc_gradient_squared_dict = dict()  # Store the accumulated squared gradient uploaded by each client
        # self.train_rounds_dict = dict()
        self.gradient_squared_dict = dict()
        self.sample_num_dict = dict()  # Store the number of train data for each client, do not need to reset
        """
        key is worker index
        """
        self.flag_client_model_uploaded_dict = dict()  # Whether the model of a client has been updated
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, worker_index, client_index, model_params, sample_num):
        logging.info(f"add_model. worker index={worker_index} client_index = {client_index}")
        self.model_dict[client_index] = model_params
        self.sample_num_dict[client_index] = sample_num
        self.flag_client_model_uploaded_dict[worker_index] = True

    def add_local_trained_gradient_squared(self, worker_index, client_index, gradient_squared):
        logging.info(f"add_gradient_squared. worker index={worker_index} client_index = {client_index}")
        # self.gradient_squared_dict[index] = gradient_squared

        self.gradient_squared_dict[client_index] = gradient_squared


    def check_whether_all_receive(self):
        """
        If all are received, set flag_client_model_uploaded_dict back to False
        """
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        #for idx in range(self.worker_num):
        for idx in self.model_dict.keys():
            if self.args.is_mobile == 1:  # true
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

        self.model_dict = dict() # reset
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def aggregate_gradient_squared(self):
        start_time = time.time()
        gradient_squared_list = []
        training_num = 0

        for idx in self.gradient_squared_dict.keys():
            gradient_squared_list.append((self.sample_num_dict[idx], self.gradient_squared_dict[idx]))
            training_num += self.sample_num_dict[idx]

        # logging.info("################aggregate: %d" % len(gradient_squared_list))
        (num0, averaged_grad_squared) = gradient_squared_list[0]
        # logging.info(averaged_grad.keys())
        for k in averaged_grad_squared.keys():
            for i in range(0, len(gradient_squared_list)):
                local_sample_number, local_grad_squared = gradient_squared_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_grad_squared[k] = local_grad_squared[k].to(self.device) * w
                else:
                    averaged_grad_squared[k] += local_grad_squared[k].to(self.device) * w

        self.gradient_squared_dict = dict()  # reset
        # self.train_rounds_dict = dict()
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_grad_squared

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Select the client to participate in the training according to the current round.
        If not all of them participate, select them randomly.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        """
        A validation set is randomly generated from the test dataset to evaluate the model performance.
        """
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

            # last five testing should be tested with full testing dataset
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
