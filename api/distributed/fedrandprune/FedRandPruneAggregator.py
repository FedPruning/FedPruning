import copy
import logging
import random
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from core.trainer.model_trainer import ModelTrainer
import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor

class FedRandPruneAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer: ModelTrainer = model_trainer

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
        self.mask_dict:dict[int, dict[str, torch.Tensor]] = dict()
        self.gradient_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def batch_generate_mask_dict(self):
        for idx in range(self.worker_num):
            self.mask_dict[idx] = self.trainer.model.generate_mask_dict()[1]

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list: list[tuple[int, dict[str, torch.Tensor]]] = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]

        origin_model_params = self.trainer.get_model_params()

        for k in averaged_params.keys():
            mask_key = k.replace("model.", "")
            if len(self.mask_dict) == 0 or self.mask_dict[0].get(mask_key) is None: # aggregate without mask
                logging.debug(f"aggregate_without_mask {k}")
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w
            else:
                logging.debug("aggregate_with_mask {k}")
                origin_parameter = origin_model_params[k]
                weight_list = [model[1][k] for model in model_list]
                mask_list = [self.mask_dict[idx][mask_key] for idx in range(self.worker_num)]
                sample_num_list = [self.sample_num_dict[idx] for idx in range(self.worker_num)]
                averaged_params[k] = aggregate_given_parameter_matrix(origin_parameter, weight_list, sample_num_list, mask_list)

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

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
            if self.args.test_without_mask:
                _, mask_dict = self.trainer.model.generate_mask_dict(target_density_override=1.0)
            else:
                _, mask_dict = self.trainer.model.generate_mask_dict()
            self.trainer.model.mask_dict = mask_dict
            self.trainer.model.apply_mask()
            if round_idx >= self.args.comm_round - 10 or self.args.num_eval == -1 :
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
            
            for key in metrics:
                if key != "test_total":
                    wandb.log({f"Test/{key}": metrics[key], "round": round_idx})
            logging.info(metrics)

def aggregate_given_parameter_matrix( 
        origin_parameter:torch.Tensor, 
        weight_list: list[torch.Tensor], 
        sample_num_list: list[int], 
        mask_list: list[torch.Tensor]
    ) ->torch.Tensor:
        worker_num = len(weight_list)
        layer_shape = weight_list[0].shape
        logging.debug(f"layer_shape: {layer_shape}")
        weighted_mask_list = torch.zeros((worker_num, *layer_shape), 
                                   dtype=torch.float, 
                                   device=mask_list[0].device) # shape: (worker_num, layer_shape)
        for i in range(worker_num):
            weighted_mask_list[i] = mask_list[i] * sample_num_list[i]
        sum_weighted_mask = torch.sum(weighted_mask_list, dim=0) # shape: (layer_shape)

        # print(f"sum_weighted_mask: {sum_weighted_mask}")

        aggregated_params = torch.zeros(layer_shape,dtype=origin_parameter.dtype,device=origin_parameter.device)
        # set the origin parameter to the aggregated parameter if all clients mask the parameter.
        mask_zero_positions = (sum_weighted_mask == 0)
        aggregated_params[mask_zero_positions] = origin_parameter[mask_zero_positions]

        non_zero_positions = (sum_weighted_mask != 0)
        for i in range(worker_num):
            aggregated_params[non_zero_positions] += weight_list[i][non_zero_positions] * weighted_mask_list[i][non_zero_positions] / sum_weighted_mask[non_zero_positions]

        return aggregated_params

if __name__ == "__main__":
    print("Testing aggregate_given_layer function...")
    
    device = torch.device("cpu")
    
    origin_parameter = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], device=device)
    
    model_list = [
        torch.tensor([
            [100,100,100,100],
            [100,100,100,100],
            [100,100,100,100],
            [100,100,100,100]
        ], device=device),
        torch.tensor([
            [200,200,200,200],
            [200,200,200,200],
            [200,200,200,200],
            [200,200,200,200]
        ], device=device),
        torch.tensor([
            [300,300,300,300],
            [300,300,300,300],
            [300,300,300,300],
            [300,300,300,300]
        ], device=device)
    ]
    
    sample_num_list = [50, 100, 150] 
    
    mask_list = [
        torch.tensor([
            [0.0, 1.0, 1.0, 1.0],  
            [0.0, 0.0, 0.0, 1.0], 
            [0.0, 0.0, 0.0, 1.0],  
            [0.0, 0.0, 0.0, 0.0]
        ], device=device),
        torch.tensor([
            [0.0, 1.0, 1.0, 1.0],  
            [0.0, 1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0],  
            [0.0, 0.0, 0.0, 0.0]
        ], device=device),
        torch.tensor([
            [0.0, 0.0, 0.0, 1.0], 
            [0.0, 0.0, 0.0, 1.0], 
            [0.0, 1.0, 1.0, 1.0], 
            [0.0, 0.0, 0.0, 0.0]
        ], device=device)
    ]
    
    result = aggregate_given_parameter_matrix(origin_parameter, model_list, sample_num_list, mask_list)
    
    print("Origin parameter:")
    print(origin_parameter)
    print("\nClient model parameters:")
    for i, model in enumerate(model_list):
        print(f"Client {i}:")
        print(model)
    
    print("\nClient masks:")
    for i, mask in enumerate(mask_list):
        print(f"Client {i} (sample number: {sample_num_list[i]}):")
        print(mask)
    
    print("\nAggregated result:")
    print(result)