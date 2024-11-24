import logging
from datasets import load_dataset
import random 
import numpy as np
from torch.utils.data import DataLoader

#logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def partition_data(train_data, test_data, partition, num_clients, alpha, train_ratio = 0.1, test_ratio = 0.1):
    total_num_train_data, total_num_test_data = len(train_data), len(test_data)
    select_num_train_data, select_num_test_data = int(total_num_train_data * train_ratio), int(total_num_test_data * test_ratio)
    subset_train_data = train_data.select(random.sample(range(total_num_train_data), select_num_train_data))
    subset_test_data = test_data.select(random.sample(range(total_num_test_data), select_num_test_data))
    
    if partition == "homo":
        idxs = np.random.permutation(select_num_train_data)
        batch_idxs = np.array_split(idxs, num_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

    elif partition == "hetero":
        idxs = np.random.permutation(select_num_train_data)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) * select_num_train_data
        batch_idxs = []
        start_id = 0
        for i, p in enumerate(proportions):
            batch_idxs.append(idxs[start_id : start_id + int(p)])
            start_id = start_id + int(p)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}
    else:
        raise Exception(f"there is no partition method named {partition}")
    
    return subset_train_data, subset_test_data, net_dataidx_map

def load_partition_data_tinystories(partition_method, partition_alpha, client_number, batch_size, ratio = 0.01):
    data = load_dataset("roneneldan/TinyStories")
    train_data, test_data = data["train"], data["validation"]
    train_data, test_data, net_dataidx_map = partition_data(train_data, test_data, partition_method, client_number, partition_alpha, train_ratio = ratio, test_ratio = ratio)

    train_data_global = get_dataloader_tinystories(train_data, batch_size=batch_size)
    test_data_global = get_dataloader_tinystories(test_data,  batch_size=batch_size)
    train_data_num = len(train_data_global)
    test_data_num = len(test_data_global)

    logging.info("train_dl_global number = " + str(train_data_num))
    logging.info("test_dl_global number = " + str(test_data_num))
    
     # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        
        train_data_local = get_dataloader_tinystories(train_data,  batch_size=batch_size, dataidxs=dataidxs)
        test_data_local =  get_dataloader_tinystories(test_data,  batch_size=batch_size)
        
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict, None

def get_dataloader_tinystories(dataset, batch_size, dataidxs=None):
    if dataidxs is not None:
        return DataLoader(dataset.select(dataidxs), batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        