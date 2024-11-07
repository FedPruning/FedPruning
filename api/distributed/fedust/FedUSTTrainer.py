from .utils import transform_tensor_to_list


class FedUSTTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        # dataloader to full forgotten_set
        self.forgotten_set_local_dict = {}
        for k, loader in train_data_local_dict.items():
            self.forgotten_set_local_dict[k] = []
            for batch_idx, (x, labels, index) in enumerate(loader):
                for i in range(x.size(0)):
                    self.forgotten_set_local_dict[k].append(index[i].item())
        self.forgotten_local = None

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.forgotten_local = self.forgotten_set_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        # self.test_local = self.test_data_local_dict[client_index] # useless

    def train(self, mode, round_idx = None, client_index = None):
        masks, forgotten_set_local = self.trainer.train(self.train_local, self.forgotten_local, self.device, self.args, mode, round_idx)
        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)

        return weights, masks, self.local_sample_number, forgotten_set_local

    # def test(self):
    #     # train data
    #     train_metrics = self.trainer.test(self.train_local, self.device, self.args)
    #     train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
    #                                                       train_metrics['test_total'], train_metrics['test_loss']

    #     # test data
    #     test_metrics = self.trainer.test(self.test_local, self.device, self.args)
    #     test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
    #                                                       test_metrics['test_total'], test_metrics['test_loss']

    #     return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample