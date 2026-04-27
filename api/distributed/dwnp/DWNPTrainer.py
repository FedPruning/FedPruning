from .utils import transform_tensor_to_list


class DWNPTrainer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args
        self._last_emb_params = None

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_hypernet(self, hn_params):
        self.trainer.set_hypernet_params(hn_params)

    def get_hypernet_params(self):
        return self.trainer.get_hypernet_params()

    def update_dynamic_emb(self, emb_params):
        self.trainer.set_dynamic_emb_params(emb_params)

    def get_dynamic_emb_params(self):
        return self.trainer.get_dynamic_emb_params()

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self, round_idx=None, train_hn=False):
        self.args.round_idx = round_idx
        server_density, device_density = self.trainer.prepare_joint_mask(
            self.client_index,
            round_idx,
            self.device,
            self.args,
        )
        self.trainer.train(self.train_local, self.device, self.args)

        if train_hn:
            self.trainer.train_hypernetwork(self.train_local, self.client_index, self.device, self.args)

        weights = self.trainer.get_model_params()
        hn_params = self.trainer.get_hypernet_params()
        emb_params = self.trainer.get_dynamic_emb_params()
        self._last_emb_params = emb_params

        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
            hn_params = transform_tensor_to_list(hn_params)
            emb_params = transform_tensor_to_list(emb_params)

        return weights, hn_params, emb_params, self.local_sample_number, server_density, device_density
