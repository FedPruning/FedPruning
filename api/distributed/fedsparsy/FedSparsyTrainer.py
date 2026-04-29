import logging
from .utils import transform_tensor_to_list
import torch


class FedSparsyTrainer(object):
    def _is_sparse_round(self, round_idx):
        dense_start_rounds = max(int(getattr(self.args, "dense_start_rounds", 1)), 1)
        return round_idx is not None and round_idx >= dense_start_rounds

    def compute_model_update(self, current_params, initial_params):
        update_dict = {}
        for key in current_params.keys():
            if key in initial_params:
                update_dict[key] = current_params[key] - initial_params[key]
        return update_dict

    def apply_topk_pruning(self, update_dict, target_sparsity):
        pruned_update = {}
        float_tensors = []

        for key, update in update_dict.items():
            if isinstance(update, torch.Tensor) and torch.is_floating_point(update):
                float_tensors.append((key, update))
            else:
                pruned_update[key] = update

        if not float_tensors:
            return pruned_update

        all_scores = torch.cat([tensor.abs().reshape(-1) for _, tensor in float_tensors])
        if target_sparsity <= 0:
            threshold = None
        else:
            keep_num = int(all_scores.numel() * (1 - target_sparsity))
            keep_num = max(min(keep_num, all_scores.numel()), 1)
            threshold = torch.topk(all_scores, keep_num, largest=True).values.min()

        for key, update in float_tensors:
            if threshold is None:
                pruned_update[key] = update
            else:
                mask = (update.abs() >= threshold).float()
                pruned_update[key] = update * mask

        return pruned_update

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

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self, round_idx=None):
        sparse_enabled = self._is_sparse_round(round_idx)
        reparam_per_step = getattr(self.args, "reparam_per_step", False)

        if not sparse_enabled:
            logging.info("Round %s is forced dense-start: activation dense + full update upload", round_idx)
        elif not reparam_per_step:
            logging.info("Sparse round %s uses original parameters for forward (reparam_per_step disabled)", round_idx)

        initial_params = self.trainer.get_model_params()
        masks = self.trainer.train(self.train_local, self.device, self.args, round_idx=round_idx)
        current_params = self.trainer.get_model_params()

        model_update = self.compute_model_update(current_params, initial_params)
        target_sparsity = self.args.target_sparsity if hasattr(self.args, "target_sparsity") else 0.9

        # The paper keeps the first round dense (no Top-K); sparse updates start from round 2
        effective_sparsity = target_sparsity if sparse_enabled else 0.0
        sparse_update = self.apply_topk_pruning(model_update, effective_sparsity)

        if self.args.is_mobile == 1:
            sparse_update = transform_tensor_to_list(sparse_update)

        return sparse_update, masks, self.local_sample_number
