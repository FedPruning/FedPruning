import torch
from typing import Dict, List
from torch import nn
from api.pruning.init_scheme import generate_layer_density_dict, pruning, sparse_update_step, sparse_pruning_step, sparse_growing_step
import warnings
import logging
import re

from FedPruning.api.distributed.fedsgc.message_define import MyMessage
from FedPruning.api.distributed.fedsgc.utils import transform_list_to_tensor


class SparseModel(nn.Module):
    def __init__(self, model,
                 target_density:float=1.,
                #  strategy:str="uniform_magnitude",
                strategy:str="ERK_magnitude",
                 mask_dict: dict = {},
                 ignore_layers:list[int, str, type]=[".*bias.*", nn.BatchNorm2d, ".*bn.*", nn.LayerNorm, ".*ln.*"], 
                 device = None,
                 ):
        super(SparseModel, self).__init__()
        # strategy is a str that [sparsity_distribution]_[pruning_strategy]
        # e.g. uniform_magnitude

        self.model = model
        self.mask_dict = mask_dict
        self.strategy = strategy
        self.target_density = target_density
        self.ignore_layers = ignore_layers
        self.device = device


        # layer_set includes all layer names
        # layer_shape_dict includes the shape of every layer
        # num_overall_elements is the number of parameters in the whole model
        self.layer_set, self.layer_shape_dict, self.num_overall_elements = self._stat_layer_info()

        # mask_dict only includes the mask of layer that should be pruned(a.k.a sparse layer)
        # sparse_layer_set the name of the sparse layer
        # layer_density_dict includes the layer-wise densities for sparse layer (not include ignored layers)

        if self.mask_dict:
            self.layer_density_dict = self._stat_density_info()
            self.sparse_layer_set = set(self.mask_dict.keys())
            logging.debug("########### call mask dict here #########")
            logging.info(f"The sparse layers are {self.layer_density_dict}")
        else:
            self.sparse_layer_set = self._determine_sparse_layers()
            self.layer_density_dict, self.mask_dict = self._init_prune()
            logging.info(f"The sparse layers are {self.layer_density_dict}")


    def to(self, device, *args, **kwargs):
        self.device = device
        self.model.to(device, *args, **kwargs)
        for name in self.mask_dict:
            self.mask_dict[name] = self.mask_dict[name].to(device, *args, **kwargs)

    def _determine_sparse_layers(self):
        sparse_layer_set = self.layer_set.copy()
        ignore_partial_names = []
        ignore_layer_idx = []
        ignore_nn_types = []
        module_length = 0
        for _ in self.model.named_modules():
            module_length += 1

        for item in self.ignore_layers:
            if isinstance(item, str):
                ignore_partial_names.append(item)
            elif isinstance(item, int):
                ignore_layer_idx.append(item)
            elif type(item) is type:
                ignore_nn_types.append(item)
            else:
                warnings.warn(f"{type(item)} is not included in int, str and class. Therefore it will be ignored")

        def _remove_by_name(layer_set, partial_name):
            ###### remove partial names (can use prefix)########
            for layer_name in list(layer_set):
                if re.match(partial_name, layer_name) is not None:
                    layer_set.remove(layer_name)
                # elif partial_name + ".weight" in layer_name:
                #     sparse_layer_set.remove(layer_name)
            return layer_set

        for partial_name in ignore_partial_names:
            sparse_layer_set = _remove_by_name(sparse_layer_set, partial_name,)

        for e, (name, module) in enumerate(self.model.named_modules()):
            # if name == "":
            #     continue

            # if e in ignore_layer_idx:
            #     sparse_layer_set.remove(name)
            #     continue
            for t in ignore_nn_types:
                if isinstance(module, t):
                    sparse_layer_set = _remove_by_name(sparse_layer_set, name)
                    break
        
        # total_length = len(sparse_layer_set)
        # for i in range(len(ignore_layer_idx)):
        #     if ignore_layer_idx[i] < 0:
        #         ignore_layer_idx[i] += total_length
        # # must sorted
        # ignore_layer_idx.sort(reverse=True)
        # sparse_layer_set = list(sparse_layer_set)
        # for idx in ignore_layer_idx:
        #     sparse_layer_set.pop(idx)
        # sparse_layer_set = set(sparse_layer_set)
        return sparse_layer_set


    def _stat_layer_info(self):
        layer_set = set()
        layer_shape_dict = {}
        num_overall_elements = 0
        for name, weight in self.model.named_parameters():
            layer_set.add(name)
            layer_shape_dict[name] = weight.shape
            num_overall_elements += weight.numel()
        return layer_set, layer_shape_dict, num_overall_elements

    def _stat_density_info(self):
        layer_density_dict = 0
        for name, weight in self.model.named_parameters():
            if name in self.mask_dict:
                remains = self.mask_dict[name].sum().item()
                overall = self.mask_dict[name].numel()
                layer_density_dict[name] = remains / overall

        return  layer_density_dict


    def _init_prune(self, **kwargs):
        layer_density_strategy, pruning_strategy = self.strategy.split("_")
        layer_density_dict = generate_layer_density_dict(self.layer_shape_dict, self.num_overall_elements,self.sparse_layer_set, self.target_density, layer_density_strategy)
        model_mask = pruning(self.model, layer_density_dict, pruning_strategy)
        return layer_density_dict, model_mask

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def named_parameters(self, **kwargs):
        return self.model.named_parameters(**kwargs)

    @torch.no_grad()
    def apply_mask(self,):
        for name, weight in self.model.named_parameters():
            if name in self.mask_dict:
                try:
                    weight.data = weight.data * self.mask_dict[name]
                except RuntimeError:
                    raise RuntimeError(f"the device for weight is {weight.device} and mask_dict is on {self.mask_dict[name].device}") 
                
    @torch.no_grad()
    def apply_mask_gradients(self):
        """
        Applies boolean mask to modules's gradients
        """
        for name, weight in self.module.named_parameters():
            if name in self.mask_dict:
                weight.grad = weight.grad * self.mask_dict[name]

    def forward(self, x, *args, **kargs):
        # mask weight
        self.apply_mask()
        y = self.model(x, *args, **kargs)
        return y

    def stat_actual_density(self):
        num_remain_elements = 0
        actual_layer_wise_density = {}
        for name, weight in self.model.named_parameters():
            layer_remain_elements = torch.sum(weight != 0. ).item()
            num_remain_elements += layer_remain_elements
            actual_layer_wise_density[name] = layer_remain_elements / weight.numel()

        actual_density = num_remain_elements/ self.num_overall_elements

        return actual_density, actual_layer_wise_density
    
    def adjust_mask_dict(self, gradients, t, T_end, alpha):
        self.mask_dict = sparse_update_step(self.model, gradients, self.mask_dict, t, T_end, alpha)

    def prune_and_grow_fedsgc(self, weights, masks, gradient_dict, local_direction_map, t, alpha, T_end, lambda_k, beta_k, global_direction_map):
        if global_direction_map is None:
            logging.warning("global_direction_map is None. Initializing it as an empty dictionary.")
            global_direction_map = {}

        # Assign global and local direction maps for easier reference.
        d_t = global_direction_map
        delta = local_direction_map

        # Iterate through all the keys in the weights dictionary.
        for key in weights:
            # Validate if the key exists in masks, global_direction_map (d_t), and local_direction_map (delta).
            if key not in masks or key not in d_t or key not in delta:
                logging.warning(f"Skipping invalid or missing key: {key}")
                continue

            # Retrieve weight, mask, global direction, and local direction for the current key.
            weight = weights[key]
            mask = masks[key]
            global_direction = d_t[key]
            local_direction = delta[key]

            # Identify active (mask == 1) and inactive (mask == 0) weight indices.
            active_indices = (mask == 1).nonzero(as_tuple=True)[0]
            inactive_indices = (mask == 0).nonzero(as_tuple=True)[0]
            active_num = len(active_indices)

            # Determine the number of weights to prune based on the current round and alpha parameter.
            k = int(((1 - t / T_end) ** alpha) * active_num)

            # Pruning strategy 1: Prune weights where global_direction aligns oppositely with local_direction.
            valid_prune_indices_1 = active_indices[global_direction[active_indices] == -local_direction[active_indices]]
            if len(valid_prune_indices_1) > 0:
                sorted_active_by_weight_1 = valid_prune_indices_1[
                    torch.argsort(torch.abs(weight[valid_prune_indices_1]))
                ]
                num_to_prune_1 = min(int(lambda_k * k), len(sorted_active_by_weight_1))
                mask[sorted_active_by_weight_1[:num_to_prune_1]] = 0
                logging.info(f"Pruned {num_to_prune_1} weights from {key} ([d_t]_i = -[Δ]_i).")

            # Pruning strategy 2: Prune weights where global_direction does not align oppositely with local_direction.
            valid_prune_indices_2 = active_indices[global_direction[active_indices] != -local_direction[active_indices]]
            if len(valid_prune_indices_2) > 0:
                sorted_active_by_weight_2 = valid_prune_indices_2[
                    torch.argsort(torch.abs(weight[valid_prune_indices_2]))
                ]
                num_to_prune_2 = min(int((1 - lambda_k) * k), len(sorted_active_by_weight_2))
                mask[sorted_active_by_weight_2[:num_to_prune_2]] = 0
                logging.info(f"Pruned {num_to_prune_2} weights from {key} ([d_t]_i ≠ -[Δ]_i).")
            
            # what is trainer ? 
            # Retrieve the gradient for the current key. Skip growing if the gradient is not found.
            # gradient = self.trainer.get_gradient(key)
            # if gradient is None:
            #     logging.error(f"Gradient for {key} not found. Skipping growth for this key.")
            #     continue

            # add gradients dict 
            gradient = gradient_dict[key]

            # Growing strategy 1: Grow weights where global_direction aligns with local_direction.
            valid_grow_indices_1 = inactive_indices[
                global_direction[inactive_indices] == local_direction[inactive_indices]]
            if len(valid_grow_indices_1) > 0:
                sorted_inactive_by_grad_1 = valid_grow_indices_1[
                    torch.argsort(torch.abs(gradient[valid_grow_indices_1]), descending=True)
                ]
                num_to_grow_1 = min(int(beta_k * k), len(sorted_inactive_by_grad_1))
                mask[sorted_inactive_by_grad_1[:num_to_grow_1]] = 1
                logging.info(f"Grew {num_to_grow_1} weights for {key} ([d_t]_i = [Δ]_i).")

            # Growing strategy 2: Grow weights where global_direction does not align with local_direction.
            valid_grow_indices_2 = inactive_indices[
                global_direction[inactive_indices] != local_direction[inactive_indices]]
            if len(valid_grow_indices_2) > 0:
                sorted_inactive_by_grad_2 = valid_grow_indices_2[
                    torch.argsort(torch.abs(gradient[valid_grow_indices_2]), descending=True)
                ]
                num_to_grow_2 = min(int((1 - beta_k) * k), len(sorted_inactive_by_grad_2))
                mask[sorted_inactive_by_grad_2[:num_to_grow_2]] = 1
                logging.info(f"Grew {num_to_grow_2} weights for {key} ([d_t]_i ≠ [Δ]_i).")

    def prune_mask_dict(self, t, T_end, alpha):
        self.mask_dict = sparse_pruning_step(self.model, self.mask_dict, t, T_end, alpha)
    def grow_mask_dict(self, gradients):
        self.mask_dict = sparse_growing_step(self.model, gradients, self.mask_dict, self.layer_density_dict)
if __name__ == "__main__":
    from torchvision.models import resnet18
    model = resnet18()
    sparse_model = SparseModel(model, target_density=0.5, )
    #sparse_model.apply_mask()
    sparse_layer_set = sparse_model.sparse_layer_set
    print("#########ignored layers##########")
    print(sparse_model.layer_set - sparse_layer_set)
    print("###############sparse layers ##########")
    print(sparse_layer_set)
    print("#############density distribution#############")
    print(sparse_model.layer_density_dict)

    ## training
    sparse_model.to("cuda")
    optim = optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, sparse_model.parameters()), lr=0.1)
    for i in range(10):
        sparse_model.zero_grad()
        x = torch.randn([32, 3, 32, 32]).cuda()
        y = sparse_model(x)
        loss = torch.sum(y * torch.randn_like(y))
        loss.backward()
        optimizer.step()

    print("##############recheck the parameter in the training################")
    _ = sparse_model(x)
    sparse_model.zero_grad()
    actual_density, actual_layer_wise_density = sparse_model.stat_actual_density()
    print("######### actual overall density ###########")
    print(actual_density)
    print("######### actual layer wise density ###########")
    print(actual_layer_wise_density)
