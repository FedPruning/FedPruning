import torch
from typing import Dict, List
from torch import nn
from api.pruning.init_scheme import generate_layer_density_dict, pruning, sparse_update_step, sparse_pruning_step, sparse_growing_step
import warnings
import logging

class SparseModel(nn.Module):
    def __init__(self, model,
                 target_density:float=1.,
                 strategy:str="uniform_magnitude",
                 mask_dict: dict = {},
                 ignore_layers: list[int, str, type]=[0, "bias", nn.BatchNorm2d, "bn"], 
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
        # num_elements_dict includes the number of elements in every layer
        # num_overall_elements is the number of parameters in the whole model
        self.layer_set, self.num_elements_dict, self.num_overall_elements = self._stat_layer_info()

        # mask_dict only includes the mask of layer that should be pruned(a.k.a sparse layer)
        # sparse_layer_set the name of the sparse layer
        # layer_density_dict includes the layer-wise densities for sparse layer (not include ignored layers)

        if self.mask_dict:
            self.layer_density_dict = self._stat_density_info()
            self.sparse_layer_set = set(self.mask_dict.keys())
        else:
            self.sparse_layer_set = self._determine_sparse_layers()
            self.layer_density_dict, self.mask_dict = self._init_prune()


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

        for item in self.ignore_layers:
            if isinstance(item, str):
                ignore_partial_names.append(item)
            elif isinstance(item, int):
                ignore_layer_idx.append(item)
            elif type(item) is type:
                ignore_nn_types.append(item)
            else:
                warnings.warn(f"{type(item)} is not included in int, str and class. Therefore it will be ignored")

        def _remove_by_name(layer_set, partial_name, ):
            ###### remove partial names (can use prefix)########
            for layer_name in list(layer_set):
                if partial_name in layer_name:
                    layer_set.remove(layer_name)
                # elif partial_name + ".weight" in layer_name:
                #     sparse_layer_set.remove(layer_name)
            return layer_set

        for partial_name in ignore_partial_names:
            sparse_layer_set = _remove_by_name(sparse_layer_set, partial_name,)

        for e, (name, module) in enumerate(self.model.named_modules()):
            if name in sparse_layer_set:
                if e in ignore_layer_idx:
                    sparse_layer_set.remove(name)
                    continue

                for t in ignore_nn_types:
                    if isinstance(module, t):
                        sparse_layer_set = _remove_by_name(sparse_layer_set, name)
                        break
        return sparse_layer_set


    def _stat_layer_info(self):
        layer_set = set()
        num_elements_dict = {}
        num_overall_elements = 0
        for name, weight in self.model.named_parameters():
            layer_set.add(name)
            num_elements_dict[name] = weight.numel()
            num_overall_elements += weight.numel()
        return layer_set, num_elements_dict, num_overall_elements

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
        layer_density_dict = generate_layer_density_dict(self.num_elements_dict, self.num_overall_elements,self.sparse_layer_set, self.target_density, layer_density_strategy)
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

    def forward(self, x):
        # mask weight
        self.apply_mask()
        y = self.model(x)
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

    def prune_mask_dict(self, t, T_end, alpha):
        self.mask_dict = sparse_pruning_step(self.model, self.mask_dict, t, T_end, alpha)

    def grow_mask_dict(self, gradients):
        self.mask_dict = sparse_growing_step(self.model, gradients, self.mask_dict, self.layer_density_dict)

## TODO
# actual density

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
