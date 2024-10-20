import torch
import random
import numpy as np 

def generate_layer_density_dict(num_elements_dict, num_overall_elements, sparse_layer_set, target_density, layer_density_strategy):
    # the maximum number of elements
    num_remain_elements = int(target_density * num_overall_elements)

    # the number of elements in the dense layer and sparse layer
    num_dense_elements = 0
    for name, number in num_elements_dict.items():
        if name not in sparse_layer_set:
            num_dense_elements += number

    assert num_remain_elements > num_dense_elements, f"the number of elements({num_dense_elements}) left in dense model is higher than minimum elements  requirement ({num_remain_elements}) under target density {target_density}. Please use higher target density or fewer ignore dense layers "

    num_remain_sparse_elements = num_remain_elements - num_dense_elements

    layer_density_dict = {}
    if layer_density_strategy == "uniform":
        layer_wise_density = num_remain_sparse_elements/(num_overall_elements - num_dense_elements)

        for name, number in num_elements_dict.items():
            if name in sparse_layer_set:
                assert int(number * layer_wise_density) >= 1 , f"the layer wise density {layer_wise_density} is so small that make {name} to be empty"

                layer_density_dict[name] = layer_wise_density

    elif layer_density_strategy == "erdos-renyi":
        pass

    else:
        raise Exception(f"layer density strategy {layer_density_strategy} is not supported")

    return layer_density_dict


def pruning(model, layer_density_dict, pruning_strategy):
    mask_dict = {}
    for name, weight in  model.named_parameters():
        if name in layer_density_dict:
            density = layer_density_dict[name]
            num_elements = weight.numel()
            mask = torch.zeros_like(
                weight, dtype=weight.data.dtype, requires_grad=False
            )

            if pruning_strategy in ["mag", "magnitude"]:
                mask_dict[name] = magnitude_prune(weight, mask, num_elements, density)
            elif pruning_strategy in ["random"]:
                mask_dict[name] =  random_prune(mask, num_elements, density)
            elif pruning_strategy in ["structure-mag"]:
                pass
            else:
                raise Exception(f"pruning strategy {pruning_strategy} is not supported")
    return mask_dict

def magnitude_prune(weight, mask, num_elements, density):
    num_remove = num_elements - int(num_elements * density)

    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[num_remove:]] = 1.0
    return mask

def random_prune(mask, num_elements, density):
    num_remove = num_elements - int(num_elements * density)
    idx = list(range(num_elements))
    random.shuffle(idx)
    mask.data.view(-1)[idx[num_remove:]] = 1.0
    return mask


def f_decay(t, alpha, T_end):
    # return int(alpha * (1 - t / T_end))
    return alpha / 2 * (1 + np.cos(t * np.pi / T_end))


def growing(model, mask_dict, growth_percentage):
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask_dict:
            grad_dict[name] = param.grad.abs().view(-1)
    
    for name, mask in mask_dict.items():
        if name in grad_dict:
            num_elements = mask.numel()
            num_grow = int(num_elements * growth_percentage)

            grad = grad_dict[name]
            inactive_indices = (mask.view(-1) == 0).nonzero(as_tuple=False).view(-1)
            grad_inactive = grad[inactive_indices]

            _, topk_indices = torch.topk(grad_inactive, num_grow, sorted=False)
            new_growth_indices = inactive_indices[topk_indices]

            mask.view(-1)[new_growth_indices] = 1.0

    return mask_dict


# 
def sparse_update_step(model, gradients, mask_dict, t, T_end, alpha):
    for name, param in model.named_parameters():
        if name in mask_dict:
            #num_elements = mask_dict[name].numel()
            # k = f_decay(t, alpha, T_end) * (1 - layer_density_dict[name])
            
            active_num = (mask_dict[name] == 1).int().sum().item()
            k = int(f_decay(t, alpha, T_end) * active_num)
            # pruning：Find the k  smallest connections among the current active connections and set them to non-active
            active_indices = (mask_dict[name].view(-1) == 1).nonzero(as_tuple=False).view(-1).cpu()
            _, prune_indices = torch.topk(torch.abs(param.data.view(-1)[active_indices]), k, largest=False)
            
            mask_dict[name].view(-1)[active_indices[prune_indices.cpu()]] = 0
            

            # growing：Find the k  largest gradients connections among the currently inactive connections and set them to active
            inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1).cpu()
            
            grad_inactive = gradients[name].abs().view(-1)[inactive_indices].cpu()
            _, grow_indices = torch.topk(grad_inactive, k, sorted=False)
            mask_dict[name].view(-1)[inactive_indices[grow_indices.cpu()]] = 1
    return mask_dict





