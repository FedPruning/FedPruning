import torch
import random

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