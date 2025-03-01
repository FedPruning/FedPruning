import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import logging
from typing import Dict, List, Union, Optional
from api.quantization.quantize import uniform_quantize, weight_quantize_fn, activation_quantize_fn, conv2d_Q_fn, linear_Q_fn


class QuantizationModel(nn.Module):
    def __init__(self, model,
                 w_bit: int = 8,
                 a_bit: int = 8,
                 g_bit: Optional[int] = None,
                 strategy: str = "per_tensor",
                 ignore_layers: list = [".*bias.*", nn.BatchNorm2d, ".*bn.*", nn.LayerNorm, ".*ln.*"],
                 device = None):
        """
        Quantization model class, supporting quantization for weights, activations, and gradients
        
        Args:
            model: Original model to be quantized
            w_bit: Bitwidth for weight quantization
            a_bit: Bitwidth for activation quantization
            g_bit: Bitwidth for gradient quantization, None means no gradient quantization
            strategy: Quantization strategy, options are "per_tensor", "per_channel"
            ignore_layers: Layers to be ignored, can be regex strings matching layer names, layer types, or integer indices
            device: Model device
        """
        super(QuantizationModel, self).__init__()
        self.model = model
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.g_bit = g_bit
        self.strategy = strategy
        self.ignore_layers = ignore_layers
        self.device = device
        
        # Collect model layer information
        self.layer_set, self.layer_shape_dict, self.num_overall_elements = self._stat_layer_info()
        
        # Determine which layers need to be quantized
        self.quant_layer_set = self._determine_quant_layers()
        
        # Replace the layers with quantized versions
        self._replace_layers_with_quantized_version()
        
        # Create activation quantizer for all activations
        self.act_quantizer = activation_quantize_fn(a_bit)
        
        # Register gradient hooks if gradient quantization is needed
        self.grad_hooks = []
        self.activation_hooks = []
        if g_bit is not None and g_bit != 32:
            self._register_grad_hooks()
        
        # Register activation hooks if activation quantization is needed
        if a_bit < 32:
            self._register_activation_hooks()

    def _stat_layer_info(self):
        """Collect statistics about model layers"""
        layer_set = set()
        layer_shape_dict = {}
        num_overall_elements = 0
        for name, weight in self.model.named_parameters():
            layer_set.add(name)
            layer_shape_dict[name] = weight.shape
            num_overall_elements += weight.numel()
        return layer_set, layer_shape_dict, num_overall_elements

    def _determine_quant_layers(self):
        """Determine which layers need to be quantized"""
        quant_layer_set = self.layer_set.copy()
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
                logging.warning(f"{type(item)} is not included in int, str and class. Therefore it will be ignored")
        
        def _remove_by_name(layer_set, partial_name):
            for layer_name in list(layer_set):
                if re.match(partial_name, layer_name) is not None:
                    layer_set.remove(layer_name)
            return layer_set
        
        for partial_name in ignore_partial_names:
            quant_layer_set = _remove_by_name(quant_layer_set, partial_name)
        
        for e, (name, module) in enumerate(self.model.named_modules()):
            for t in ignore_nn_types:
                if isinstance(module, t):
                    quant_layer_set = _remove_by_name(quant_layer_set, name)
                    break
        
        return quant_layer_set

    def _replace_layers_with_quantized_version(self):
        """Replace layers with their quantized versions"""
        # Track replaced modules to prevent duplicated replacements
        replaced_modules = set()
        
        for name, module in self.model.named_modules():
            # Check if the current module needs to be replaced
            if name in replaced_modules:
                continue
                
            # Replace convolutional layers
            if isinstance(module, nn.Conv2d):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                layer_name = name.split('.')[-1]
                
                # Check if this layer should be quantized
                param_name = f"{name}.weight"
                if param_name not in self.quant_layer_set:
                    continue
                
                # Create quantized version of conv layer
                Conv2d_Q = conv2d_Q_fn(self.w_bit)
                quantized_conv = Conv2d_Q(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None
                )
                
                # Copy weights and biases
                quantized_conv.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quantized_conv.bias.data = module.bias.data.clone()
                
                # Replace the original module
                if parent_name:
                    parent = self.model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, layer_name, quantized_conv)
                else:
                    setattr(self.model, layer_name, quantized_conv)
                
                replaced_modules.add(name)
            
            # Replace linear layers
            elif isinstance(module, nn.Linear):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                layer_name = name.split('.')[-1]
                
                # Check if this layer should be quantized
                param_name = f"{name}.weight"
                if param_name not in self.quant_layer_set:
                    continue
                
                # Create quantized version of linear layer
                Linear_Q = linear_Q_fn(self.w_bit)
                quantized_linear = Linear_Q(
                    module.in_features,
                    module.out_features,
                    module.bias is not None
                )
                
                # Copy weights and biases
                quantized_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    quantized_linear.bias.data = module.bias.data.clone()
                
                # Replace the original module
                if parent_name:
                    parent = self.model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, layer_name, quantized_linear)
                else:
                    setattr(self.model, layer_name, quantized_linear)
                
                replaced_modules.add(name)

    def _register_grad_hooks(self):
        """Register hooks for gradient quantization for all parameters"""
        for name, param in self.model.named_parameters():
            if name in self.quant_layer_set and param.requires_grad:
                hook = param.register_hook(lambda grad: uniform_quantize(k=32, k_g=self.g_bit)(grad))
                self.grad_hooks.append(hook)

    def _register_activation_hooks(self):
        """Register hooks for activation quantization"""
        def _activation_hook(module, input, output):
            return self.act_quantizer(output)
            
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
                hook = module.register_forward_hook(_activation_hook)
                self.activation_hooks.append(hook)

    def forward(self, x, *args, **kwargs):
        """Forward pass, using the original model's forward pass with activation quantization done through hooks"""
        return self.model(x, *args, **kwargs)

    def to(self, device, *args, **kwargs):
        """Move the model to the specified device"""
        self.device = device
        self.model.to(device, *args, **kwargs)
        return self
    
    def parameters(self, **kwargs):
        """Return model parameters"""
        return self.model.parameters(**kwargs)
    
    def named_parameters(self, **kwargs):
        """Return named model parameters"""
        return self.model.named_parameters(**kwargs)
    
    def remove_hooks(self):
        """Remove all quantization hooks"""
        for hook in self.grad_hooks:
            hook.remove()
        self.grad_hooks = []
        
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []

    def stat_bitwidth_info(self):
        """Collect bitwidth information statistics"""
        stats = {
            "weight_bit": self.w_bit,
            "activation_bit": self.a_bit,
            "gradient_bit": self.g_bit,
            "ignored_layers": list(self.layer_set - self.quant_layer_set),
            "quantized_layers": list(self.quant_layer_set)
        }
        return stats


if __name__ == "__main__":
    # Import ResNet18 model for testing
    from torchvision.models import resnet18
    
    # Create original model
    model = resnet18()
    print("Original model created")
    
    # Create quantized model with 8-bit weights, 8-bit activations, no gradient quantization
    quant_model = QuantizationModel(
        model, 
        w_bit=8, 
        a_bit=8, 
        g_bit=None,
        strategy="per_tensor"
    )
    print("Quantized model created")
    
    # Display quantization information
    bitwidth_info = quant_model.stat_bitwidth_info()
    print("\nQuantization bitwidth information:")
    for key, value in bitwidth_info.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    # Test model inference
    quant_model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(quant_model.parameters()).device
    
    # Create random input data
    x = torch.randn(1, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        y = quant_model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output type: {y.dtype}")
    
    # Test training process
    optimizer = torch.optim.SGD(quant_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create random target
    target = torch.randint(0, 1000, (1,)).to(device)
    
    # Forward pass, backward pass, optimization
    quant_model.zero_grad()
    y = quant_model(x)
    loss = criterion(y, target)
    loss.backward()
    optimizer.step()
    
    print("\nCompleted one training iteration")
    print(f"Loss value: {loss.item()}")
    
    # Test different bitwidth quantization
    print("\nTesting different bitwidth quantization:")
    
    # Create binary weight model
    binary_model = QuantizationModel(
        resnet18(),
        w_bit=1,
        a_bit=8,
        g_bit=None
    )
    binary_model.to(device)
    
    # Create 4-bit weight, 4-bit activation model
    low_precision_model = QuantizationModel(
        resnet18(),
        w_bit=4,
        a_bit=4,
        g_bit=8
    )
    low_precision_model.to(device)
    
    # Forward pass test
    with torch.no_grad():
        y1 = binary_model(x)
        y2 = low_precision_model(x)
    
    print(f"Binary weight model output shape: {y1.shape}")
    print(f"4-bit weight/4-bit activation model output shape: {y2.shape}")
    print("\nQuantization test completed!")
    
    # Remove all hooks after testing
    quant_model.remove_hooks()
    binary_model.remove_hooks()
    low_precision_model.remove_hooks()