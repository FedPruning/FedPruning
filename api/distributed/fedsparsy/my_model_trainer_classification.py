import logging
import os

import torch
from torch import nn
from torch.func import functional_call

try:
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def _get_base_model(self, model):
        return model.model if hasattr(model, "model") else model

    def _use_sparse_training(self, round_idx, args=None):
        dense_start_rounds = max(int(getattr(args, "dense_start_rounds", 1)), 1)
        return round_idx is not None and round_idx >= dense_start_rounds

    def _build_reparam_parameter_map(self, model, beta=1.25, clip=0.0):
        param_map = {}
        for name, param in model.named_parameters():
            if not (param.requires_grad and torch.is_floating_point(param)):
                param_map[name] = param
                continue

            transformed = torch.sign(param) * torch.abs(param).clamp_min(1e-12).pow(beta)
            if clip and clip > 0:
                transformed = torch.clamp(transformed, min=-clip, max=clip)
            param_map[name] = transformed

        return param_map

    def _raise_if_non_finite_tensor(self, tensor, tensor_name, epoch_idx, batch_idx):
        if torch.isfinite(tensor).all():
            return
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_values = detached[finite_mask]
        max_abs = finite_values.abs().max().item() if finite_values.numel() > 0 else float("nan")
        raise FloatingPointError(
            f"Non-finite {tensor_name} detected at epoch={epoch_idx}, batch={batch_idx}, max_abs_finite={max_abs}"
        )

    def _raise_if_non_finite_gradients(self, model, epoch_idx, batch_idx):
        for name, param in model.named_parameters():
            if param.grad is None or not torch.is_floating_point(param.grad):
                continue
            if torch.isfinite(param.grad).all():
                continue
            raise FloatingPointError(
                f"Non-finite gradient detected at epoch={epoch_idx}, batch={batch_idx}, parameter={name}"
            )

    def _raise_if_non_finite_parameters(self, model, epoch_idx, batch_idx):
        for name, param in model.named_parameters():
            if not torch.is_floating_point(param.data):
                continue
            if torch.isfinite(param.data).all():
                continue
            raise FloatingPointError(
                f"Non-finite parameter detected after optimizer step at epoch={epoch_idx}, batch={batch_idx}, parameter={name}"
            )

    def compute_layer_sparsity(self, model, default_sparsity=0.0):
        base_model = self._get_base_model(model)
        layer_density_dict = getattr(base_model, "layer_density_dict", {})
        layer_sparsity = {}

        for name, layer in base_model.named_modules():
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                continue

            weight_name = f"{name}.weight"
            if weight_name in layer_density_dict:
                sparsity = 1.0 - layer_density_dict[weight_name]
            else:
                sparsity = default_sparsity

            layer_sparsity[name] = min(max(sparsity, 0.0), 1.0)

        return layer_sparsity

    def _build_activation_mask(self, activation, sparsity):
        if sparsity <= 0:
            return torch.ones_like(activation)
        if sparsity >= 1:
            return torch.zeros_like(activation)

        flat_abs = activation.detach().abs().reshape(-1)
        keep_num = max(int(flat_abs.numel() * (1 - sparsity)), 1)
        if keep_num >= flat_abs.numel():
            return torch.ones_like(activation)

        threshold = torch.topk(flat_abs, keep_num, largest=True).values.min()
        return (activation.detach().abs() >= threshold).to(dtype=activation.dtype)

    def register_activation_gradient_hooks(self, model, layer_sparsity):
        base_model = self._get_base_model(model)
        activation_masks = {}
        hooks = []

        for name, layer in base_model.named_modules():
            if not isinstance(layer, (nn.Conv2d, nn.Linear)) or name not in layer_sparsity:
                continue

            sparsity = layer_sparsity[name]

            def hook(_, __, output, layer_name=name, current_sparsity=sparsity):
                if not torch.is_tensor(output) or not output.requires_grad:
                    return output
                mask = self._build_activation_mask(output, current_sparsity)
                activation_masks[layer_name] = mask.detach().cpu()
                output.register_hook(lambda grad, grad_mask=mask: grad * grad_mask)
                return output

            hooks.append(layer.register_forward_hook(hook))

        return hooks, activation_masks

    def _append_batch_metric(self, args, metric_name, payload):
        log_dir = getattr(args, "batch_metric_log_dir", "./batch_metric_logs")
        os.makedirs(log_dir, exist_ok=True)
        process_label = f"clientproc_{getattr(self, 'id', 'unknown')}"
        log_path = os.path.join(log_dir, f"{process_label}.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{metric_name} | {payload}\n")

    def _get_cosine_lr(self, args, round_idx):
        if round_idx is None:
            return args.lr
        total_rounds = args.comm_round
        import math
        # η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(t/T * π))
        lr_min = 0.0
        lr_max = args.lr
        cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (
            1 + math.cos(math.pi * round_idx / total_rounds)
        )
        return cosine_lr

    def _set_optimizer_lr(self, optimizer, target_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = target_lr

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def train(self, train_data, device, args, round_idx=None):
        model = self.model
        base_model = self._get_base_model(model)
        sparse_enabled = self._use_sparse_training(round_idx, args)

        model.to(device)
        base_model.train()

        # Calculate current Cosine LR
        current_lr = self._get_cosine_lr(args, round_idx)
        logging.info("[client %s][round %s] Cosine LR scheduled: %.6f", getattr(self, "id", None), round_idx, current_lr)

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=current_lr,
                weight_decay=args.wd,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=current_lr,
                weight_decay=args.wd,
                amsgrad=True,
            )

        beta = getattr(args, "reparam_beta", 1.25)
        reparam_per_step = getattr(args, "reparam_per_step", False)
        reparam_clip = getattr(args, "reparam_clip", 0.0)
        latest_masks = {}

        if not sparse_enabled:
            logging.info("Round %s keeps dense local training", round_idx)

        for epoch_idx in range(args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()

                hooks = []
                batch_masks = {}
                if sparse_enabled:
                    layer_sparsity = self.compute_layer_sparsity(base_model)
                    hooks, batch_masks = self.register_activation_gradient_hooks(base_model, layer_sparsity)

                try:
                    if sparse_enabled and reparam_per_step:
                        reparam_param_map = self._build_reparam_parameter_map(base_model, beta=beta, clip=reparam_clip)
                        log_probs = functional_call(base_model, reparam_param_map, (x,))
                    else:
                        log_probs = base_model(x)
                    if not torch.isfinite(log_probs).all():
                        logging.warning(
                            "[client %s][round %s][epoch %s][batch %s] non-finite logits, skip batch",
                            getattr(self, "id", None),
                            round_idx,
                            epoch_idx,
                            batch_idx,
                        )
                        continue
                    loss = criterion(log_probs, labels)
                    if not torch.isfinite(loss).all():
                        logging.warning(
                            "[client %s][round %s][epoch %s][batch %s] non-finite loss, skip batch",
                            getattr(self, "id", None),
                            round_idx,
                            epoch_idx,
                            batch_idx,
                        )
                        continue

                    if sparse_enabled and getattr(args, "log_activation_sparsity", False) and batch_masks:
                        act_log_interval = max(getattr(args, "act_sparsity_log_interval", 20), 1)
                        if batch_idx % act_log_interval == 0:
                            layer_sparsities = {}
                            for lname, mask in batch_masks.items():
                                if mask.numel() == 0:
                                    continue
                                density = mask.float().mean().item()
                                layer_sparsities[lname] = 1.0 - density
                            if layer_sparsities:
                                mean_sparsity = sum(layer_sparsities.values()) / len(layer_sparsities)
                                self._append_batch_metric(
                                    args,
                                    "act_sparsity",
                                    {
                                        "client": getattr(self, "id", None),
                                        "round": round_idx,
                                        "epoch": epoch_idx,
                                        "batch": batch_idx,
                                        "mean": round(mean_sparsity, 6),
                                        "sample": {k: round(v, 4) for k, v in list(layer_sparsities.items())[:3]},
                                    },
                                )

                    loss_log_interval = max(getattr(args, "train_loss_log_interval", 20), 1)
                    if batch_idx % loss_log_interval == 0:
                        self._append_batch_metric(
                            args,
                            "train_loss",
                            {
                                "client": getattr(self, "id", None),
                                "round": round_idx,
                                "epoch": epoch_idx,
                                "batch": batch_idx,
                                "value": round(loss.item(), 6),
                                "lr": round(optimizer.param_groups[0]["lr"], 8),
                            },
                        )
                    loss.backward()
                    self._raise_if_non_finite_gradients(base_model, epoch_idx, batch_idx)
                finally:
                    for hook in hooks:
                        hook.remove()

                optimizer.step()
                self._raise_if_non_finite_parameters(base_model, epoch_idx, batch_idx)
                latest_masks = batch_masks

        return latest_masks

    def test(self, test_data, device, args):
        model = self.model
        base_model = self._get_base_model(model)

        model.to(device)
        base_model.eval()

        metrics = {
            "Accuracy": 0,
            "Loss": 0,
            "test_total": 0,
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for x, target in test_data:
                x = x.to(device)
                target = target.to(device)

                # Same as training: if per-step reparameterization is enabled, evaluate with reparameterized weights
                if getattr(args, "reparam_per_step", False):
                    beta = getattr(args, "reparam_beta", 1.25)
                    clip = getattr(args, "reparam_clip", 0.0)
                    reparam_param_map = self._build_reparam_parameter_map(base_model, beta=beta, clip=clip)
                    pred = functional_call(base_model, reparam_param_map, (x,))
                else:
                    pred = base_model(x)

                loss = criterion(pred, target)

                self._raise_if_non_finite_tensor(pred, "eval_logits", epoch_idx=-1, batch_idx=-1)
                self._raise_if_non_finite_tensor(loss, "eval_loss", epoch_idx=-1, batch_idx=-1)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["Accuracy"] += correct.item()
                metrics["Loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        metrics["Accuracy"] /= metrics["test_total"]
        metrics["Loss"] /= metrics["test_total"]
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
