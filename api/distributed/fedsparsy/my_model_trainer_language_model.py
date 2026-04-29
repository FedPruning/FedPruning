import logging

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

try:
    from api.model.nlp.gpt2 import Conv1D
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.api.model.nlp.gpt2 import Conv1D
    from FedPruning.core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, dataset_name, args=None):
        super().__init__(model, args)
        if dataset_name == "tinystories":
            self.tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 256

    def _get_base_model(self, model):
        return model.model if hasattr(model, "model") else model

    def _use_sparse_training(self, round_idx, args=None):
        dense_start_rounds = max(int(getattr(args, "dense_start_rounds", 1)), 1)
        return round_idx is not None and round_idx >= dense_start_rounds

    def reparameterize_weights(self, model, beta=1.25):
        for _, param in model.named_parameters():
            if not (param.requires_grad and param.data is not None and torch.is_floating_point(param.data)):
                continue
            sign = torch.sign(param.data)
            magnitude = torch.abs(param.data).clamp_min(1e-12).pow(beta)
            param.data.copy_(sign * magnitude)

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
        supported_layers = (nn.Linear, Conv1D)

        for name, layer in base_model.named_modules():
            if not isinstance(layer, supported_layers):
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
        supported_layers = (nn.Linear, Conv1D)

        for name, layer in base_model.named_modules():
            if not isinstance(layer, supported_layers) or name not in layer_sparsity:
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

    def _tokenize_batch(self, batch, device):
        return self.tokenizer(
            batch["text"],
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )["input_ids"].to(device)

    def _resolve_loss_bucket_lr(self, args, loss_value, base_lr):
        if not getattr(args, "loss_based_lr_schedule", False):
            return base_lr, "disabled"
        if loss_value < 1.0:
            return getattr(args, "loss_lr_0_1", base_lr), "[0,1)"
        if loss_value < 2.0:
            return getattr(args, "loss_lr_1_2", base_lr), "[1,2)"
        if loss_value < 3.0:
            return getattr(args, "loss_lr_2_3", base_lr), "[2,3)"
        return getattr(args, "loss_lr_ge_3", base_lr), "[3,+inf)"

    def _set_optimizer_lr(self, optimizer, target_lr):
        changed = False
        for param_group in optimizer.param_groups:
            current_lr = param_group.get("lr", target_lr)
            if abs(current_lr - target_lr) > 1e-12:
                changed = True
            param_group["lr"] = target_lr
        return changed

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

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
                amsgrad=True,
            )

        beta = getattr(args, "reparam_beta", 1.25)
        base_lr = args.lr
        latest_masks = {}

        if not sparse_enabled:
            logging.info("Round %s keeps dense local training", round_idx)

        for epoch_idx in range(args.epochs):
            for batch_idx, batch in enumerate(train_data):
                tokenized = self._tokenize_batch(batch, device)
                optimizer.zero_grad()

                hooks = []
                batch_masks = {}
                if sparse_enabled:
                    layer_sparsity = self.compute_layer_sparsity(base_model)
                    hooks, batch_masks = self.register_activation_gradient_hooks(base_model, layer_sparsity)

                try:
                    logits, loss = base_model(tokenized, tokenized)
                    self._raise_if_non_finite_tensor(logits, "logits", epoch_idx, batch_idx)
                    self._raise_if_non_finite_tensor(loss, "loss", epoch_idx, batch_idx)

                    current_loss = loss.item()
                    target_lr, loss_bucket = self._resolve_loss_bucket_lr(args, current_loss, base_lr)
                    lr_changed = self._set_optimizer_lr(optimizer, target_lr)
                    if lr_changed:
                        logging.info(
                            "[client %s][round %s][epoch %s][batch %s] loss=%.6f, switch lr to %.6f (bucket %s)",
                            getattr(self, "id", None),
                            round_idx,
                            epoch_idx,
                            batch_idx,
                            current_loss,
                            target_lr,
                            loss_bucket,
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
            "Perplexity": 0,
            "test_total": 0,
        }
        nlls = []

        with torch.no_grad():
            for batch in test_data:
                tokenized = self._tokenize_batch(batch, device)
                labels = tokenized[..., 1:].cpu()
                logits, loss = base_model(tokenized, tokenized)
                self._raise_if_non_finite_tensor(logits, "eval_logits", epoch_idx=-1, batch_idx=-1)
                self._raise_if_non_finite_tensor(loss, "eval_loss", epoch_idx=-1, batch_idx=-1)
                pred_ids = torch.argmax(logits, dim=-1)[..., :-1].cpu()
                pad_token_id = self.tokenizer.pad_token_id

                metrics["Loss"] += loss.item() * len(batch["text"])
                metrics["test_total"] += len(batch["text"])

                for i in range(len(labels)):
                    hit, total = 0, 0
                    token_probs = []
                    for j in range(len(labels[i])):
                        if labels[i][j] != pad_token_id:
                            poss = torch.nn.functional.softmax(logits[i][j], dim=-1)
                            prob = max(poss[labels[i][j]].item(), 1e-12)
                            token_probs.append(prob)
                            total += 1
                            if labels[i][j] == pred_ids[i][j]:
                                hit += 1

                    if total > 0:
                        metrics["Accuracy"] += hit / total
                        nlls.append(np.exp(-np.mean(np.log(token_probs))))

        if metrics["test_total"] > 0:
            metrics["Loss"] /= metrics["test_total"]
            metrics["Accuracy"] /= metrics["test_total"]
        metrics["Perplexity"] = float(np.mean(nlls)) if nlls else 0.0

        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
