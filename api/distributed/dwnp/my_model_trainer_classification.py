import torch
from torch import nn

try:
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.core.trainer.model_trainer import ModelTrainer


class DeviceWiseHyperNetwork(nn.Module):
    def __init__(self, client_num_in_total, num_sparse_layers, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.client_embeddings = nn.Embedding(client_num_in_total + 1, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_sparse_layers),
        )

    def _gumbel_sigmoid(self, logits, temperature=1.0, stochastic=False, hard=False):
        temperature = max(float(temperature), 1e-6)

        if stochastic:
            eps = 1e-10
            uniform = torch.rand_like(logits)
            logistic_noise = torch.log(uniform + eps) - torch.log(1.0 - uniform + eps)
            y_soft = torch.sigmoid((logits + logistic_noise) / temperature)
        else:
            y_soft = torch.sigmoid(logits / temperature)

        if hard:
            y_hard = (y_soft >= 0.5).to(y_soft.dtype)
            # Straight-through estimator: hard in forward, soft in backward.
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def forward(self, client_codes, temperature=1.0, stochastic=False, hard=False):
        emb = self.client_embeddings(client_codes)
        logits = self.mlp(emb)
        return self._gumbel_sigmoid(
            logits,
            temperature=temperature,
            stochastic=stochastic,
            hard=hard,
        )


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.args = args

        self.sparse_layer_names = list(self.model.mask_dict.keys())
        num_sparse_layers = len(self.sparse_layer_names)
        self.server_mask_dict = {k: v.detach().clone() for k, v in self.model.mask_dict.items()}
        self.device_mask_cache = dict()

        emb_dim = getattr(args, "hn_embedding_dim", 32)
        hidden_dim = getattr(args, "hn_hidden_dim", 64)
        self.hypernet = DeviceWiseHyperNetwork(
            args.client_num_in_total,
            num_sparse_layers,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def get_hypernet_params(self):
        state = self.hypernet.cpu().state_dict()
        return {k: v for k, v in state.items() if k != "client_embeddings.weight"}

    def set_hypernet_params(self, hn_parameters):
        current = self.hypernet.state_dict()
        for k, v in hn_parameters.items():
            if k in current and k != "client_embeddings.weight":
                current[k] = v
        self.hypernet.load_state_dict(current, strict=False)

    def get_dynamic_emb_params(self):
        return {"client_embeddings.weight": self.hypernet.client_embeddings.weight.detach().cpu()}

    def set_dynamic_emb_params(self, emb_parameters):
        if "client_embeddings.weight" in emb_parameters:
            self.hypernet.client_embeddings.weight.data.copy_(emb_parameters["client_embeddings.weight"])

    def build_server_mask(self, device, round_idx=None):
        self.model.to(device)
        self.hypernet.to(device)

        min_r = getattr(self.args, "hn_min_retention", 0.05)
        max_r = getattr(self.args, "hn_max_retention", 1.0)
        temperature = getattr(self.args, "hn_temperature", 0.5)
        use_hn_mask = bool(getattr(self.args, "use_hn_server_mask", False)) or bool(
            getattr(self.args, "dwnp_enable_final_prune", 1)
        )

        if use_hn_mask:
            server_code = torch.tensor([0], dtype=torch.long, device=device)
            raw_server_ratio = self.hypernet(
                server_code,
                temperature=temperature,
                stochastic=False,
                hard=False,
            ).squeeze(0)
            server_ratio = min_r + (max_r - min_r) * raw_server_ratio
            return self._ratios_to_masks(server_ratio)

        return {k: v.detach().clone().to(device) for k, v in self.server_mask_dict.items()}

    def apply_server_pruning(self, device, round_idx=None):
        server_mask = self.build_server_mask(device, round_idx=round_idx)
        self.model.mask_dict = server_mask
        self.model.apply_mask()
        return server_mask

    def _build_single_mask(self, weights, retain_ratio):
        flat = weights.detach().abs().view(-1)
        k = max(1, int(retain_ratio * flat.numel()))
        if k >= flat.numel():
            return torch.ones_like(weights, dtype=weights.dtype, device=weights.device)
        threshold = torch.topk(flat, k, largest=True).values.min()
        return (weights.detach().abs() >= threshold).to(weights.dtype)

    def _build_single_mask_with_base(self, weights, retain_ratio, base_mask):
        base_mask = base_mask.to(weights.device)
        active_idx = (base_mask.view(-1) > 0.5).nonzero(as_tuple=False).view(-1)
        if active_idx.numel() == 0:
            return torch.ones_like(base_mask, dtype=weights.dtype, device=weights.device)

        active_w = weights.detach().abs().view(-1)[active_idx]
        k = max(1, int(retain_ratio * active_idx.numel()))
        k = min(k, active_idx.numel())

        top_idx = torch.topk(active_w, k, largest=True).indices
        chosen = active_idx[top_idx]

        out = torch.ones_like(base_mask, dtype=weights.dtype, device=weights.device).view(-1)
        out[active_idx] = 0.0
        out[chosen] = 1.0
        return out.view_as(base_mask)

    def _ratios_to_masks(self, ratios):
        mask_dict = dict()
        named_params = dict(self.model.model.named_parameters())
        for idx, name in enumerate(self.sparse_layer_names):
            param = named_params[name]
            mask_dict[name] = self._build_single_mask(param, ratios[idx].item())
        return mask_dict

    def _ratios_to_device_masks(self, ratios, server_mask_dict):
        mask_dict = dict()
        named_params = dict(self.model.model.named_parameters())
        for idx, name in enumerate(self.sparse_layer_names):
            param = named_params[name]
            base_mask = server_mask_dict[name]
            mask_dict[name] = self._build_single_mask_with_base(param, ratios[idx].item(), base_mask)
        return mask_dict

    def _density(self, mask_dict):
        remain, total = 0.0, 0.0
        for name in self.sparse_layer_names:
            m = mask_dict[name]
            remain += float(m.sum().item())
            total += float(m.numel())
        return remain / max(total, 1.0)

    def _smoothness_penalty(self, gates):
        if gates.numel() <= 1:
            return torch.tensor(0.0, device=gates.device)
        return (gates[1:] - gates[:-1]).pow(2).mean()

    def prepare_joint_mask(self, client_index, round_idx, device, args):
        self.model.to(device)
        self.hypernet.to(device)

        min_r = getattr(args, "hn_min_retention", 0.05)
        max_r = getattr(args, "hn_max_retention", 1.0)
        warmup_rounds = getattr(args, "dwnp_warmup_rounds", 0)
        temperature = getattr(args, "hn_temperature", 0.5)
        stochastic_gate = bool(getattr(args, "hn_stochastic_gate", 0))

        server_code = torch.tensor([0], dtype=torch.long, device=device)
        device_code = torch.tensor([int(client_index) + 1], dtype=torch.long, device=device)

        raw_server_ratio = self.hypernet(
            server_code,
            temperature=temperature,
            stochastic=stochastic_gate,
            hard=False,
        ).squeeze(0)
        raw_device_ratio = self.hypernet(
            device_code,
            temperature=temperature,
            stochastic=stochastic_gate,
            hard=False,
        ).squeeze(0)

        server_ratio = min_r + (max_r - min_r) * raw_server_ratio
        device_ratio = min_r + (max_r - min_r) * raw_device_ratio

        if getattr(args, "use_hn_server_mask", False):
            server_mask = self._ratios_to_masks(server_ratio)
            self.server_mask_dict = {k: v.detach().clone() for k, v in server_mask.items()}
        else:
            server_mask = {k: v.detach().clone().to(device) for k, v in self.server_mask_dict.items()}

        if round_idx is not None and round_idx < warmup_rounds:
            device_mask = {k: torch.ones_like(v, device=device) for k, v in server_mask.items()}
        else:
            device_mask = self._ratios_to_device_masks(device_ratio, server_mask)

        self.device_mask_cache[int(client_index)] = {k: v.detach().clone() for k, v in device_mask.items()}

        joint_mask = dict()
        for name in self.sparse_layer_names:
            joint_mask[name] = (server_mask[name] * device_mask[name]).to(device)

        self.model.mask_dict = joint_mask
        self.model.apply_mask()

        server_density = self._density(server_mask)
        device_density = self._density(joint_mask)
        return server_density, device_density

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
                amsgrad=True,
            )

        for _ in range(args.epochs):
            for x, labels in train_data:
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

    def _sample_batch_task_loss(self, train_data, device):
        criterion = nn.CrossEntropyLoss().to(device)
        for x, labels in train_data:
            x, labels = x.to(device), labels.to(device)
            with torch.no_grad():
                logits = self.model(x)
                return criterion(logits, labels).detach()
        return torch.tensor(0.0, device=device)

    def train_hypernetwork(self, train_data, client_index, device, args):
        self.hypernet.to(device)
        optimizer = torch.optim.Adam(self.hypernet.parameters(), lr=args.hn_lr)

        target_server = torch.tensor(args.server_flops_retention, dtype=torch.float32, device=device)
        target_device = torch.tensor(args.device_flops_retention, dtype=torch.float32, device=device)

        min_r = getattr(args, "hn_min_retention", 0.05)
        max_r = getattr(args, "hn_max_retention", 1.0)
        smooth_coeff = getattr(args, "hn_smooth_coeff", 0.1)
        task_loss_alpha = getattr(args, "task_loss_alpha", 0.0)
        temperature = getattr(args, "hn_temperature", 0.5)

        for _ in range(max(1, args.hn_local_epochs)):
            optimizer.zero_grad()
            server_code = torch.tensor([0], dtype=torch.long, device=device)
            device_code = torch.tensor([int(client_index) + 1], dtype=torch.long, device=device)

            raw_server_ratio = self.hypernet(
                server_code,
                temperature=temperature,
                stochastic=True,
                hard=True,
            ).squeeze(0)
            raw_device_ratio = self.hypernet(
                device_code,
                temperature=temperature,
                stochastic=True,
                hard=True,
            ).squeeze(0)

            server_ratio = min_r + (max_r - min_r) * raw_server_ratio
            device_ratio = min_r + (max_r - min_r) * raw_device_ratio

            flops_loss = (server_ratio.mean() - target_server) ** 2 + (device_ratio.mean() - target_device) ** 2
            smooth_loss = self._smoothness_penalty(server_ratio) + self._smoothness_penalty(device_ratio)

            # Detached task-loss acts as a scalar anchor to modulate regularization strength.
            task_loss = self._sample_batch_task_loss(train_data, device)
            reg_loss = args.lambda_reg * flops_loss + smooth_coeff * smooth_loss
            loss = (1.0 + task_loss_alpha * task_loss) * reg_loss
            loss.backward()
            optimizer.step()

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "Accuracy": 0,
            "Loss": 0,
            "test_total": 0,
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for _, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["Accuracy"] += correct.item()
                metrics["Loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        metrics["Accuracy"] /= max(metrics["test_total"], 1)
        metrics["Loss"] /= max(metrics["test_total"], 1)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False