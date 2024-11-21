import logging

import torch
from torch import nn
from ...pruning.init_scheme import f_decay

try:
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.lambda_l2 = args.lambda_l2
        self.initial_lr = args.lr
        self.psi = args.psi_of_lr
        self.xi = args.max_lr # args.max_lr, at first set initial_lr
        self.enable_dynamic_lowest_k = args.enable_dynamic_lowest_k
        self.penalty_indices = {}

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def update_global_penalty_index(self, model, round_idx):
        global_penalty_indices = {}
        for name, param in model.named_parameters():
            if name in model.mask_dict:
                active_num = (model.mask_dict[name] == 1).int().sum().item()
                k = int(f_decay(round_idx, self.args.gamma, self.args.T_end) * active_num)
                _, idx = torch.topk(param.flatten().abs(), k, largest=False)
                global_penalty_indices[name] = idx
        self.penalty_indices = global_penalty_indices

    def calc_bae_loss(self, init_loss, round_idx, model):
        low_magnitude_params = []
        for name, param in model.named_parameters():
            if name in model.mask_dict:
                if self.enable_dynamic_lowest_k:
                    sorted_params = torch.sort(param.abs().flatten())[0]
                    active_num = (model.mask_dict[name] == 1).int().sum().item()
                    lowest_k = int(f_decay(round_idx, self.args.gamma, self.args.T_end) * active_num)
                    threshold = sorted_params[lowest_k]  # 获取前k个最小值
                    mask_low = (param.abs() <= threshold).float()
                    low_magnitude_params.append(param * mask_low)
                else:
                    low_magnitude_params.append(param.flatten()[self.penalty_indices[name]])

        l2_loss = sum(torch.norm(param, 2) for param in low_magnitude_params)
        l1_loss = sum(torch.norm(param, 1) for param in low_magnitude_params)
        total_loss = init_loss + l2_loss
        return total_loss, l1_loss

    def adjust_learning_rate(self, optimizer, l1_loss, total_round, current_round, current_step, step_per_round):
        B = step_per_round * total_round
        b = current_step + step_per_round * current_round

        decay = (2 * B - 2 * b) / (2 * B - b)
        sigmoid_adjustment = 2 * torch.sigmoid(l1_loss) - 1
        adjusted_lr = decay * sigmoid_adjustment * self.initial_lr

        for param_group in optimizer.param_groups:
            final_lr = min(self.xi, max(param_group['lr'], self.psi * adjusted_lr))
            param_group['lr'] = final_lr

    def train(self, train_data, device, args, mode, round_idx = None):

        # mode 0 : training with mask 
        # mode 1 : training with mask 
        # mode 2 : training with mask, calculate the gradient
        # mode 3 : training with mask, calculate the gradient
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        if not args.enable_dynamic_lowest_k:
            self.update_global_penalty_index(model, round_idx)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                loss, l1_loss = self.calc_bae_loss(loss, round_idx, model)
                self.adjust_learning_rate(optimizer, l1_loss, args.epochs, epoch, batch_idx, len(train_data))

                loss.backward()
                #self.model.apply_mask_gradients()  # apply pruning mask
                
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        # Collect gradients
        if mode in [2, 3]:
            model.zero_grad()
            if args.growth_data_mode == "random":
                return {name: torch.randn_like(param, device='cpu').clone() for name, param in model.named_parameters() if param.requires_grad}

            elif args.growth_data_mode == "single":
                x, labels = next(iter(train_data))
                x, labels = x[0].unsqueeze(0).repeat(2, 1, 1, 1).to(device), labels[0].unsqueeze(0).repeat(2).to(device)  # Duplicate the sample to create a pseudo-batch
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
            else:
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    if args.growth_data_mode == "batch":
                        break
                    
            gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
            model.zero_grad()
            return gradients

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
    