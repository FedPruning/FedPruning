import copy
import logging

import torch
from torch import nn

try:
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)
    
    @torch.no_grad()
    def proximal_term_compute(self, global_model, local_model):
        proximal_term = 0.0
        for (local_name, weight), (global_name, weight_t) in zip(local_model.named_parameters(), global_model.named_parameters()):
            proximal_term += (weight - weight_t).norm(2)
        return proximal_term
    
    def compute_current_lambda_shrink(self, args, round_idx, num_segments=20):
        initial_lambda = 5e-04
        final_lambda = args.lambda_shrink
        epochs = args.comm_round
        step_size = (final_lambda - initial_lambda) / (num_segments - 1)
        segment_length = epochs // num_segments
        lambda_schedule = []
        for i in range(num_segments):
            if i == 0:
                current_lambda = 0
            else:
                current_lambda = initial_lambda + step_size * (i - 1)
            lambda_schedule.extend([current_lambda] * segment_length)
            if i == num_segments - 1:
                # Adjust the last segment to include the remaining epochs
                lambda_schedule.extend([current_lambda] * (epochs % num_segments))

        return lambda_schedule[round_idx]

    def prunable_layer_norm(self, current_lambda):
        norm_sum = 0
        for name, weight in self.model.named_parameters():
            if name in self.model.mask_dict:
                norm_sum += current_lambda * torch.norm(weight, 2)
        return norm_sum

    def train(self, train_data, device, args, mode, round_idx = None):

        # mode 0 :  training with mask 
        # mode 1 : training with mask 
        # mode 2 : training with mask, calculate the gradient
        # mode 3 : training with mask, calculate the gradient
        model = self.model
        global_model = copy.copy(self.model)
        
        model.to(device)
        global_model.to(device)
        model.train()

        
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        if mode in [2, 3]:
            local_epochs = args.adjustment_epochs if args.adjustment_epochs is not None else args.epochs
        else:
            local_epochs = args.epochs
        
        if mode in [2, 3]:
            A_epochs = local_epochs // 2 if args.A_epochs is None else args.A_epochs
            first_epochs = min(local_epochs, A_epochs)
        else:
            first_epochs = local_epochs

        epoch_loss = []
        for epoch in range(first_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                proximal_term = self.proximal_term_compute(global_model, model)
                loss += (args.mu / 2) * proximal_term
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
            
        for epoch in range(first_epochs, local_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                current_lambda = self.compute_current_lambda_shrink(args=args, round_idx=round_idx)
                prunable_layer_norm = self.prunable_layer_norm(current_lambda)
                loss += prunable_layer_norm
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

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
    