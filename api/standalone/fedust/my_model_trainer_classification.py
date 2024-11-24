import logging

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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

    def train(self, train_data, forgotten_set, device, args, mode, round_idx = None):

        # mode 0 : training with mask
        # mode 1 : training with mask 
        # mode 2 : training with mask, calculate mask
        # mode 3 : training with mask, calculate mask
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
            
        epoch_loss = []

        if mode in [2, 3]:
            local_epochs = args.adjustment_epochs if args.adjustment_epochs is not None else args.epochs
        else:
            local_epochs = args.epochs

        new_forgotten_set = []
        if mode in [2, 3]:
            A_epochs = local_epochs // 2 if args.A_epochs is None else args.A_epochs
            first_epochs = min(local_epochs, A_epochs)
        else:
            first_epochs = args.epochs

        # first training
        for epoch in range(first_epochs):
            batch_loss = []
            for batch_idx, (x, labels, index) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        if mode in [2, 3]:
            # all predicted result
            result = {}
            with torch.no_grad():
                for batch_idx, (x, target, index) in enumerate(train_data):
                    x = x.to(device)
                    pred = model(x)
                    _, predicted = torch.max(pred, -1)
                    if args.forgotten_correct == 1:
                        for i in range(predicted.size(0)):
                            if predicted[i] == target[i]:
                                result[index[i].item()] = predicted[i]
                    else:
                        for i in range(predicted.size(0)):
                            result[index[i].item()] = predicted[i]

            # pruning
            model.prune_mask_dict(t=round_idx, T_end=args.T_end, alpha=args.adjust_alpha)
            model.apply_mask()

            # update new forgotten set
            with torch.no_grad():
                for batch_idx, (x, target, index) in enumerate(train_data):
                    x = x.to(device)
                    pred = model(x)
                    _, predicted = torch.max(pred, -1)
                    for i in range(predicted.size(0)):
                        if index[i].item() in result and result[index[i].item()] != predicted[i]:
                            new_forgotten_set.append(index[i].item())

            # growing
            if len(new_forgotten_set) > 1:
                x_tensors = []
                y_tensors = []
                # Collect (x, y) pairs from the old DataLoader at (batch_idx, i)
                for batch_idx, (x, target, index) in enumerate(train_data):
                    for i in range(x.size(0)):
                        if index[i].item() in new_forgotten_set:
                            x_tensors.append(x[i])
                            y_tensors.append(target[i])

                selected_x = torch.stack(x_tensors).to(device)
                selected_y = torch.stack(y_tensors).to(device)

                # Create a DataLoader for the forgotten_dataset
                forgotten_dataset = TensorDataset(selected_x, selected_y)
                forgotten_loader = DataLoader(forgotten_dataset, batch_size=args.batch_size, shuffle=True)

                # forgotten gradient
                for batch_idx, (x, labels) in enumerate(forgotten_loader):
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    if args.growth_data_mode == "batch":
                        break
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
            # batch
            else:
                for batch_idx, (x, labels, index) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    if args.growth_data_mode == "batch":
                        break
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
            model.grow_mask_dict(gradients)
            model.apply_mask()
            model.zero_grad()

        # training after adjustment
        if args.forgotten_train == 1 and len(new_forgotten_set) > 1:
            for epoch in range(first_epochs, local_epochs):
                batch_loss = []
                for batch_idx, (x, labels) in enumerate(forgotten_loader):
                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        else:
            for epoch in range(first_epochs, local_epochs):
                batch_loss = []
                for batch_idx, (x, labels, index) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        logging.info('Client Index = {}\told_forgotten_set_len: {}\tnew_forgotten_set_len: {}'.format(self.id, len(forgotten_set), len(new_forgotten_set)))

        return model.mask_dict, new_forgotten_set

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'Accuracy': 0,
            'Loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target, index) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['Accuracy'] += correct.item()
                metrics['Loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        
        metrics['Accuracy'] /= metrics['test_total'] 
        metrics['Loss'] /= metrics['test_total'] 
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False