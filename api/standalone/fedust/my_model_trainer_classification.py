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

    def train(self, train_data, forgotten_set, device, args, mode, round_idx = None):

        # mode 0 :  training with mask 
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
        
        if mode in [2, 3]:
            A_epochs = local_epochs // 2 if args.A_epochs is None else args.A_epochs
            first_epochs = min(local_epochs, A_epochs)
            new_forgotten_set = []
        else:
            first_epochs = args.epochs
            new_forgotten_set = []
            
            # pred_and_statistics = {}
            # for i in forgotten_set:
            #     pred_and_statistics[i] = [None, 0]

        for epoch in range(first_epochs):
            batch_loss = []
            for batch_idx, (x, labels, index) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)

                # update pred_and_statistics
                # if mode in [2, 3]:
                #     _, predicted = torch.max(log_probs, -1)
                #     for i in range(predicted.size(0)):
                #         if index[i].item() in forgotten_set:
                #             if pred_and_statistics[index[i].item()][0] != predicted[i] and pred_and_statistics[index[i].item()][0] is not None:
                #                 pred_and_statistics[index[i].item()][0] = predicted[i]
                #                 pred_and_statistics[index[i].item()][1] += 1

                loss = criterion(log_probs, labels)
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

        if mode in [2, 3]:

            # for k, v in pred_and_statistics.items():
            #     if v[1] > args.forgotten_sigma * first_epochs:
            #         new_forgotten_set.append(k)

            # all predicted result
            result = {}
            with torch.no_grad():
                for batch_idx, (x, target, index) in enumerate(train_data):
                    x = x.to(device)
                    pred = model(x)
                    _, predicted = torch.max(pred, -1)
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
            if len(new_forgotten_set) > 0:
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
                log_probs = model(selected_x)
                loss = criterion(log_probs, selected_y)
                loss.backward()
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
            else:
                for batch_idx, (x, labels, index) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    break
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
            model.grow_mask_dict(gradients)
            model.apply_mask()
            model.zero_grad()

        for epoch in range(first_epochs, args.epochs):
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
            'test_correct': 0,
            'test_loss': 0,
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

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False