import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer

from fedml_api.pruning.init_scheme import generate_layer_density_dict, pruning, growing, f_decay, magnitude_prune

class MyModelTrainer(ModelTrainer):

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)



    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        mask_dict=model.mask_dict
        layer_density_dict=model.layer_density_dict


        # train and update
        #criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
      #  initial_lr = args.initial_lr
      # final_lr = args.final_lr


        epoch_loss = []
        delta_T = 10
        T_end = 100
        alpha = 0.1 

        for epoch in range(args.epochs):
            # Calculate the decayed learning rate
            #current_lr = initial_lr * math.exp((epoch / args.epochs) * math.log(final_lr / initial_lr))
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = current_lr
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                loss = sparse_train_step(model, optimizer, x, labels, mask_dict, t=epoch*len(train_data)+batch_idx, 
                        delta_T=delta_T, T_end=T_end, alpha=alpha, layer_density_dict=layer_density_dict)
                #model.zero_grad()
                #log_probs = model(x)
                #loss = criterion(log_probs, labels)
                #loss.backward()
                #self.model.apply_mask_gradients()  # apply pruning mask

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                #optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

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
    
def sparse_train_step(model, optimizer, data, target, mask_dict, t, delta_T, T_end, alpha, layer_density_dict): #rigL
    #model.train()
    model.zero_grad()
    log_probs = model(data)
    loss = criterion(log_probs, target)
    loss.backward()
    if t % delta_T == 0 and t < T_end:
        for name, param in model.named_parameters():
            if name in mask_dict:
                num_elements = mask_dict[name].numel()
                k = f_decay(t, alpha, T_end) * (1 - layer_density_dict[name])
                inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1)
                _, topk_indices = torch.topk(torch.abs(param.data.view(-1)[inactive_indices]), k, sorted=False)
                mask_dict[name].view(-1)[inactive_indices[topk_indices]] = 1.0
    # apply mask to gradients
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.grad.data *= mask_dict[name]
    optimizer.step()
    # Newly activated connections are initialized to zero
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data *= mask_dict[name]
    return loss.item()
