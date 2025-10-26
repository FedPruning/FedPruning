import logging

import torch
from torch import nn
from transformers import AutoTokenizer
import evaluate
import numpy as np

from core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):
    def __init__(self, model,  dataset_name, args=None,):
        super().__init__(model, args)
        if dataset_name == "tinystories":
            self.tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.rouge = evaluate.load('rouge')

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def train(self, train_data, device, args, mode, round_idx = None):

        # mode 0 :  training with mask 
        # mode 1 : training with mask 
        # mode 2 : training with mask, calculate the gradient
        # mode 3 : training with mask, calculate the gradient
        model = self.model

        model.to(device)
        model.train()

        # train and update
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=args.wd, amsgrad=True)

        if mode in [2, 3]:
            local_epochs = args.adjustment_epochs if args.adjustment_epochs is not None else args.epochs
        else:
            local_epochs = args.epochs

        epoch_loss = []
        for epoch in range(local_epochs):
            batch_loss = []
            for batch_idx, batch in enumerate(train_data):
                tokenized = self.tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=256, truncation=True)['input_ids'].to(device)
                model.zero_grad()
                logits, loss = model(tokenized, tokenized)
                loss.backward()
                #self.model.apply_mask_gradients()  # apply pruning mask
                
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))

            #     batch_loss.append(loss.item())
            # epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        # Collect gradients
        if mode in [2, 3]:
            model.zero_grad()
            if args.growth_data_mode == "random":
                return {name: torch.randn_like(param, device='cpu').clone() for name, param in model.named_parameters() if param.requires_grad}

            else:
                for batch_idx, batch in enumerate(train_data):
                    tokenized = self.tokenizer(batch['text'], padding=True, return_tensors='pt', max_length=256, truncation=True)['input_ids'].to(device)
                    logits, loss = model(tokenized, tokenized)
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
            'Accuracy': 0,
            'Loss': 0,
            'test_total': 0
        }
        # predictions = []
        # references = []
        nlls = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                tokenized = self.tokenizer(batch['text'], padding=True, return_tensors='pt', max_length = 256, truncation = True)['input_ids'].to(device)
                labels = tokenized[..., 1:].cpu()
                logits, loss = model(tokenized, tokenized)
                pred_ids = torch.argmax(logits, dim=-1)[..., :-1].cpu()
                
                pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
                # pred_ids = np.where(labels != pad_token_id, pred_ids, pad_token_id)
                # preds = self.tokenizer.batch_decode( pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True )

                # predictions += preds
                # references += batch['text']

                # logging.info(tokenized[0])
                # logging.info(pred_ids[0])

                metrics['Loss'] += loss.item() * len(batch['text'])
                metrics['test_total'] += len(batch['text'])
                
                for i in range(len(labels)):
                    hit, total = 0, 0 
                    nlls.append([])
                    for j in range(len(labels[i])):
                        if labels[i][j] != pad_token_id:
                            poss = torch.nn.functional.softmax(logits[i][j])
                            logit = poss[labels[i][j]].item()
                            nlls[-1].append(logit)
                            
                            total += 1
                            if labels[i][j] == pred_ids[i][j]:
                                hit += 1

                    metrics['Accuracy'] += (hit / total)
                    ppl = np.exp(-np.sum(np.log(nlls[-1])) /len(nlls[-1]))

                    # # debug inf problem
                    # if np.isinf(ppl):
                    #     print(nlls[-1])

                    nlls[-1] = ppl
                

        # rouge_results = self.rouge.compute(references=references, predictions=predictions)
        # metrics.update(rouge_results)
        metrics["Perplexity"] = np.mean(nlls)
        metrics['Loss'] /= metrics['test_total']
        metrics['Accuracy'] /= metrics['test_total']
        
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
    