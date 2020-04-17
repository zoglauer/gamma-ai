"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time

# Externals
import torch
from torch import nn

# Locals
from .base_trainer import BaseTrainer
from models import get_model

class GNNTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""

    def __init__(self, **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)

    def build_model(self, model_type='gnn_segment_classifier',
                    optimizer='Adam', learning_rate=0.001,
                    loss_func='BCELoss', **model_args):
        """Instantiate our model"""
        self.model = get_model(name=model_type, **model_args)
        if self.distributed:
            print("Using", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
            print("Parallelized Data")
        self.model.to(self.device)
        print("Ported Model to Device")
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=learning_rate)
        self.loss_func = getattr(torch.nn, loss_func)()
        print("Finished Building Model")
    
    #Each model consists of three networks, so might have to restore them one by one
    def save_model(self, model_path='saved_model_state.pt'):
        torch.save(self.model.state_dict(), model_path)
    
    def restore_model(self, model_path='saved_model_state.pt'):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        summary = dict()
        sum_loss = 0
        start_time = time.time()
        i_final = 0
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            self.logger.debug('  batch %i', i)
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            print('Batch ' + str(i) + ' Loss: ' + str(batch_loss.item()))
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            i_final = i
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i_final + 1)
        self.logger.debug(' Processed %i batches' % (i_final + 1))
        self.logger.info('  Training loss: %.3f' % summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        start_time = time.time()
        # Loop over batches
        i_final = 0
        X, Ri, Ro, Edge_Labels = [], [], [], []
        for i, (batch_input, batch_target) in enumerate(data_loader):
            self.logger.debug(' batch %i', i)
            X += batch_input[0]
            Ri += batch_input[1]
            Ro += batch_input[2]
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            Edge_Labels += (batch_output > 0.5).item()
            sum_loss += self.loss_func(batch_output, batch_target).item()
            # Count number of correct predictions
            matches = ((batch_output > 0.5) == (batch_target > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            i_final = i
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i_final + 1)
        summary['valid_acc'] = sum_correct / (sum_total + 1e-10)
        summary['X'] = X
        summary['Ri'] = Ri
        summary['Ro'] = Ro
        summary['Edge_Labels'] = Edge_Labels
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i_final + 1)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
