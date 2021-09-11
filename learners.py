#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:51:20 2021

@author: af1tang
"""
import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from _tokenizer import act_tok, start_tok, p1_tok, p2_tok
from utils._device import to_device
from utils._reshape import flatten
from utils._optim_helpers import fit_on_batch
from utils._visualization import plot_losses

def _get_grouped_params(model):
    with torch.no_grad():
        fast_group = flatten([[p[act_tok], p[start_tok], p[p1_tok], p[p2_tok]] for n,p in model.named_parameters() if n == 'transformer.wte.weight']) #['transformer.wte.weight']
        freeze_group = [p[:start_tok] for n,p in model.named_parameters() if n == 'transformer.wte.weight']#['transformer.wte.weight']
        slow_group = [p for n,p in model.named_parameters() if n == 'transformer.wpe.weight']
        normal_group = [p for n,p in model.named_parameters() if n not in ('transformer.wte.weight',
                                                                           'transformer.wpe.weight')]
    
    optimizer_grouped_parameters = [{"params": fast_group, 'lr': 6.25e-4}, 
                                    {"params": freeze_group, 'lr': 1e-6}, 
                                    {"params": slow_group, 'lr': 1e-5}, 
                                    {"params": normal_group, 'lr': 5e-5}]
    return optimizer_grouped_parameters

class Learner(nn.Module):
    """
    Training wrapper for the decoder model. Updates the model using:  
        - active learning batch
        - supervised learning batch using given training_data.

    Parameters
    ----------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT model card.
        
    training_data : torch.utils.data.Dataset or List of tuples 
        List of training batches to sample batches from. Used to re-fit model to prevent catastrophic forgetting.
        
    lr : float, optional
        Learning rate for model parameters. The default is 5e-5.
        
    use_param_groups : bool, optional
        Whether to use different learn rates for special tokens, positional tokens and normal tokens.
        The default is True.
        
    schedule_func : torch.optim.lr_scheduler object, optional
        Learn rate scheduler function. The default is get_linear_schedule_with_warmup.
        
    gradient_accumulation_steps : int, optional
        Number of gradient accumulation steps. The default is 8.
        
    max_grad_norm : float, optional
        Max gradient norm size for gradient clipping. The default is 1.0.
        
    optim_func : torch.optim object, optional
        Optimizer used to update parameters. The default is AdamW.
        
    total_iters : int, optional
        Max number of training iters. The default is 20000.

    Attributes
    -------
    steps : int
        Tracks how many global update steps already taken during active learning.
    
    dataloader : torch.utils.data.DataLoader object
        Stores training_data as DataLoader object to generate batch samples.

    """
    def __init__(self, model, training_data, lr=5e-5, use_param_groups=True,
                 schedule_func=None,
                 gradient_accumulation_steps=8, max_grad_norm=1.0,
                 optim_func=None, total_iters=20000):
        if use_param_groups:
            optimizer_grouped_parameters = _get_grouped_params(model)
        else:
            optimizer_grouped_parameters = model.parameters()
        self.model = model
        if not optim_func:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        if not schedule_func:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                num_warmup_steps=int(.2 * total_iters), 
                                                num_training_steps=total_iters)
        self.dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
        self.data_iter = iter(self.dataloader)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.steps = 0
    
    def _reset_iterator(self):
        """
        Resets data iterator object using self.dataloader.
        
        Attributes
        -------
        data_iter : torch.utils.data.dataloader._SingleProcessDataLoaderIter
            Generator object to yield the next batch sample.

        Returns
        -------
        None.

        """
        self.data_iter = iter(self.dataloader)
        
    def fit_on_active_batch(self, active_batch):
        """
        Updates model paramters based on input active learning batch.

        Parameters
        ----------
        active_batch : tuple
            Format of tuple is (x,y): 
                - X: List of int
                    Input tokens (encoded by tokenizer) of action prefix + dialog history.
                - Y: List of int
                    Encoded human-provided response (ground truth).
        Returns
        -------
        None.

        """
        print(); print("Fitting on batch ... ")
        self.model.train()
        new_loss = fit_on_batch(self.model, active_batch, self.gradient_accumulation_steps)
        try:
            old_batch = self.data_iter.next()
        except:
            self._reset_iterator()
            old_batch = self.data_iter.next()
        old_loss = fit_on_batch(self.model, old_batch, self.gradient_accumulation_steps)
        # step
        if (self.steps+1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            
            print('Iter: %d | new_loss: %.3f | old_loss: %.3f | lr: %s ' %( 
                    self.steps, new_loss.item() * self.gradient_accumulation_steps, 
                    old_loss.item() * self.gradient_accumulation_steps,
                    str(self.scheduler.get_last_lr()[0])) )
        self.steps += 1
        self.model.eval()
    
    def fit(self, num_train_epochs=3, logging_steps=10, save_steps=250):
        """
        Trains self.model on the provided dataset (self.dataloader). 
        

        Parameters
        ----------
        num_train_epochs : int, optional
            Total number of epochs . The default is 3.
            
        logging_steps : int, optional
            Training progress is printed every logging_steps. The default is 10.
            
        save_steps : int, optional
            Training loss and learn rate plots are saved every save_steps. The default is 250.

        Returns
        -------
        stats : dict
            Dictionary of training performance.
                - keys: global steps (self.steps)
                - values: pretrain loss and learn rate.

        """
        stats = {}        
        tr_loss, logging_loss = 0.0, 0.0
        # model to training mode
        self.model.zero_grad(); self.model.train()
        start_time = time.time()
        
        t_total = len(self.dataloader) // self.gradient_accumulation_steps * num_train_epochs
        for epoch in range(num_train_epochs):
            self._reset_iterator()
            for step in range(len(self.dataloader)):
                ### step ###
                batch = self.data_iter.next()
                loss = fit_on_batch(batch); del batch
                # logging (new data only)
                tr_loss += loss.item()
                
                # gradient accumulation
                if (step+1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    self.steps += 1
                    
                    # reporting 
                    if (self.steps+1) % logging_steps ==0:
                        stats[self.steps] = {'pretrain_loss': (tr_loss - logging_loss) / logging_steps, 
                                              'pretrain_lr': self.scheduler.get_last_lr()[-1]}
                        logging_loss = tr_loss
                        
                        elapsed_time = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
                        print('Epoch: %d | Iter: [%d/%d] | loss: %.3f | lr: %s | time: %s' %( 
                        epoch, self.steps, t_total, stats[self.steps]['pretrain_loss'],                             
                                str(stats[self.steps]['pretrain_lr']), elapsed_time))
                        start_time = time.time()
                        
                    if (self.step + 1) % save_steps==0:
                        print("Plotting training loss and lr ... ")
                        plot_losses(stats, title='pretrain_loss' )
                        plot_losses(stats, title='pretrain_lr')
                        print("Done.")
                        
        return stats
        
    def save(self, save_path):
        """
        Saves model parameters and config to save_path.
        """
        self.model.save_pretrained(save_path)

    def load(self, load_path):
        """
        Tries to load model parameters from load_path, raises exception if fails.
        """
        try: 
            self.model.from_pretrained(load_path)
        except Exception as e:
            raise e