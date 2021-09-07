#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 01:47:23 2021

@author: af1tang
"""
import random, math, os, pickle
import numpy as np
from functools import partial

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils._swa_utils import AveragedModel
from utils._device import to_data, to_var, to_device
from utils._visualization import plot_losses
from utils._reshape import flatten

from environments import Env
from policies import Policy, default_policy
from loggers import Logger
from rewards import Reward
from gyms import RLGym

from models import ID, Identifier

from _configs import opts
from _decoder import model
from _tokenizer import tokenizer
from _prepare_persona_data import prepare_persona_dataset

## batching utils ##
def _process_dqn_batch(batch, policy, swa_policy, gamma):
    try:
        x, c, y, x_next, c_next, y_next, r = batch
        y_next = to_var(y_next)

    except:
        x, c, y, x_next, c_next, r = batch
    x,c = torch.stack(x, dim=-1).type(torch.cuda.FloatTensor), torch.stack(c, dim=-1).type(torch.cuda.FloatTensor)
    x_next, c_next = torch.stack(x_next, dim=-1).type(torch.cuda.FloatTensor), torch.stack(c_next, dim=-1).type(torch.cuda.FloatTensor)
    y, r = to_var(y), r.type(torch.cuda.FloatTensor)
    xx = torch.cat((x,c), dim=-1)
    xx_next = torch.cat((x_next, c_next), dim=-1)
    # calculate q-values
    with torch.no_grad():
        # use target network to predict q-targets
        q_values = policy(xx_next)
        idx = q_values.max(1)[1]
        q_values = swa_policy(xx_next)
        dones = (x[:,-1] >= 7).long()
        q_targets = r + (1-dones) * gamma * q_values[torch.arange(len(idx)), idx]
    return xx, y, q_targets

## state estimation ##
def _get_state(identifier, dialog_history):
    # hx x action -> state
    with torch.no_grad():
        state_conv = model(to_var(flatten(dialog_history)).long(), 
                     output_hidden_states=True)[2][24].squeeze(0).mean(0)
        # predicted personality -> 1024 clf state
        state_clf = identifier.clf(state_conv.unsqueeze(0))[0].squeeze(0)
    return state_conv, state_clf

## reward calculation ##
def _calculate_reward(identifier,state, context, personas, k=20):
    with torch.no_grad():
        pos, neg = identifier._generate_candidates([personas], k)
        loss = identifier.criterion(state.unsqueeze(0), pos[0], neg[0])
        return - loss.item()
    
## eval metrics ##
def _evaluation_func(identifier, scb, k=20):
    prec1, prec5, rec5, rec10 = identifier.score(scb.context.view(1,-1), 
                                                    [scb.personas], candidates=k)
    return prec1, prec5, rec5, rec10
    

## logger module ##
class IRLogger(Logger):
    def __init__(self, eval_func = _evaluation_func, candidates=20):
        self.evaluate = eval_func
        self.k = candidates
        self.metrics = {'prec1s': [], 'prec5s': [],
                        'rec5s': [], 'rec10s': [], 'rewards': []}
        
    def __call__(self, scb, mcb):
        prec1, prec5, rec5, rec10 = self.evaluate(scb=scb, k=self.k)
        # log data
        self.metrics['prec1s'].extend(prec1)
        self.metrics['prec5s'].extend(prec5)
        self.metrics['rec5s'].extend(rec5)
        self.metrics['rec10s'].extend(rec10)
        self.metrics['rewards'].append(scb.rewards)
        print("prec@1: %.2f | prec@5: %.2f | rec@5: %.2f | rec@10: %.2f" % (prec1[0], prec5[0], rec5[0], rec10[0]))
        print("avg p@1: %.2f | avg p@5: %.2f | avg r@5: %.2f | avg. r@10: %.2f" % (np.mean(self.metrics['prec1s']),
                                                                                   np.mean(self.metrics['prec5s']), 
                                                                                    np.mean(self.metrics['rec5s']), 
                                                                                    np.mean(self.metrics['rec10s'])))
        print()
## policy module ##
class Verifier(Policy):
    def __init__(self, policy=default_policy, lr=1e-4, t_total=20000, gamma=.95,
                 EPS_START=.5, EPS_END=.05, EPS_DECAY=2048):
        super().__init__(lr=lr, gamma = gamma,
                                     EPS_START=EPS_START, 
                                     EPS_END=EPS_END, 
                                     EPS_DECAY=EPS_DECAY)
        self.policy = policy
        # loss func
        self.huber_loss = nn.SmoothL1Loss()
        # optim and sched
        self.optimizer = AdamW(self.policy.parameters(), lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, 
                                                num_training_steps=t_total)
        # swa tools
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                0.99 * averaged_model_parameter + 0.01 * model_parameter
        self.swa_policy = AveragedModel(self.policy, avg_fn = ema_avg)
        self.swa_policy.eval()
        # hyperparams
        self.EPS_START, self.EPS_END, self.EPS_DECAY, self.gamma = EPS_START, EPS_END, EPS_DECAY, gamma
        self.reset_stats()
    
    def update_policy(self, memory_buffer, num_train_epochs=1):
        self.policy.train()
        for epoch in range(num_train_epochs):
            print()
            dataloader = DataLoader(memory_buffer, batch_size=64, shuffle=True)
            epoch_loss = []
            for i, batch in enumerate(dataloader):#dataloader_old):
                # get batch on buffer trajectories 
                xx, y, q_targets = _process_dqn_batch(batch, self.policy, self.swa_policy, self.gamma)
                # forward
                yhat = self.policy(xx)   # logits
                loss = self.huber_loss(yhat[torch.arange(len(y)),y], q_targets)      
                # backward
                self.policy.zero_grad()
                loss.backward()
                self.optimizer.step()
                # take swa (polyak avg)
                self.swa_policy.update_parameters(self.policy)
                # tracking
                epoch_loss.append(loss.item())
            print("epoch: %d | loss: %.2f | lr: %s" %(epoch, np.mean(epoch_loss), str(self.scheduler.get_last_lr()[-1])))
            # on epoch end
            self.scheduler.step()
            self.stats['losses'][self.global_epoch] = {'dqn_loss': np.mean(epoch_loss)}
            plot_losses(self.stats['losses'], title='dqn_loss' )
            self.global_epoch +=1
            
if __name__ == '__main__':
    # prepare persona vector dictionary
    try:
        with open(os.path.join(opts.example_path, 'p2v'), 'rb') as f:
            p2v = pickle.load(f)
        with open(os.path.join(opts.example_path, 'vec_train_data'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(opts.example_path, 'vec_test_data'), 'rb') as f:
            test_data = pickle.load(f)
    except:
        p2v, train_data, test_data = prepare_persona_dataset(model, tokenizer, 
                                 data_path = os.path.join(opts.data_path, 'personachat.csv'))
    # init identifier network
    id_net = ID()
    identifier = Identifier(id_net, p2v)
    # policy 
    policy = Verifier()
    # environment
    env = Env(state_estimator=partial(_get_state, identifier=identifier))
    logger = IRLogger(eval_func=partial(_evaluation_func, identifier=identifier))
    r_func = Reward(reward_func=partial(_calculate_reward, identifier=identifier))
    # train identifier if not load params 
    if not identifier.load_params(os.path.join(opts.example_path, 'ID.pt')):
        identifier.fit(train_data, 
                       save_path = os.path.join(opts.example_path, 'ID.pt'),
                       epochs=10, lr=1e-3,
                       bs=64, k=20, save_steps=100, logging_steps=10)
        eval_stats = identifier.evaluate(test_data,
                                         bs=64, k=20)
    # gym 
    gym = RLGym(model, policy, env, logger, r_func, length=8)
    try:
        gym.policy.load(os.path.join(opts.example_path, 'policy.pt'))
    except:
        print("Training policy ... ")
        # train
        gym.sim_convos(training=True)
        gym.policy.save(os.path.join(opts.example_path, 'policy.pt'))
    # test
    print("Evaluating policy ... ")
    gym.sim_convos(training=False)