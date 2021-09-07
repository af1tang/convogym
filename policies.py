#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:07:07 2021

@author: af1tang
"""
import warnings
import random, math
import torch, torch.nn as nn, torch.nn.functional as F
from utils._device import to_device

default_policy = nn.Sequential(nn.Linear(2050, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, 11))

class Policy(nn.Module):
    def __init__(self, policy=default_policy, lr=1e-4, 
                 gamma=.95, EPS_START=.5, EPS_END=.05, EPS_DECAY=2048):
        super(Policy, self).__init__()
        self.policy = to_device(policy)
        # loss func
        self.huber_loss = nn.SmoothL1Loss()
        # epsilon sampling params
        self.EPS_START, self.EPS_END, self.EPS_DECAY, self.gamma = EPS_START, EPS_END, EPS_DECAY, gamma
        self.reset_stats()
        
    def reset_stats(self):
        # eps greedy params
        self.global_step = 0
        # logging
        self.stats, self.global_epoch = {'losses': {}}, 0
    
    def act(self, inp, turn):
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(inp)
            p = F.softmax(logits,-1)
            # sampling w/ eps-greedy
            eps = random.random()
            threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.global_step / self.EPS_DECAY)
            self.global_step +=1
            if eps > threshold:
                if turn == 0:
                    action = torch.multinomial(p, num_samples=1).item()
                else:
                    action = p.max(1)[1].item()
            else:
                action = random.randrange(11)
        return action
    
    def update_policy(self, *args, **kwargs):
        warnings.warn("update_policy() is called but not implemented.")
    
    def load(self, load_path):
        try: 
            state_dict = torch.load(load_path)
            self.policy.load_state_dict(state_dict)
        except Exception as e:
            print(e)
    
    def save(self, save_path):
        torch.save(self.policy.state_dict(), save_path)
