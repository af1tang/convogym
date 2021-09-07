#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:00:29 2021

@author: af1tang
"""
import torch
from utils._device import to_var

## environment helpers 
default_state_estimator = lambda dialog_history: (to_var(torch.zeros(1024)), to_var(torch.zeros(1024)))

class Env:        
    def __init__(self, state_estimator=default_state_estimator):
        self._get_state = state_estimator
        
    def reset(self, scb):
        scb.state, scb.context = to_var(torch.zeros(1024)), to_var(torch.zeros(1024))
        scb.states, scb.contexts, scb.rewards, scb.actions = [scb.state.tolist() + [0.0]], [scb.context.tolist() + [0.0]], [], []
        return scb
    
    def get_policy_inp(self, scb):
        turn_tensor = to_var(torch.ones(1,1)) * scb.turn
        state, context = scb.state.view(1,1024), scb.context.view(1,1024)
        state_t = torch.cat((state, turn_tensor), dim=-1)
        context_t = torch.cat((context, turn_tensor), dim=-1)
        return torch.cat((state_t, context_t), dim=-1)
    
    def get_curr_state(self, scb, mcb):
        scb.state, scb.context = self._get_state(dialog_history=mcb.dialog_hx[1::2])
        return scb, mcb
        
    def get_next_state(self, scb, mcb):
        _state, _context = scb.state.tolist(), scb.context.tolist()
        _state.append(scb.turn+1.0); _context.append(scb.turn+1.0)
        scb.states.append(_state); scb.contexts.append(_context)
        return scb, mcb