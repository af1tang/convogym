#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:22:53 2020

@author: af1tang
"""
import torch, os, pickle, random, numpy as np
import torch.nn as nn, torch.nn.functional as F
from load_configs import *
from utils import *

#### Baseline UserSimulator ####
class Agent(object):
    def __init__(self, personas, reverse = False,
                 top_k=10, top_p = .92, max_length=1024):
        self.top_k, self.top_p, self.max_length = top_k, top_p, max_length
        self.reversed = reverse
        if not self.reversed:
            self.p1 = [] # self.NA_token # don't know your partner's persona
            self.p2 = personas
        else:
            self.p2 = [] #self.NA_token
            self.p1 = personas
        self.reset_convo()
        
    def __call__(self, inp, state=None, context=None, reward=None, act=False):
        return self.step(inp, act)
        
    def reset_convo(self):
        # reset dialog history
        self.dialog_history, self.turn = [], 0
        
    def _update_persona(self, action):
        if action not in action_space:
            raise NotImplementedError("this action is not currently in the set of learned action space")
        else:
            if self.reversed:
                self.p2 = [action]
            else:
                self.p1 = [action]
    
    def _reset_inp(self, act = False):
        # action vs. persona code
        if not act:
            if self.reversed:
                self.inp = tokenizer.encode(''.join(['<|p1|>'] + self.p1 + ['<|sep|>'] + ['<|start|>']))
            else:
                self.inp = tokenizer.encode(''.join(['<|p2|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
        else:
            if self.reversed:
                self.inp =  tokenizer.encode(''.join(['<|act|> '] + self.p2 + ['<|sep|>'] + ['<|p1|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
            else:
                self.inp =  tokenizer.encode(''.join(['<|act|> '] + self.p1 + ['<|sep|>'] + ['<|p1|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
        # incorporate dialog dx
        self.inp += flatten(self.dialog_history)
        self.inp, self.curr_len, self.past = to_var(torch.tensor([self.inp])), len(self.inp), None

    def _update_dialog_hx(self, new_inp):
        if new_inp is not None:
            self.dialog_history.append(new_inp)
        
    def step(self, inp, act=False):
        self._update_dialog_hx(inp)
        self._reset_inp(act)
        outp = []
        with torch.no_grad():
            while (tokenizer.eos_token_id not in outp) and (self.curr_len + len(outp) < self.max_length):
                logits, self.past = model(self.inp, past=self.past)
                # top k sampling          
                log_scores = top_k_top_p_filtering(logits[:,-1,:], top_k=self.top_k, top_p=self.top_p)
                probs = F.softmax(log_scores, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(1)
                # update tokens for next output
                outp += token.tolist()
                self.inp = token.unsqueeze(0) 
                self.curr_len+=1
        self.dialog_history.append(outp)
        self.turn+=1
        return outp
    