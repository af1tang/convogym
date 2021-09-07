#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:22:53 2020

@author: af1tang
"""
import torch, torch.nn.functional as F
from _tokenizer import tokenizer
from _configs import action_space
from utils._reshape import flatten
from utils._device import to_var
from utils._sampling import top_k_top_p_filtering

#### Baseline Agent ####
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
        
    def __call__(self, model, inp, act=False):
        return self.step(model, inp, act)
        
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
        """
        Reformats the input to the agent. 
        
        Args: 
            act: 
                (boolean) Whether the input prefix is an action code (True)
                            or a set of persona facts (False). 
                            Default = False.
        """
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
        
    def step(self, model, inp, act=False):
        ''' Generates a response message based on input message.
        Args: 
            model: 
                The decoder model e.g., a transformer decoder (GPT).
            inp: 
                The input message, does NOT include history of dialog.
        Returns:
            The output message as an array of integers, each corresponding to a token.
        '''
        self._update_dialog_hx(inp)
        self._reset_inp(act)
        outp = []
        model.eval()
        with torch.no_grad():
            while (tokenizer.eos_token_id not in outp) and (self.curr_len + len(outp) < self.max_length):
                outputs = model(self.inp, past_key_values=self.past)
                logits, self.past = outputs.logits, outputs.past_key_values
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
    