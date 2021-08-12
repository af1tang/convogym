#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 20:25:09 2021

@author: af1tang
"""
import torch, os, pickle, random, numpy as np, random
import torch.nn as nn, torch.nn.functional as F
from load_configs import *
from utils import *
from agents import Agent

## interaction funcs ##
def get_custom_persona():
    personas = []
    for i in range(opts.num_personas):
        response = ""
        while len(response) <1:
            response = input(">> Fact %d: "%(i+1))+ tokenizer.eos_token
        personas.append(response)
    return personas

def get_random_train_persona():
    return random.sample(train_personas, 1)[0]
        
def get_random_test_persona():
    return random.sample(test_personas,1)[0]


## callbacks ##
class StateCb:
    def __init__(self, state=None, context=None, action=None, reward=None):
        self.state, self.context, self.action, self.reward = state, context, action, reward

class MessageCb:
    def __init__(self, msg=None, x=None, y=None):
        self.msg, self.x, self.y, self.dialog_hx = msg, x, y, []

## gym environments ##
class Gym:
    '''default gym environment'''
    def __init__(self, user=None, interactive = True, num_convos = 1, 
                 reset_persona_func=get_custom_persona,
                 length=8, top_k=10, top_p = .92, max_length=1024 ):
        self.user = user
        self.interactive = interactive
        self.num_convos = num_convos
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        self.reset_persona_func = reset_persona_func
        
    def _reset_env(self):
        self.turn, self.done = 0, False
        self.data = {'hx': [], 'p1': [], 'p2': []}
    
    def _reset_agents(self, persona_input):
        '''reset agent conversational states'''
        if persona_input:
            self.agent = Agent(persona_input, top_k=self.top_k,
                               top_p=self.top_p, max_length=self.max_length)
        else:
            persona = self.reset_persona_func()
            self.agent = Agent(persona, reverse=False, top_k=self.top_k, 
                              top_p=self.top_p, max_length=self.max_length)
            
        if self.user:
            # policy
            self.user.reset_convo()
        elif self.interactive:
            # human input
            self.user = self._interact
        else:
            # another persona model
            persona = self.reset_persona_func()
            self.user = Agent(persona, reverse=True, top_k=self.top_k, 
                              top_p=self.top_p, max_length=self.max_length)
        
    def _interact(self, msg):
        if msg:
            print("Bot: {}".format(tokenizer.decode(msg, skip_special_tokens=True)))
        msg = tokenizer.encode(input(">> User: ") + tokenizer.eos_token)
        return msg     
    
    def _on_convo_begin(self):
        self.turn = 0
        self.done = False
        
    def _on_user_begin(self, scb, mcb):
        return scb, mcb
    
    def _on_user_end(self, scb,mcb):
        mcb.dialog_hx.append(mcb.msg)
        return scb, mcb
    
    def _on_agent_end(self, scb, mcb):
        mcb.dialog_hx.append(mcb.msg)
        self.turn +=1
        if self.turn >= self.length:
            self.done = True
        return scb, mcb
    
    def _on_convo_end(self, scb, mcb):
        print('p1: ', self.user.p1)
        print('p2: ', self.agent.p2)
        display_dialog_history(mcb.dialog_hx)
        self.data['hx'].append(mcb.dialog_hx)
        if self.interactive:
            self.data['p1'].append([])
        else:
            self.data['p1'].append(self.user.p1)
        self.data['p2'].append(self.agent.p2)
        del scb, mcb

    def _generate_trajectory(self, persona_input = None): 
        '''generate a trajectory of 
            - dialog history
            - reward associated w/ each turn  '''
        self._reset_agents(persona_input)
        scb, mcb = StateCb(), MessageCb()
        model.eval()
        self._on_convo_begin()
        # run for turns, updating rewards at each turn
        while not self.done:
            # person 1 (user) 
            scb,mcb = self._on_user_begin(scb,mcb)             
            mcb.msg = self.user(mcb.msg)
            scb,mcb = self._on_user_end(scb,mcb)

            # person 2 (agent)
            mcb.msg = self.agent(mcb.msg)
            scb, mcb = self._on_agent_end(scb,mcb)
        self._on_convo_end(scb, mcb) 
    
    def simulate_convos(self):
        self._reset_env()
        print("Conducting conversations ...")
        print("="*50)
        for i in range(self.num_convos):
            self._generate_trajectory(None)
 
    def run_evaluation(self):
        raise NotImplementedError("evaluation loop not implemented for current environment.")

# RL environment 
class RLGym(Gym):
    def __init__(self, user, reward_func, state_func,
                 length=8, top_k=10, top_p = .92, max_length=1024):
        super().__init__(user, interactive=False, num_convos=len(train_personas) * opts.num_epochs, 
                         reset_persona_func=get_random_train_persona,
                         length=8, top_k=10, top_p = .92, max_length=1024)