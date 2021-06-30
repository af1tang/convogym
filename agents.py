#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:22:53 2020

@author: af1tang
"""
import torch, os, pickle, random, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from load_configs import *
from utils import *
from models import *

#### Baseline UserSimulator ####
class UserSim(object):
    def __init__(self, i2p, persona_ids, reverse = False,
                 top_k=10, top_p = .92, max_length=1024):
        self.i2p = i2p
        self.top_k, self.top_p, self.max_length = top_k, top_p, max_length
        self.persona_ids, self.reversed = persona_ids, reverse
        self.NA_token = ['Not Available.' + tokenizer.eos_token]
        if not self.reversed:
            self.p1 = [] # self.NA_token # don't know your partner's persona
            if persona_ids is not None:
                self.p2 = [self.i2p[pi] for pi in self.persona_ids] # given persona
            else:
                self.p2 = []
        else:
            self.p2 = [] #self.NA_token
            if persona_ids is not None:
                self.p1 = [self.i2p[pi] for pi in self.persona_ids] # given persona
            else:
                self.p1 = []
        self.reset_convo()
        
    def __call__(self, inp, state=None, context=None, reward=None, act=False):
        return self.step(inp, act)
        
    def reset_convo(self):
        # reset personas, pasts and dialog history
        self.p1_descr, self.p2_descr, self.sep_tok = ['<|p1|>']+ ['person 1: '], ['<|p2|>']+ ['person 2: '],  [tokenizer.sep_token]
        self.act_descr = ['<|act|>'] + ['action 1: ']
        #self.inp = tokenizer.encode(''.join(self.p1_descr + self.p1 + self.sep_tok + self.p2_descr + self.p2 +self.sep_tok + ['<|start|>']), return_tensors='pt').to(device)
        # use PAST (src and trg) to keep track of (a) gradients, (b) history for separate decoding
        #self.past, self.curr_len = None, self.inp.size(1)
        self.dialog_history, self.turn = [], 0
        
    def _update_persona(self, action):
        raise NotImplementedError("Update agent not specified for this user.")
        
    def _on_episode_end(self, last_msg, last_reward):
        self._update_dialog_hx(last_msg)
        # episodic memory
        self.episode = {'states':[], 'actions': [], 'log_probs':[], 'rewards': [], 
                        'returns': [], 'contexts': [], 'logits': [] }
        
    
    def _reset_inp(self, act = False):
        # action vs. persona code
        if not act:
            if self.reversed:
                self.inp = tokenizer.encode(''.join(['<|p1|>'] + self.p1 + ['<|sep|>'] + ['<|start|>']))
            else:
                self.inp = tokenizer.encode(''.join(['<|p2|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
            #self.inp = tokenizer.encode(''.join(self.p1_descr + self.p1 + self.sep_tok + self.p2_descr + self.p2 +self.sep_tok + ['<|start|>']))
        else:
            self.inp =  tokenizer.encode(''.join(['<|act|>'] + self.p1 + ['<|sep|>'] + ['<|p1|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
            #self.inp = tokenizer.encode(''.join(self.act_descr + self.p1 + self.sep_tok + self.p2_descr + self.p2 + self.sep_tok + ['<|start|>']))
        # token_type_ids usage    
        if opts.use_token_ids:
            ctx_ids = torch.full_like(torch.tensor(self.inp), 0)
            ctx_ids[-1] = 1
            if len(self.dialog_history) > 0:
                self.token_types = torch.cat( ( ctx_ids, torch.full_like(torch.tensor(flatten(self.dialog_history)), 1) ), -1).to(device)
            else:
                self.token_types = ctx_ids.to(device)
        else: self.token_types = None
        # incorporate dialog dx
        self.inp += flatten(self.dialog_history)
        self.inp, self.curr_len, self.past = to_var(torch.tensor([self.inp])), len(self.inp), None

    def _update_dialog_hx(self, new_inp):
        if new_inp is not None:
            self.dialog_history.append(new_inp)
        
    def step(self, inp, act=False):
        self._update_dialog_hx(inp)
        #self._update_persona(action) # do on agent side
        self._reset_inp(act)
        outp = []
        with torch.no_grad():
            while (tokenizer.eos_token_id not in outp) and (self.curr_len + len(outp) < self.max_length):
                logits, self.past = model(self.inp, past=self.past, token_type_ids = self.token_types)
                # top k sampling          
                log_scores = top_k_top_p_filtering(logits[:,-1,:], top_k=self.top_k, top_p=self.top_p)
                probs = F.softmax(log_scores, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(1)
                # update tokens for next output
                outp += token.tolist()
                self.inp = token.unsqueeze(0) 
                self.token_types = torch.full_like(self.inp, 1).to(device) if opts.use_token_ids else None
                self.curr_len+=1
        self.dialog_history.append(outp)
        self.turn+=1
        return outp
    
    def update(self, *args, **kwargs):
        pass

#### Agents ####
''' Each inherits from UserSim class to generate w/ GPT-2.
     Each has its own update/fit, _on_episode_end, 
                     _calculate_returns, save/load methods. '''
class MetaAgent(UserSim):
    def __init__(self, i2p, persona_ids=[], 
                 inp_size=2048, hidden_size=512, dropout=.2, 
                 top_k=10, top_p = .92, max_length=1024):
        super(MetaAgent, self).__init__(i2p, persona_ids)
        self.i2p = dict([(k,v + tokenizer.eos_token) for k,v in i2p.items()])
        self.meta_policy = MetaNet(inp_size, len(self.i2p), hidden_size, dropout).to(device)
        self.optimizer = torch.optim.Adam(self.meta_policy.parameters(), lr=1e-2)

    def _update_persona(self, action):
        if action is not None:
            #self.persona_ids.append(action)
            self.persona_ids = [action]
            self.p1 = [self.i2p[pi] for pi in self.persona_ids]
        
    def _reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.meta_policy.parameters(), lr=1e-2)
        
    def reset_convo(self):
        # reset personas, pasts and dialog history
        self.act_descr, self.p2_descr, self.sep_tok = ['<|act|>'] + ['action 1: '], ['<|p2|>']+ ['person 2: '],  [tokenizer.sep_token]        
        self.dialog_history, self.turn = [], 0
        self.persona_ids = []
        # episodic memory
        self.episode = {'states':[], 'actions': [], 'log_probs':[], 'rewards': [], 
                        'returns': [], 'contexts': [], 'logits': [] }
        
    def _calculate_returns(self, rewards):
        '''R_t = r @ final timestep '''
        returns = np.ones_like(rewards) * rewards[-1]
        return returns        
        
    def _on_episode_end(self, last_msg, last_reward):
        self._update_dialog_hx(last_msg)
        #self.episode['states'].append(last_state)
        #self.episode['contexts'].append(last_context)
        self.episode['rewards'].append(last_reward)
        # calculate returns
        self.episode['returns'] = self._calculate_returns(self.episode['rewards'])
        
    def __call__(self, inp, state=None, context=None, reward=None):
        # update state representation
        if state is None: 
            #TODO: better state initialization?
            state = to_var(torch.zeros((1,self.inp_size)))
            assert context is None
            context = to_var(torch.zeros((1, self.inp_size)))
        else: 
            if len(state.shape) < 2: state = state.unsqueeze(0)
            if len(context.shape ) <2: context = context.unsqueeze(0)
            assert len(state.shape) == len(context.shape) == 2
            state = to_var(state); context = to_var(context)
        # track reward if not first step
        if reward is not None:
            self.episode['rewards'].append(reward)
            
        # act and update episodic memory
        sampled_action = self.policy.act(state, context, self.episode)
        # update persona and action history
        self._update_persona(sampled_action)
        return self.step(inp, act=True)        
        
    def update(self, *args, **kwargs):
        raise NotImplementedError("update is NOT implemented for meta-agent!")
        
    def fit(self, data):
        dataloader = DataLoader(data, batch_size = min(opts.meta_bs, len(data)) )
        i=0; print(); print("-"*10, "Meta training (%d samples, central policy) " %(len(data)), "-"*10 )
        for _ in range(opts.meta_policy_iters):
            total_loss = []
            for batch in dataloader:
                states, contexts, logits = batch['states'], batch['contexts'], batch['logits']
                states, contexts, logits = map(to_var, (states, contexts, logits))
                
                ### loss ###
                meta_logits = self.meta_policy(states, contexts)
                loss = F.smooth_l1_loss(meta_logits, logits)

                # backprop
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
                # logging
                total_loss.append(loss.sum().item()); i +=1;
            # report epoch score    
            if (_+1)%5 ==0:    
                print("   epoch %d, loss = %.2f"% (_, np.mean(total_loss)) )
        self.save()
        print(" Done. "); print("-"*50)
                          
    def save(self):
        torch.save(self.meta_policy.state_dict(), os.path.join(opts.authenticator_path,'meta_policy.pt'))
        
    def load(self):
        policy_state_dict = torch.load(os.path.join(opts.authenticator_path, 'meta_policy.pt'))
        self.meta_policy.load_state_dict(policy_state_dict)

class Agent(UserSim):
    def __init__(self, i2p, persona_ids=[], reverse = True,
                 inp_size=1024, hidden_size=512, dropout=.1,
                 top_k=10, top_p = .92, max_length=1024):
        super(Agent, self).__init__(i2p, persona_ids, reverse=True)

        self.i2p = dict([(k,v + tokenizer.eos_token) for k,v in i2p.items()])
        self.inp_size = inp_size
        self.p2 = [] #['Not Available.' + tokenizer.eos_token]

        action_size = len(self.i2p)
        self.steps = 1
        # policy networks: old and new
        self.policy = ACNet(inp_size=inp_size, action_size=action_size, 
                            hidden_size=hidden_size, dropout=dropout).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
    
    def _update_persona(self, action):
        if action is not None:
            #self.persona_ids.append(action)
            self.persona_ids = [action]
            self.p1 = [self.i2p[pi] for pi in self.persona_ids]
            
    def __call__(self, inp, state=None, context=None, reward=None):
        # update state representation
        if state is None: 
            #TODO: better state initialization?
            state = to_var(torch.zeros((1,self.inp_size)))
            assert context is None
            context = to_var(torch.zeros((1, self.inp_size)))
        else: 
            if len(state.shape) < 2: state = state.unsqueeze(0)
            if len(context.shape ) <2: context = context.unsqueeze(0)
            assert len(state.shape) == len(context.shape) == 2
            state = to_var(state)
        # track reward if not first step
        if reward is not None:
            self.episode['rewards'].append(reward)
            
        # act and update episodic memory
        sampled_action = self.policy.act(state, context, self.episode)
        # update persona and action history
        self._update_persona(sampled_action)
        return self.step(inp, act=True)
    
    def reset_convo(self):
        # reset personas, pasts and dialog history
        self.act_descr, self.p2_descr, self.sep_tok = ['<|act|>'] + ['action 1: '], ['<|p2|>']+ ['person 2: '],  [tokenizer.sep_token]        
        self.dialog_history, self.turn = [], 0
        self.persona_ids = []
        # episodic memory
        self.episode = {'states':[], 'actions': [], 'log_probs':[], 'rewards': [], 
                        'returns': [], 'contexts': [], 'logits': [] }
        
    #TODO: compare classic returns vs. shaped returns
    def _calculate_classic_returns(self, rewards):
        '''calculates the RAW returns for each episode. '''
        R, returns = 0, []
        for r in rewards[::-1]:
            R = r + opts.gamma * R
            returns.insert(0,R)
        returns = np.array(returns)
        return returns
    
    def _calculate_sparse_returns(self, rewards):
        '''R_t = r @ final timestep '''
        returns = np.ones_like(rewards) * rewards[-1]
        return returns
    
    def _on_episode_end(self, last_msg, last_reward):
        self._update_dialog_hx(last_msg)
        #self.episode['states'].append(last_state)
        #self.episode['contexts'].append(last_context)
        self.episode['rewards'].append(last_reward)
        # calculate returns
        self.episode['returns'] = self._calculate_sparse_returns(self.episode['rewards'])
        #return np.array(self.episode['actions']), np.array(self.episode['log_probs']), np.array(self.episode['returns'])
        
    def update(self, batch):
        '''batch: python dictionary of states, actions, log_probs, and returns
        calculates the batch rewards (losses) and performs update on actor and critic'''
        states, actions, log_probs, logits, returns = batch['states'], batch['actions'], batch['log_probs'], batch['logits'], batch['returns']
        states, log_probs, logits = torch.stack(states,0), torch.stack(log_probs, 0), torch.stack(logits,0)
        actions, returns = map(torch.tensor, (actions, returns))
        states, actions, log_probs, logits, returns = map(to_var, (states, actions, log_probs, logits, returns))

        ### actor loss ###
        #advantages = returns - values.detach() # don't backprop through value net for this         
        actor_loss = -log_probs * returns
        
        ### critic loss ###
        # max entropy regularization on values
        max_ent = torch.ones_like(logits).to(device)
        H = 1e-2 * torch.sum((logits - max_ent)**2) 
        
        # backprop and step
        loss =  actor_loss.mean() + H
        if opts.gradient_accumulation_steps > 1:
            loss = loss / opts.gradient_accumulation_steps
        loss.backward()
        
        self.steps += 1
        if (self.steps) % opts.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.policy.zero_grad()
            print()
            print("grad updated ! ")
            print()
                
    
    def save(self):
        #torch.save(self.old_policy.state_dict(), os.path.join(opts.authenticator_path,'old_policy.pt'))
        torch.save(self.policy.state_dict(), os.path.join(opts.authenticator_path,'policy.pt'))
    
    def load(self):
        policy_state_dict = torch.load(os.path.join(opts.authenticator_path, 'policy.pt'))
        self.policy.load_state_dict(policy_state_dict)
    
