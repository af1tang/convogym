#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 20:25:09 2021

@author: af1tang
"""
import torch, random

from _configs import opts, action_space
from _tokenizer import tokenizer, start_tok
from _personas import train_personas, test_personas

from utils._visualization import display_dialog_history 
from utils._reshape import flatten 
from utils._turn_filter import to_tokens

from agents import Agent
from callbacks import StateCb, MessageCb

## interaction funcs ##
def get_custom_persona(*args, **kwargs):
    personas = []
    for i in range(opts.num_personas):
        response = ""
        while len(response) <1:
            response = input(">> Fact %d: "%(i+1))+ tokenizer.eos_token
        personas.append(response)
    return personas

def get_random_persona(persona_list):
    if not persona_list:
        return random.sample(train_personas, 1)[0]
    else:
        return random.sample(persona_list, 1)[0]
        

def get_sequence_personas(persona_list):
    random.shuffle(persona_list)
    p_iter = iter(persona_list)
    while p_iter:
        try:
            yield next(p_iter)
        except:
            return

## gym environments ##
class Gym:
    '''default gym environment'''
    def __init__(self, model, user=None, interactive = True,  
                 reset_persona_func=get_custom_persona, 
                 length=8, top_k=10, top_p = .92, max_length=1024 ):
        self.model = model
        self.user = user
        self.interactive = interactive
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        self.reset_persona_func = reset_persona_func
        self.data = {'hx': [], 'p1': [], 'p2': []}
    
    def _reset_agents(self, persona_input, agent_personas, user_personas):
        '''reset agent conversational states'''
        if persona_input:
            self.agent = Agent(persona_input, top_k=self.top_k,
                               top_p=self.top_p, max_length=self.max_length)
        else:
            persona = self.reset_persona_func(agent_personas)
            self.agent = Agent(persona, reverse=False, top_k=self.top_k, 
                              top_p=self.top_p, max_length=self.max_length)
            
        if self.interactive:
            # human input
            self.user = self._interact
        else:
            # another persona model
            persona = self.reset_persona_func(user_personas)
            self.user = Agent(persona, reverse=True, top_k=self.top_k, 
                              top_p=self.top_p, max_length=self.max_length)
        
    def _interact(self, model, msg, act):
        if msg:
            print("Bot: {}".format(tokenizer.decode(msg, skip_special_tokens=True)))
        msg = tokenizer.encode(input(">> User: ") + tokenizer.eos_token)
        return msg     
    
    def _on_convo_begin(self, scb, mcb):
        scb.personas = self.agent.p2
        return scb,mcb
        
    def _on_user_begin(self, scb, mcb):
        return scb, mcb
    
    def _on_user_end(self, scb,mcb):
        mcb.dialog_hx.append(mcb.msg)
        return scb, mcb
    
    def _on_agent_end(self, scb, mcb):
        mcb.dialog_hx.append(mcb.msg)
        scb.turn +=1
        if scb.turn >= self.length:
            scb.done = True
        return scb, mcb
    
    def _on_convo_end(self, scb, mcb):
        if not self.interactive:
            print('p1: ')
            print()
            for p in self.user.p1: print(p)
        print('-'*10)
        print('p2: ')
        print()
        for p in scb.personas: print(p)
        print('-'*10)
        print()
        display_dialog_history(mcb.dialog_hx)
        self.data['hx'].append(to_tokens( mcb.dialog_hx ))
        if self.interactive:
            self.data['p1'].append([])
        else:
            self.data['p1'].append(self.user.p1)
        self.data['p2'].append(scb.personas)
        del scb, mcb

    def _sim_convo(self, persona_input = None, 
                   agent_personas=None, user_personas=None): 
        '''generate a trajectory of 
            - dialog history
            - reward associated w/ each turn  '''
        self._reset_agents(persona_input, agent_personas, user_personas)
        scb, mcb = StateCb(personas=agent_personas), MessageCb()
        scb, mcb = self._on_convo_begin(scb,mcb)
        # run for turns, updating rewards at each turn
        while not scb.done:
            # person 1 (user) 
            scb,mcb = self._on_user_begin(scb,mcb)             
            mcb.msg = self.user(self.model, mcb.msg, act=scb.act)
            scb,mcb = self._on_user_end(scb,mcb)

            # person 2 (agent)
            mcb.msg = self.agent(self.model, mcb.msg, act=scb.act)
            scb, mcb = self._on_agent_end(scb,mcb)
        self._on_convo_end(scb, mcb) 
    
    def sim_convos(self, num_convos=1, agent_personas=None, user_personas=None):
        print("Conducting conversations ...")
        print()
        for i in range(num_convos):
            self._sim_convo(None, agent_personas, user_personas)
 
    def run_evaluation(self):
        raise NotImplementedError("evaluation loop not implemented for current environment.")
    
        
# active learning environments
class ActiveGym(Gym):
    '''active learning environment'''
    def __init__(self, model, user=None, length=8, top_k=10, top_p = .92, max_length=1024):
        super().__init__(model=model, user=None, interactive=True, 
                         reset_persona_func=get_sequence_personas,
                         length=length, top_k=top_k, top_p=top_p, max_length=max_length)
        self.data = {'X': [], 'y': [], 'dialog_hx': [], 'actions': []}
    
    def _reset_agents(self, persona_input, agent_personas=None, user_personas=None):
        self.user = Agent([], top_k=self.top_k, top_p=self.top_p, max_length=self.max_length)
        self.agent = Agent(persona_input, top_k=self.top_k,
                               top_p=self.top_p, max_length=self.max_length)
        
    def _on_convo_begin(self, scb, mcb):
        scb.personas = self.agent.p2
        scb.record = True
        return scb,mcb
        
    def _on_user_begin(self, scb, mcb):
        if mcb.msg is not None:
            mcb.dialog_hx.append(mcb.msg)
        action = None
        while action not in action_space:
            display_dialog_history(mcb.dialog_hx)
            print()
            print(" actions: ")
            for k,v in enumerate(action_space): print(k,v)
            try:
                int_act = int(input(" input [0-10]: " ))
                action = action_space[int_act]
            except:
                action = None            
        scb.action = action
        scb.actions.append(int_act)
        scb.act = True
        self.user.p1 = [scb.action]
        # cache current x
        x = tokenizer.encode(''.join(['<|act|> '] + self.user.p1 + ['<|sep|>'] + ['<|p1|>'] + [] + ['<|sep|>'] + ['<|start|>']))
        x += flatten(mcb.dialog_hx)
        x = torch.tensor([x])
        # set inp as input_ids for dataset
        if scb.turn == 0:
            x = x[:, :-1]
        mcb.x = x
        return scb, mcb
    
    def _on_user_end(self, scb,mcb):
        mcb.dialog_hx.append(mcb.msg)
        # check if need revision
        print(); print('-'*50)
        display_dialog_history(self.user.dialog_history)
        print('-'*12, ' iter %d, turn %d '%(self.iter, scb.turn), '-'*12 )
        print("action: ", scb.action)
        decision = input(" continue? [y/n] ")
        # decision tree
        if decision == 'y':
            # augment even more turns to active data
            # self.data['X'].extend(mcb.x.tolist()); self.data['y'].append(mcb.msg)
            # continue conversation
            x = tokenizer.encode(''.join(['<|p2|>'] + scb.personas + ['<|sep|>'] + ['<|start|>']))
            x += flatten(mcb.dialog_hx)
            x = torch.tensor([x])
            mcb.x = x
        else:
            y = [[]]
            while len(y[0]) < 2:
                y = tokenizer.encode(input("  >> user: ") + tokenizer.eos_token, return_tensors='pt')
            if scb.turn ==0:
                y = torch.cat( (torch.tensor([start_tok]).unsqueeze(0), y), -1)
            self.data['X'].extend( mcb.x.tolist() ); self.data['y'].extend( y.tolist())
            # retart convo
            scb.done, scb.record = True, False
        scb.act = False
        return scb, mcb
    
    def _on_agent_end(self, scb, mcb):
        if not scb.done:
            mcb.dialog_hx.append(mcb.msg)
            display_dialog_history(self.agent.dialog_history)
            print('-'* 12, ' iter %d, turn %d ' %(self.iter, scb.turn), '-' * 12)
            print(" personas: ")
            for p in scb.personas: print(p)
            decision = input( " continue? [y/n] " )
            if decision == 'y':
                scb.turn +=1
                if scb.turn >= self.length:
                    scb.done = True
            else:
                y = [[]]
                while len(y[0]) < 2:
                    y = tokenizer.encode(input("  >> user: ") + tokenizer.eos_token, return_tensors='pt')
                self.data['X'].extend( mcb.x.tolist() ); self.data['y'].extend( y.tolist())
                # retart convo
                scb.done, scb.record = True, False
        return scb, mcb
    
    def _on_convo_end(self, scb, mcb):
        if scb.record:
            self.data['dialog_hx'].append(to_tokens(mcb.dialog_hx))
            self.data['actions'].append(scb.actions)
        del scb, mcb
        
    def sim_convos(self, num_convos=9999):
        print("Conducting conversations ...")
        print()
        max_num_convos = min(len(train_personas) * opts.num_epochs, num_convos)
        for self.iter, persona in enumerate(self.reset_persona_func(train_personas)):
            self._sim_convo(persona, None, None)
            if self.iter > max_num_convos:
                break

# RL environments
class RLGym(Gym):
    '''DQN environment'''
    def __init__(self, model, policy, env, logger, reward_obj, 
                 max_buffer_size = 1000,
                 length=8, top_k=10, top_p = .92, max_length=1024):
        super().__init__(model=model, user=None, interactive=False, 
                         reset_persona_func=get_sequence_personas,
                         length=length, top_k=top_k, top_p=top_p, max_length=max_length)
        # REQUIRE: policy object
        self.policy = policy
        self.R = reward_obj
        self.Env = env
        self.logger = logger
        self.data = {'dialog_hx':[], 'actions': [], 'personas': []}
        self.memory_buffer, self.max_buffer_size = [], max_buffer_size
    
    def _reset_agents(self, persona_input, agent_personas=None, user_personas=None):
        self.user = Agent([], top_k=self.top_k, top_p=self.top_p, max_length=self.max_length)
        self.agent = Agent(persona_input, top_k=self.top_k,
                               top_p=self.top_p, max_length=self.max_length)
    
    def _on_convo_begin(self, scb, mcb):
        scb = self.Env.reset(scb)
        scb.personas = self.agent.p2
        return scb, mcb
        
    def _on_user_begin(self, scb, mcb):
        # sample action from policy
        #self.policy.eval()
        inp = self.Env.get_policy_inp(scb)
        int_act = self.policy.act(inp, scb.turn)
        # update action
        scb.action = action_space[int_act]
        scb.actions.append(int_act)
        scb.act = True
        self.user.p1 = [scb.action]
        return scb, mcb
    
    def _on_user_end(self, scb,mcb):
        mcb.dialog_hx.append(mcb.msg)
        scb.act = False
        return scb, mcb
    
    def _on_agent_end(self, scb, mcb):
        mcb.dialog_hx.append(mcb.msg)
        # update state based on dialog history
        scb, mcb = self.Env.get_curr_state(scb, mcb)
        # calculate reward 
        scb, mcb = self.R.calculate_reward(scb, mcb)
        # transition to next state
        scb, mcb = self.Env.get_next_state(scb, mcb)
        # next turn
        scb.turn +=1
        if scb.turn >= self.length:
            scb.done = True
        return scb, mcb
    
    def _update_memory(self,batch):
        self.memory_buffer.extend([list(tuples) for tuples in batch])
        # delete from front of memory batch if full
        if len(self.memory_buffer) > self.max_buffer_size:
            self.memory_buffer = self.memory_buffer[len(batch):]
    
    def _on_convo_end(self, scb, mcb):
        # display
        print('actions: ')
        print()
        for p in scb.actions: print(action_space[p])
        print('-'*10)
        print('p2: ')
        print()
        for p in scb.personas: print(p)
        print('-'*10)
        print()
        display_dialog_history(mcb.dialog_hx)
        # evaluate dialog and log
        self.logger(scb, mcb)
        self.data['dialog_hx'].append(to_tokens(mcb.dialog_hx))
        self.data['actions'].append(scb.actions)
        self.data['personas'].append(self.agent.p2)
        # update memory buffer
        next_states = scb.states[1:]; states = scb.states[:-1]
        next_contexts = scb.contexts[1:]; contexts = scb.contexts[:-1]
        next_acts = scb.actions[1:] + [0]
        batch = list(zip(states, contexts, scb.actions, next_states, next_contexts, next_acts, scb.rewards))
        self._update_memory(batch)
        del scb, mcb
    
    def sim_convos(self, num_convos=9999, training=True):
        print("Conducting conversations ...")
        print()
        if training:
            max_num_convos = min(len(train_personas) * opts.num_epochs, num_convos)
            for epoch in range(opts.num_epochs):
                for self.iter, persona in enumerate(self.reset_persona_func(train_personas)):
                    print("="*20, "epoch %d, iter %d, training" %(epoch+1, self.iter), "="*20)
                    self._sim_convo(persona, None, None)
                    # policy update
                    print("Offline batch updates ... ")
                    self.policy.update_policy(self.memory_buffer)
                    print()
                    if self.iter > max_num_convos:
                        break
        else:
            max_num_convos = min(len(test_personas), num_convos)
            for self.iter, persona in enumerate(self.reset_persona_func(test_personas)):
                print("="*20, "iter %d, testing" %self.iter, "="*20)
                self._sim_convo(persona, None, None)
                if self.iter > max_num_convos:
                    break
        