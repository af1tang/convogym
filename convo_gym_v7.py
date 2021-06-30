#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:55:18 2021

V7.0 vanilla PG run 
    - R = exp(-L)
    - A = R
    - J = logp * A - ent
    - NN: relu -> tanh
    - running mean R and var calculation
    - k = 20
    
@author: af1tang
"""
import torch, os, pickle, random, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
from load_configs import *
from utils import *
from models import *
from agents import *
    
class ConvGym(object):
    def __init__(self, clf, i2p, action_dict, train_set = None, test_set = None,
                 num_convos = 20,  title = None,
                 k=20, alpha=1., sigma=10.,
                 length=8, top_k=10, top_p = .92, max_length=1024 ):
        self.i2p = i2p
        self.clf = clf
        self.action_dict = action_dict
        # background model and reward parameters
        self.k, self.alpha, self.sigma = k, alpha, sigma
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        # conversation details
        self.num_convos = num_convos
        #self.R_mean, self.R_var, self.N = None, None, 1
        self.total_reward = []
        self.train_set, self.test_set = train_set, test_set
        self.title = title
        # initialize datasets and model
        self._reset_memory()
        self._reset_env()
        self.dataset, self.meta_data = {}, {}
        self.meta_agent = MetaAgent(self.action_dict)
        
    ### reset functions ###
    def _reset_env(self):
        self.agent = None
        self.agent = Agent(self.action_dict, top_k=self.top_k, top_p=self.top_p)
        
    def _reset_memory(self):
        self.memory = {'states':[], 'actions': [], 'log_probs': [], 
                       'returns':[], 'contexts':[], 'logits': []}
    
    def _reset_agents(self, p2_ids):
        '''reset agent and user conversational states'''
        self.agent.reset_convo()
        self.user = UserSim(self.i2p, p2_ids, top_k=self.top_k, top_p=self.top_p)
    
    ### reward and state feedbacks ###
    def _get_state(self, dialog_hx):#, actions):
        # hx x action -> state
        with torch.no_grad():
            state_conv = model(to_var(flatten(dialog_hx[1::2])).long(), 
                         output_hidden_states=True)[2][24].squeeze(0).mean(0)
            # predicted personality -> 1024 clf state
            state_clf = self.clf.clf(state_conv.unsqueeze(0))[0].squeeze(0)
        return state_conv, state_clf
    
        
    def _calculate_reward(self, state, labels):
        with torch.no_grad():
            pos, neg = self.clf._generate_candidates([labels], self.k)
            loss = self.clf.criterion(state.unsqueeze(0), pos[0], neg[0])
            # turn loss into a distribution q_t over actions
            reward = self.alpha * torch.exp( - loss / self.sigma) + 1e-5
            return reward.item()

    ### data management and batching ###
    def _update_buffer(self, states, actions, log_probs, returns, contexts, logits):
        self.memory['states'].extend(states)
        self.memory['actions'].extend(actions)
        self.memory['returns'].extend(returns)
        self.memory['log_probs'].extend(log_probs)
        self.memory['contexts'].extend(contexts)
        self.memory['logits'].extend(logits)
        
    def _update_dataset(self, i, memories, data):
        # update dataset
        if i not in self.dataset.keys():
            self.dataset[i] = {'states': [], 'contexts': [], 'logits': []}
        self.dataset[i]['states'].extend(memories['states'])
        self.dataset[i]['contexts'].extend(memories['contexts'])
        self.dataset[i]['logits'].extend(memories['logits'])
        # update meta-data
        if i not in self.meta_data.keys():
            self.meta_data[i] = data
        else: self.meta_data.update(data)
        
    def _normalize_returns(self):
        if len(self.total_reward)<=2:
            normalized_returns = (self.agent.episode['rewards'] - np.mean(self.agent.episode['rewards'])) / (np.std(self.agent.episode['rewards']) + 1e-5)
        else:
            normalized_returns = (self.agent.episode['returns'] - np.mean(self.total_reward)) / (np.std(self.total_reward) + 1e-5)
        return normalized_returns
    
    def _get_policy_batch(self):
        '''on-policy batch '''
        #memory_size = len(self.memory['states'])
        #indices = list(range(memory_size))[-opts.ppo_memory_size:] # get last 10 trajectories
        normalized_returns = self._normalize_returns()
        # batch = dict([(k, {'states': self.agent.episode['states'][k], 
        #                 'actions': self.agent.episode['actions'][k],
        #                 'log_probs': self.agent.episode['log_probs'][k],
        #                 'returns': normalized_returns[k],
        #                 #'returns': self.memory['returns'][v]
        #                 }) for k in list(range(len(self.agent.episode['states'])))])
        batch = {'states': self.agent.episode['states'], 'actions': self.agent.episode['actions'],
                 'log_probs': self.agent.episode['log_probs'], 
                 'logits': self.agent.episode['logits'], 'returns': normalized_returns}
        return batch
    
    def _get_sl_batch(self, num_samples = 1024, indices = None):
        '''off-policy batch '''
        states, contexts, logits = [],[],[]
        keys = indices if indices is not None else sorted(self.dataset.keys()) 
        for i in keys:
            states.extend(self.dataset[i]['states'])
            contexts.extend(self.dataset[i]['contexts'])
            logits.extend(self.dataset[i]['logits'])
        
        if indices is not None:    
            batch = dict([(k, {'states': states[k], 'contexts': contexts[k], 
                           'logits': logits[k]}) for k in range(len(states))])
        else:
            #max_samples = min(num_samples, len(states))
            #batch_indices = random.sample(set(range(len(states))), max_samples)
            batch_indices = list(range(len(states)))
            random.shuffle(batch_indices)
            batch = dict([ (k, {'states': states[v], 'contexts': contexts[v], 
                           'logits': logits[v]}) for k,v in enumerate(batch_indices) ])

        return batch
    
    ### convo simulation ###
    def _generate_trajectory(self, persona_input = None): 
        '''generate a trajectory of 
            - dialog history
            - reward associated w/ each turn  '''
        self._reset_agents(persona_input)
        msg, rewards, states, contexts = None, [], [],[]
        state, reward, context = None, None, None
        # run for turns, updating rewards at each turn
        for turn in range(self.length):
            # get action
            msg = self.agent(msg, state, context, reward)
            msg = self.user(msg)
            # get next state
            state, context = self._get_state(self.user.dialog_history) #, self.agent.persona_ids)
            states.append(state); contexts.append(context)
            # get reward using CONTEXT (state_clf)
            reward = self._calculate_reward(context, self.user.persona_ids)
            rewards.append(reward)
        # update dialog hx
        dialog_hx = self.user.dialog_history
        # on episode end
        self.agent._on_episode_end(msg, reward)
        self._update_buffer(np.array([to_data(state) for state in self.agent.episode['states']]),
                            np.array(self.agent.episode['actions']), 
                            np.array([to_data(logp) for logp in self.agent.episode['log_probs']]),
                            np.array(self.agent.episode['returns']), 
                            np.array([to_data(ctx) for ctx in self.agent.episode['contexts']]),
                            np.array([to_data(y) for y in self.agent.episode['logits']]) )
        # return FINAL context, which is clf input to calculate rewards
        return dialog_hx, reward, context, np.array(self.agent.episode['actions'])  
              
    def simulate_convos(self, idx, persona_inp, num_convos, training = False):
        #self._reset_env()
        data = {}
        print("="*50)
        print("Conducting conversations with environment %d ..."%idx)
        print("="*50)
        total_rewards = [] # the learning curve

        prec1s, prec5s, rec5s, rec10s = [], [], [], []
        for i in range(num_convos):
            dialog_hx, episode_reward, context, actions = self._generate_trajectory(persona_inp)
            user_ids = self.user.persona_ids 
            # track rewards 
            #episode_loss = -np.log(episode_reward+1e-5) * self.sigma
            self.total_reward.append(episode_reward); total_rewards.append(episode_reward)
            # if self.R_mean is not None:
            #     self.N += 1; alpha = 2/ self.N
            #     self.R_mean = alpha * episode_reward + (1-alpha) * self.R_mean
            #     self.R_var = (1 - alpha) * (self.R_var + alpha * (episode_reward - self.R_mean)**2)
            # calculate metrics 
            clf_inp = to_var(context).view(1,-1)    # NOT state, use context vector 
            prec1, prec5, rec5, rec10 = clf.evaluate(clf_inp, [user_ids])
            prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
            
            # Q function is off-policy, inputs are single actions rather than logits.
            data[i] = {'dialog_hx': dialog_hx, #'states': states,
                       'user_ids': self.user.persona_ids, 'agent_ids': actions}

            print("[env: %d, iter: %d / %d] reward: %.2f" %(idx, i+1, num_convos, episode_reward))
            print("prec@1: %.1f, prec@5: %.1f, rec@5: %.1f, rec@10: %.1f "%(100*np.mean(prec1s), 
                                                                            100*np.mean(prec5s), 
                                                                            100*np.mean(rec5s),
                                                                            100*np.mean(rec10s)))
            print("user ids: ",self.user.persona_ids)
            print("agent actions:", actions.reshape(-1))
            #print(); display_dialog_history(dialog_hx); print()
            #print("-"*50)
            
            if training:
                batch = self._get_policy_batch()
                self.agent.update(batch)
                #self._reset_memory()
            # logging
            if (i+1) % 4 == 0:
                plot_rewards(self.total_reward, True, self.title)
                mean_prec1, mean_prec5, mean_rec5, mean_rec10 = list(map(np.mean, (prec1s, prec5s, rec5s, rec10s)))
                plot_ir_metrics( mean_prec1, mean_prec5, mean_rec5, mean_rec10, True, self.title)
        
        data['prec@1'] = prec1s; data['prec@5'] = prec5s; 
        data['rec@5'] = rec5s; data['rec@10'] = rec10s
        data['total_rewards'] = total_rewards
        return data
    
    ### train and eval ###
    def train_agent(self):
        #TODO: figure out resets and multiple policies
        #for e in range(opts.epochs):
            # run policy in different environments
        for i in range(len(self.train_set)):
            self._reset_memory()            
            persona_inp = self.train_set[i]
            data = self.simulate_convos(i, persona_inp, self.num_convos, training=True)
            self._update_dataset(i, self.memory, data)
        # run GPS on meta policy
        #batch = self._get_sl_batch()
        #self.meta_agent._reset_optimizer()
        #self.meta_agent.fit(batch)
        
    def pretrain_agent(self, iters = None):
        prec1s, prec5s, rec5s, rec10s = [], [], [], []
        histories, p1s, p2s = [], [], []
        
        data = {}
        if iters is None: iters = len(self.train_set)
        for i in range(iters):
            p2 = self.train_set[i]
            # reset convo agents 
            agent = UserSim(i2p, None, top_k=top_k, top_p = top_p)
            user = UserSim(i2p, p2, top_k = top_k, top_p = top_p)
            state, reward, context = None, None, None
            msg, rewards, states, contexts, actions = None, [], [], [], []
            # run for first N-1 turns
            for turn in range(8):
                if len(user.dialog_history) > 0:
                    display_dialog_history(user.dialog_history)
                print('-'*12, ' iter %d, turn %d '%(i, turn), '-'*12 )
                print(" personas: ")
                for p in user.p2: print(p)
                print()
                print(" actions: ")
                for k in range(len(action_dict)): print(k, action_dict[k])
                action = int(input(" input [0-11]: " ))
                agent.p2 = [action_dict[action]]
                msg = agent(msg, state, context, reward, act=True)
                msg = user(msg)
                state, context = self._get_state(user.dialog_history)
                states.append(state.tolist()); contexts.append(context.tolist())
                actions.append(action)
        
                reward = self._calculate_reward(context, p2)
                rewards.append(reward)
        
            agent._on_episode_end(msg, reward)
            #display_dialog_history(agent.dialog_history)
            prec1, prec5, rec5, rec10 = self.clf.evaluate(context.view(1,-1), [p2])
            prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
            histories.append(agent.dialog_history); p1s.append((agent.p1, agent.p2)); p2s.append((user.p1, user.p2))
            # update database
            data[i] = {'states': np.array(states), 'actions': np.array(actions), 
                       'contexts': np.array(contexts), 'rewards': np.array(rewards), 'dialog_hx': user.dialog_history}
        
        data['prec1s'] = prec1s; data['prec5s'] = prec5s; data['rec5s'] = rec5s; data['rec10s'] = rec10s
        return data
    
    def eval_agent(self, agent_setting, user_setting):
        if agent_setting == 'none':
            agent = UserSim(self.i2p, None, top_k = self.top_k, top_p = self.top_p)
        elif agent_setting == 'random':
            agent = UserSim(self.i2p, None, reverse=True, top_k = self.top_k, top_p = self.top_p)
        elif agent_setting == 'policy':
            agent = Agent(self.action_dict, top_k=self.top_k, top_p = self.top_p)
            agent.load()
        # reset stats
        prec1s, prec5s, rec5s, rec10s = [], [], [], []
        dialog_histories, p1s, p2s = [], [], []
        # cycle through eval set
        for i in range(len(self.test_set)):            
            # reset user
            p2 = self.test_set[i]
            if user_setting == 'transition':
                user = UserSim(self.i2p, [])
            elif user_setting == 'weak':
                user = UserSim(self.i2p, p2, top_k = 500, top_p = .85)
            else:
                user = UserSim(self.i2p, p2, top_k = self.top_k, top_p = self.top_p)
            # reset agent
            if agent_setting == 'random': action_seq = random.choices(list(self.action_dict.values()), k=8)
            elif agent_setting == 'persona':
                p1 = random.sample(self.train_set, 1)[0]
                agent = UserSim(self.i2p, p1, top_k = self.top_k, top_p = self.top_p)
            agent.reset_convo()
            # run convo trajectory
            msg, state, context = None, None, None
            for turn in range(8):
                if agent_setting == 'random':
                    agent.p2 = [action_seq[turn]]                
                # update msg rounds
                if agent_setting in ['none', 'persona']:
                    msg = agent(msg, act=False)
                elif agent_setting == 'random':
                    msg = agent(msg, act=True)
                else:
                    msg = agent(msg, state, context)
                msg = user(msg)
                state, context = self._get_state(user.dialog_history)
            
            agent._update_dialog_hx(msg)
            # evaluate dialog
            prec1, prec5, rec5, rec10 = self.clf.evaluate(context.view(1,-1), [user.persona_ids])
            # log data
            prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
            dialog_histories.append(agent.dialog_history)
            p2s.append(p2)
            if agent_setting == 'random': p1s.append(action_seq)
            elif agent_setting == 'persona': p1s.append(agent.p1)
            
            print( "[%d / %d ] prec1: %.1f, prec5: %.1f, rec5: %.1f, rec10: %.1f" % (i, 
                                    len(self.test_set), 100*np.mean(prec1s), 100*np.mean(prec5s), 
                                    100*np.mean(rec5s), 100*np.mean(rec10s)) )
            
        return {'prec1': prec1s, 'prec5': prec5s, 'rec5': rec5s, 'rec10': rec10s, 
                'histories': dialog_histories, 'p1': p1s, 'p2': p2s}

    
def create_train_test_personas():
    # make dirs
    create_dir(opts.authenticator_path)
    create_dir(os.path.join(opts.plot_path, 'authenticator-exp'))
    # get train and test persona sets
    with open(os.path.join(opts.identifier_path, 'persona_dict'), 'rb') as f: 
        persona_dict = pickle.load(f)
    tr_personas = [sorted(persona_dict['tr'][i]['p2']) for i in sorted(persona_dict['tr'].keys())]
    #tr_personas.extend(sorted(persona_dict['tr'][i]['p1']) for i in sorted(persona_dict['tr'].keys()))
    tr_unique = []
    for lst in tr_personas:
        # not in current list
        if lst not in tr_unique: 
            relatives = []
            for lst2 in tr_unique:
                # do not have more than 2 differences
                if len(set(lst2) - set(lst)) <3:
                    relatives.append(lst2)
            if len(relatives) <1:
                tr_unique.append(lst)
       #     tr_unique.append(lst)
    te_personas = [sorted(persona_dict['te'][i]['p2']) for i in sorted(persona_dict['te'].keys())]
    #te_personas.extend(sorted(persona_dict['te'][i]['p1']) for i in sorted(persona_dict['te'].keys()))
    te_unique = []
    for lst in te_personas:
        if lst not in te_unique: 
            relatives = []
            for lst2 in te_unique:
                if len(set(lst2) - set(lst)) < 3:
                    relatives.append(lst2)
            if len(relatives) < 1:
                te_unique.append(lst)    
            
    # save
    with open(os.path.join(opts.authenticator_path, 'tr_personas'), 'wb') as f: pickle.dump(tr_unique, f)
    with open(os.path.join(opts.authenticator_path, 'te_personas'), 'wb') as f: pickle.dump(te_unique, f)
    
    
if __name__ == "__main__":
    with open(os.path.join(opts.identifier_path, 'i2p'), 'rb') as f: 
        i2p = pickle.load(f)
    i2v = torch.load(os.path.join(opts.identifier_path, 'i2v'))
    clf = CLF(i2v, mode= opts.identifier_mode, zsl=opts.zsl).cuda()
    # load state dict
    id_save_path = os.path.join(opts.identifier_path, '%s.pt' % opts.identifier_mode)
    state_dict = torch.load(id_save_path)
    clf.load_state_dict(state_dict)
    # initialize train-test personas for experiment
    #create_train_test_personas()
    with open(os.path.join(opts.authenticator_path, 'tr_personas'), 'rb') as f: tr_personas = pickle.load(f)
    with open(os.path.join(opts.authenticator_path, 'te_personas'), 'rb') as f: te_personas = pickle.load(f)

    # action space
    action_space = [ 'ask about kids.', "ask about pets.", 'talk about work.', 
               'ask about martial status.', 'talk about travel.', 'ask about age and gender.',
        'ask about hobbies.', 'ask about favorite food.', 'talk about movies.', 
        'talk about music.', 'talk about politics.']
    action_dict = dict([(k,v) for k,v in enumerate(action_space)])
    
    gym = ConvGym(clf, i2p, action_dict, tr_personas, te_personas, num_convos=32)
