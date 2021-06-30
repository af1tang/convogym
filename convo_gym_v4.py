#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:41:47 2020

V4.0 guided policy search: 
    - individual environments (1,282 of them!)
    - train a "local policy" for each using PPO
    - mild reward shaping: R + R_loss + R_info
    - NEW: state space (S), and persona space (Z) to id the MDP.
    - save (s, a, log_p, R) to memory; 
    - save (s, z, p) to a central dataset D.
    - train "central policy" f(s,z) -> q s.t. d(p,q) is minimized.
    - log rewards for f, p for each environment. 
    - stop criteria for each env: when f,p close AND R(p) good enough.

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
                 num_convos = 20, reward_shaping = False,
                 fat_tailed = False, title = None,
                 k=100, alpha=1., beta=1., sigma=1e2,
                 length=8, top_k=10, top_p = .92, max_length=1024 ):
        self.i2p = i2p
        self.clf = clf
        self.action_dict, self.reward_shaping = action_dict, reward_shaping
        # background model and reward parameters
        self.k, self.alpha, self.sigma, self.beta = k, alpha, sigma, beta
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        # conversation details
        self.num_convos = num_convos
        self.train_set, self.test_set = train_set, test_set
        self.fat_tailed, self.title = fat_tailed, title
        # initialize datasets and model
        self._reset_memory()
        self.dataset, self.meta_data = {}, {}
        self.meta_agent = MetaAgent(self.action_dict)
        
    def _reset_env(self):
        self.agent = None
        self.agent = Agent(self.action_dict, reward_shaping=self.reward_shaping)
        self.meta_agent._reset_optimizer()
    
    def _reset_memory(self):
        self.memory = {'states':[], 'actions': [], 'log_probs': [], 
                       'returns':[], 'contexts':[], 'logits': []}
    
    def _reset_agents(self, personas):
        '''reset agent and user conversational states'''
        self.agent.reset_convo()
        p2_ids = personas
        self.user = UserSim(self.i2p, p2_ids, self.top_k, self.top_p, self.max_length)
        
    def _get_state(self, dialog_hx):#, actions):
        # hx x action -> state
        with torch.no_grad():
            state_conv = model(to_var(flatten(dialog_hx[1::2])).long(), 
                         output_hidden_states=True)[2][24].squeeze(0).mean(0)
            # predicted personality -> 1024 clf state
            state_clf = self.clf.clf(state_conv.unsqueeze(0))[0].squeeze(0)
        return state_conv, state_clf
    
    def _e_func(self, x, alpha=1., beta=None, sigma=1.):
        '''thin-tailed reward func'''
        return alpha * torch.exp( - x / sigma) + 1e-5
        #return - x
    
    def _l_func(self, x, alpha=None, beta=None, sigma=None):
        '''loss (log q) as reward '''
        return - x
    
    def _r_func(self, x, alpha=1., beta=1., sigma=1.):
        '''fat-tailed reward func'''
        return alpha / ( 1 + ( 1 + beta*x )**sigma ) + 1e-5
        
    def _calculate_reward(self, state, labels):
        with torch.no_grad():
            pos, neg = self.clf._generate_candidates([labels], self.k)
            loss = self.clf.criterion(state.unsqueeze(0), pos[0], neg[0])
            # turn loss into a distribution q_t over actions
            reward = self._r_func(loss, self.alpha, self.beta, self.sigma) if self.fat_tailed else self._e_func(loss, self.alpha, self.beta, self.sigma)
            return reward.item()

    def _generate_trajectory(self, persona_input = None): 
        '''generate a trajectory of 
            - dialog history
            - reward associated w/ each turn  '''
        self._reset_agents(persona_input)
        msg, rewards, states, contexts = None, [],[], []
        state, reward, context = None, None, None
        # run for turns, updating rewards at each turn
        for turn in range(self.length):
            # get action
            msg = self.agent(msg, state, context, reward)
            msg = self.user(msg)
            # get next state
            state, context = self._get_state(self.user.dialog_history) #, self.agent.persona_ids)
            states.append(state)
            contexts.append(context)
            # get reward using CONTEXT (state_clf)
            reward = self._calculate_reward(context, self.user.persona_ids)
            rewards.append(reward)
            
        dialog_hx = self.user.dialog_history
        #states = to_data(torch.stack(states))
        #contexts = to_data(torch.stack(contexts))
        #actions, log_probs, returns = self.agent._on_episode_end(msg, reward)
        # update memory w/ agent's trajectory info
        self.agent._on_episode_end(msg, reward)
        self._update_buffer(np.array(self.agent.episode['states']), np.array(self.agent.episode['actions']), 
                            np.array(self.agent.episode['log_probs']), np.array(self.agent.episode['returns']), 
                            np.array(self.agent.episode['contexts']), np.array(self.agent.episode['logits']))
        # return FINAL context, which is clf input to calculate rewards
        return dialog_hx, reward, context, np.array(self.agent.episode['actions'])

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
        
    def _normalize_returns(self, returns):
        normalized_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-5)
        return normalized_returns
    
    def _get_policy_batch(self):
        memory_size = len(self.memory['states'])
        indices = list(range(memory_size))[-opts.ppo_memory_size:] # get last 10 trajectories
        normalized_returns = self._normalize_returns( np.array(self.memory['returns'])[indices] )
        batch = dict([(k, {'states': self.memory['states'][v], 
                        'actions': self.memory['actions'][v],
                        'log_probs': self.memory['log_probs'][v],
                        'returns': normalized_returns[k],
                        }) for k,v in enumerate(indices)])
        return batch
    
    def _get_sl_batch(self, num_samples = 1024, indices = None):
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
                
    def simulate_convos(self, idx, persona_inp, num_convos, training = False):
        self._reset_env()
        data = {}
        print("="*50)
        print("Conducting conversations with environment %d ..."%idx)
        print("="*50)
        total_rewards = [] # the learning curve

        prec1s, prec5s, rec5s, rec10s = [], [], [], []
        for i in range(num_convos):
            dialog_hx, episode_reward, context, actions = self._generate_trajectory(persona_inp)
            user_ids = self.user.persona_ids 
            # update memory
            # self._update_buffer(states, actions, log_probs, returns, contexts)
            # track rewards 
            episode_loss = -np.log(episode_reward+1e-5) * self.sigma
            total_rewards.append(episode_loss)
            # calculate metrics 
            clf_inp = to_var(context).view(1,-1)    # NOT state, use context vector 
            prec1, prec5, rec5, rec10 = clf.evaluate(clf_inp, [user_ids])
            prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
            
            # Q function is off-policy, inputs are single actions rather than logits.
            data[i] = {'trajectory': dialog_hx, #'states': states,
                       'user_ids': self.user.persona_ids, 'agent_ids': actions}

            print("[env: %d, iter: %d / %d] loss: %.2f" %(idx, i+1, num_convos, episode_loss))
            print("prec@1: %.1f, prec@5: %.1f, rec@5: %.1f, rec@10: %.1f "%(100*np.mean(prec1s), 
                                                                            100*np.mean(prec5s), 
                                                                            100*np.mean(rec5s),
                                                                            100*np.mean(rec10s)))
            print("user ids: ",self.user.persona_ids)
            print("agent actions:", actions.reshape(-1))
            print("-"*50)
            
            if training:
                batch = self._get_policy_batch()
                self.agent.update(batch)
                #self._reset_memory()
            # logging
            if (i+1) % 10 == 0:
                plot_rewards(total_rewards, True, self.title)
                mean_prec1, mean_prec5, mean_rec5, mean_rec10 = list(map(np.mean, (prec1s, prec5s, rec5s, rec10s)))
                plot_ir_metrics( mean_prec1, mean_prec5, mean_rec5, mean_rec10, True, self.title)
        
        data['prec@1'] = prec1s; data['prec@5'] = prec5s; 
        data['rec@5'] = rec5s; data['rec@10'] = rec10s
        data['total_rewards'] = total_rewards
        return data
    
    def train_agent(self):
        #TODO: run evaluation as GPS outer loop
        for i in range(len(self.train_set)):
            persona_inp = self.train_set[i]
            self._reset_memory()
            data = self.simulate_convos(i, persona_inp, self.num_convos, training=True)
            self._update_dataset(i, self.memory, data)
            batch = self._get_sl_batch()
            self.meta_agent.fit(batch)

    def eval_agent(self):
        self._reset_env()
        data = {}
        print("Conducting conversations on test set ...")
        print("="*50)
        for i, persona_inp in enumerate(self.test_set):
            dialog_hx, rewards, states, actions= self._generate_trajectory(persona_inp)
            # track rewards 
            total_rewards.append(sum(rewards))
            # calculate metrics 
            state = to_var(states[-1]).view(1,-1)
            prec1, prec5, rec5, rec10 = clf.evaluate(state, [user_ids])
            prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
            
            data[i] = {'trajectory': dialog_hx, 'states': states,
                       'user_ids': self.user.persona_ids, 'agent_ids': self.agent.persona_ids ,
                       'actions': actions, 
                       'metrics': {'rewards': rewards, 'prec1': prec1, 'prec5': prec5, 
                                   'rec5': rec5, 'rec10': rec10}}
            
            print( " [%d / %d] rewards: %.2f" %(i+1, self.num_convos, np.sum(rewards)))
            print("user ids: ",self.user.persona_ids)
            print("agent ids:", self.agent.persona_ids)
            #print("actions: ", actions)
            print("-"*50)

        # logging
        plot_rewards(total_rewards, False)
        mean_prec1, mean_prec5, mean_rec5, mean_rec10 = list(map(np.mean, (prec1s, prec5s, rec5s, rec10s)))
        plot_ir_metrics( mean_prec1, mean_prec5, mean_rec5, mean_rec10, False)
                
        data['summary'] = {'total_rewards': total_rewards, 'prec1': prec1s, 'prec5': prec5s,
                           'rec5': rec5s, 'rec10': rec10s}
        return data
    
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

    # run training loop
    with open(os.path.join(opts.authenticator_path, 'actions_qs.pkl'), 'rb') as f: actions_qs = pickle.load(f)
    with open(os.path.join(opts.authenticator_path,'actions_descr.pkl'), 'rb') as f: actions_ps = pickle.load(f)

    action_dict = dict([(k,v) for k,v in enumerate(actions_qs)])
    title = "local_test"
    
    #gym = ConvGym(clf, i2p, action_dict, title=title, fat_tailed=False,
    #              num_convos=10, reward_shaping=False,
    #                train_set = tr_personas, test_set = te_personas, k=100)
    #gym.train_agent()
    # saving stats
    #with open(os.path.join(opts.authenticator_path, 'stats_%s'%title), 'wb') as f: pickle.dump(data, f)
    
