#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:16:51 2020

@author: af1tang
"""

import torch, os, pickle, random, numpy as np
import torch.nn as nn, torch.nn.functional as F
from load_configs import *
from utils import *
from models import *

class UserSim(object):
    def __init__(self, i2p, persona_ids, 
                 top_k=10, top_p = .92, max_length=1024):
        self.i2p = i2p
        self.top_k, self.top_p, self.max_length = top_k, top_p, max_length
        self.NA_token = ['Not Available.'] + [tokenizer.eos_token]
        self.p1 = self.NA_token # don't know your partner's persona
        self.persona_ids = persona_ids
        if persona_ids is not None:
            self.p2 = [self.i2p[pi] for pi in self.persona_ids] # given persona
        else:
            self.p2 = []
        self.reset_convo()
        
    def __call__(self, inp, state=None):
        return self.step(inp)
        
    def reset_convo(self):
        # reset personas, pasts and dialog history
        self.p1_descr, self.p2_descr, self.sep_tok = ['<|p1|>']+ ['person 1: '], ['<|p2|>']+ ['person 2: '],  [tokenizer.sep_token]
        #self.inp = tokenizer.encode(''.join(self.p1_descr + self.p1 + self.sep_tok + self.p2_descr + self.p2 +self.sep_tok + ['<|start|>']), return_tensors='pt').to(device)
        # use PAST (src and trg) to keep track of (a) gradients, (b) history for separate decoding
        #self.past, self.curr_len = None, self.inp.size(1)
        self.dialog_history, self.turn, self.actions = [], 0, []
        
    def _update_persona(self, action):
        raise NotImplementedError("Update agent not specified for this user.")
    
    def _reset_inp(self):
        self.inp = tokenizer.encode(''.join(self.p1_descr + self.p1 + self.sep_tok + self.p2_descr + self.p2 +self.sep_tok + ['<|start|>']))
        self.inp += flatten(self.dialog_history)
        self.inp, self.curr_len, self.past = to_var(torch.tensor([self.inp])), len(self.inp), None
    
    def _update_dialog_hx(self, new_inp):
        if new_inp is not None:
            self.dialog_history.append(new_inp)
        
    def step(self, inp):
        self._update_dialog_hx(inp)
        #self._update_persona(action) # do on agent side
        self._reset_inp()
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
    
    def update(self, *args, **kwargs):
        pass
    
class Agent(UserSim):
    def __init__(self, i2p, persona_ids=[], persona_actions= None,
                 use_loss = True,
                 inp_size=1024, embed_size=64,
                 hidden_size=512, num_layers=1, 
                 top_k=10, top_p = .92, max_length=1024):
        super(Agent, self).__init__(i2p, persona_ids)
        #TODO: make sure that i2p is training personas only
        self.i2p = dict([(k+1, v) for k,v in i2p.items()])
        self.i2p[0] = "Not Available." + tokenizer.eos_token
        self.inp_size = inp_size
        self.persona_actions = persona_actions
        action_size = len(self.persona_actions) if self.persona_actions is not None else len(self.i2p)
        self.policy = ACNet(inp_size, action_size, embed_size,
                            hidden_size, num_layers).to(device)
        self.action = to_var(torch.zeros((1,1)).long())
        self.hidden_state, self.steps = None,0 
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.use_loss = use_loss
    
    def _update_persona(self, action):
        if action is not None:
            #self.persona_ids.append(action)
            #TODO: persona id to list of persona traits
            self.persona_ids = self.persona_actions[action] if self.persona_actions is not None else [action]
            self.p2 = [self.i2p[pi] for pi in self.persona_ids]
            
    def __call__(self, inp, state=None):
        if state is None: 
            state = to_var(torch.zeros((1,1, self.inp_size)))
        else: 
            l = len(state.shape)
            while l < 3: state = state.unsqueeze(0); l+=1
            state = to_var(state)
        log_prob, value, self.hidden_state = self.policy(state, self.action, self.hidden_state)
        self.actions.append([log_prob.squeeze(1), value])
        sampled_action = self.policy.act(log_prob.squeeze(1))
        # update persona and action history
        self._update_persona(sampled_action.item())
        self.action_history.append(sampled_action.item())
        # update action input to current sampled action
        self.action = sampled_action.unsqueeze(0)
        return self.step(inp)
    
    def reset_convo(self):
        # reset personas, pasts and dialog history
        self.p1_descr, self.p2_descr, self.sep_tok = ['<|p1|>']+ ['person 1: '], ['<|p2|>']+ ['person 2: '],  [tokenizer.sep_token]
        self.dialog_history, self.turn, self.actions = [], 0, []
        self.persona_ids, self.action_history = [], []
        self.hidden_state = None

    def update(self, episode=None, step=None):
        '''episde: tuple of (states, actions, rewards) 
        calculates the batch rewards (losses) and performs update on actor and critic'''
        if episode is not None:
            states, actions, rewards = episode
            actions, values = list(zip(*actions))
            actions = torch.stack(actions).to(device)
            policy_losses, value_losses = [], []
            
            # TODO: change the loss (no value loss) if policy only.
            # policy gradient ascent
            if not self.use_loss:
                R , returns = 0, []  #init reward for batch
                # calculate "G" for each turn
                for r in rewards[::-1]:
                    # calculate the discounted value
                    R = r + opts.gamma * R
                    returns.insert(0, R)
                    
                returns = torch.tensor(returns).to(device)
                # baseline
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                
                # calculate losses 
                for log_prob, value, R in zip(actions, values, returns):
                    advantage = R - value.item()
                    # actor loss
                    policy_losses.append(- log_prob * advantage)
                    # critic loss
                    value_losses.append(F.smooth_l1_loss(value, torch.tensor([[[R]]]).to(device)))
                    
                loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
                    
            # policy gradient descent   
            else:
                for log_prob, value, R in zip(actions, values, rewards):
                    policy_losses.append( log_prob * R)
                    value_losses.append( F.smooth_l1_loss(value, torch.tensor([[[R]]]).to(device)))
                #TODO: check gradient descent implementation; use policy only
                #loss = .8 * torch.stack(policy_losses).sum() + .2* torch.stack(value_losses).sum()
                loss = torch.stack(policy_losses).sum() 
                
            # backprop through policy 
            # grad accumulation: 
            if opts.gradient_accumulation_steps > 1:
                loss = loss / opts.gradient_accumulation_steps
            loss.backward()
            
            if step is not None:
                self.steps = step +1
            else:
                self.steps += 1
                
            if (self.steps) % opts.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.policy.zero_grad()
                print()
                print("grad updated ! ")
                print()
    
    def save(self):
        torch.save(self.policy.state_dict(), os.path.join(opts.authenticator_path,'policy.pt'))
    
    def load(self):
        policy_state_dict = torch.load(os.path.join(opts.authenticator_path, 'policy.pt'))
        self.policy.load_state_dict(policy_state_dict)
    
class ACNet(nn.Module):
    def __init__(self, inp_size, action_size, embed_size=128, hidden_size=256, 
                                    num_layers=1, batch_first=True):
        super(ACNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=action_size, embedding_dim=embed_size)
        self.rnn = nn.LSTM(inp_size + embed_size, hidden_size, num_layers=1, batch_first=True)
        # TODO: should actor and critic networks be more than 1 layer?
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action, hidden_state):
        action_emb = self.embedding(action)
        inp = torch.cat([state, action_emb], dim =-1)
        hidden, hidden_out = self.rnn(inp, hidden_state)
        action_logits = self.actor(hidden)#F.softmax(self.actor(hidden), dim=-1)
        score = self.critic(hidden)
        return action_logits, score, hidden_out
    
    def act(self, logits):
        #log_scores = top_k_top_p_filtering(logits[:,-1,:], top_k=self.top_k, top_p=self.top_p)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action
    

class ConvGym(object):
    def __init__(self, clf, i2p, agent, train_set = None, test_set = None,
                 num_convos = 10, epochs = 5, use_loss = True, title = None,
                 weaken_mode=False, k=100, alpha=10, sigma=100,
                 length=8, top_k=10, top_p = .92, max_length=1024 ):
        self.i2p = i2p
        self.clf = clf
        self.agent = agent
        self.k, self.alpha, self.sigma = k, alpha, sigma
        self.epochs, self.weaken_mode, self.banned = epochs, weaken_mode, []
        self.num_convos = len(train_set) if train_set is not None else num_convos
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        self.train_set, self.test_set = train_set, test_set
        self.use_loss, self.title = use_loss, title
        
    def _reset_env(self):
        self.banned = []
    
    def _reset_agents(self, personas = None):
        '''reset agent and user conversational states'''
        if self.agent is not None:
            self.agent.reset_convo()
        else:
            p1_ids = random.sample(self.i2p.keys() - set(self.banned), 5)
            self.banned.extend(p1_ids)
            self.agent = UserSim(i2p, p1_ids, self.top_k, self.top_p, self.max_length)
            
        if personas is not None:
            p2_ids = personas
        else:
            p2_ids = random.sample(self.i2p.keys() - set(self.banned), 5)
            self.banned.extend(p2_ids)
        self.user = UserSim(self.i2p, p2_ids, self.top_k, self.top_p, self.max_length)
        
    def _get_state(self, dialog_hx):#, actions):
        # hx x action -> state
        with torch.no_grad():
            state_conv = model(to_var(flatten(dialog_hx[1::2])).long(), 
                         output_hidden_states=True)[2][24].squeeze(0).mean(0)
            # predicted personality -> 1024 clf state
            state_clf = self.clf.clf(state_conv.unsqueeze(0))[0].squeeze(0)
        #TODO: how to distinguish b/t users transitions?
        #action_state = to_var(np.eye(len(self.agent.i2p))[actions].sum(0))
        #return torch.cat((state_clf, action_state), dim=-1)
        return state_clf
        
    def _calculate_reward(self, state, labels):
        with torch.no_grad():
            pos, neg = self.clf._generate_candidates([labels], self.k)
            loss = self.clf.criterion(state.unsqueeze(0), pos[0], neg[0])
            # thin tailed
            reward = self.alpha * torch.exp(- loss / self.sigma )
            # fat tailed
            #reward = self.alpha * (1 / (1+(loss/self.sigma) ) )
        if self.use_loss:
            return loss
        else:
            return reward.item()

    def _generate_trajectory(self, persona_input = None): 
        '''generate a trajectory of 
            - dialog history
            - reward associated w/ each turn  '''
        self._reset_agents(persona_input)
        msg, rewards, states = None, [],[]
        state = None
        # run for turns, updating rewards at each turn
        for turn in range(self.length):
            # get action
            msg = self.agent(msg, state)
            msg = self.user(msg)
            # get next state
            state = self._get_state(self.user.dialog_history) #, self.agent.persona_ids)
            states.append(state)
            # get reward
            rewards.append(self._calculate_reward(state, self.user.persona_ids))
        self.agent._update_dialog_hx(msg)   # optional, user's hx complete
        dialog_hx = self.user.dialog_history
        states = torch.stack(states) 
        actions = self.agent.actions
        # spot checks
        assert tokenizer.decode(flatten(self.agent.dialog_history)) == tokenizer.decode(flatten(self.user.dialog_history))
        assert len(dialog_hx[1::2]) == len(rewards)
        return dialog_hx, rewards, states, actions
    
    def simulate_convos(self, training = False):
        self._reset_env()
        data = {}
        print("Conducting conversations with training set ...")
        print("="*50)
        total_rewards = [] # the learning curve
        for e in range(self.epochs):
            data[e] = {}
            if self.train_set is not None: random.shuffle(self.train_set)
            # reset epoch stats for metrics
            prec1s, prec5s, rec5s, rec10s = [], [], [], []
            for i in range(self.num_convos):
                if self.train_set is not None:
                    persona_inp = self.train_set[i]
                else: persona_inp = None
                dialog_hx, rewards, states, actions= self._generate_trajectory(persona_inp)
                user_ids = self.user.persona_ids 
                # track rewards 
                if (e==0) and (i==0): running_reward = sum(rewards) * 1.0
                else: running_reward = running_reward * (0.95) + 0.05 * sum(rewards)
                total_rewards.append(running_reward)
                # calculate metrics 
                state = states[-1].view(1,-1).detach()
                prec1, prec5, rec5, rec10 = clf.evaluate(state, [user_ids])
                prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
                
                # Q function is off-policy, inputs are single actions rather than logits.
                data[e][i] = {'trajectory': dialog_hx, 'states': to_var(states),
                           'user_ids': self.user.persona_ids, 'agent_ids': self.agent.action_history ,
                           #'actions': actions, 
                           'metrics': {'rewards': rewards, 'prec1': prec1, 'prec5': prec5, 
                                       'rec5': rec5, 'rec10': rec10}}
                
                print("[epoch %d, iter: %d / %d] rewards: %.2f" %(e+1, i+1, self.num_convos, running_reward))
                print("user ids: ",self.user.persona_ids)
                #print("agent ids:", self.agent.persona_ids)
                print("agent actions:", self.agent.action_history)
                #print("actions: ", actions)
                print("-"*50)
                if training:
                    episode = (states, actions, rewards)
                    self.agent.update(episode,i)
                # logging
                if (i+1) % 10 == 0:
                    plot_rewards(total_rewards, True, self.title)
                    mean_prec1, mean_prec5, mean_rec5, mean_rec10 = list(map(np.mean, (prec1s, prec5s, rec5s, rec10s)))
                    plot_ir_metrics( mean_prec1, mean_prec5, mean_rec5, mean_rec10, True, self.title)
                
        data['summary'] = {'total_rewards': total_rewards, 'prec1': prec1s, 'prec5': prec5s,
                           'rec5': rec5s, 'rec10': rec10s}
        return data
 
    def eval_convs(self):
        self._reset_env()
        data = {}
        print("Conducting conversations on test set ...")
        print("="*50)
        for i, persona_inp in enumerate(self.test_set):
            dialog_hx, rewards, states, actions= self._generate_trajectory(persona_inp)
            # track rewards 
            total_rewards.append(sum(rewards))
            # calculate metrics 
            state = states[-1].view(1,-1).detach()
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
    
    #TODO: __call__ method that generates (data, scores) for given num_convos

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
    with open(os.path.join(opts.authenticator_path, 'action_qs'), 'rb') as f: action_qs = pickle.load(f)
    with open(os.path.join(opts.authenticator_path,'action_descr'), 'rb') as f: actions_ps = pickle.load(f)
    # persona_actions = dict([(k+1,v) for k,v in enumerate(tr_personas) ])
    # persona_actions[0] = [0]
    # agent = Agent(i2p, use_loss=True, persona_actions=persona_actions)
    # title = "1283_logp_loss"
    
    # gym = ConvGym(clf, i2p, agent, use_loss=True, title=title ,
    #               train_set = tr_personas, test_set = te_personas, epochs=5, k=100)
    # data = gym.simulate_convos(True)
    # # saving stats
    # with open(os.path.join(opts.authenticator_path, 'stats_%s'%title), 'wb') as f: pickle.dump(data, f)
    
'''
#### SCRATCH WORK #### 
class OLDConvGym(object):
    def __init__(self, clf, i2p, p2_ids, weaken_mode=False,
                 length=8, top_k=10, top_p=.92, max_length=1024):
        self.i2p = i2p
        self.clf = clf
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        self.NA_token = ['Not Available.'] + [tokenizer.eos_token]
        self.p1 = self.NA_token
        self.p1_ids, self.p2_ids = [], p2_ids
        if not weaken_mode: 
            self.p2 = [self.i2p[pi] for pi in self.p2_ids] 
        else: 
            self.p2 = self.NA_token
        self.reset_convo()
        
    
    def _env_transition(self):
        new_inp = []
        with torch.no_grad():
            while (tokenizer.eos_token_id not in new_inp) and (self.trg_len + len(new_inp) < self.max_length):
                # fast decode
                outp, self.past_trg = model(self.trg_inp, past=self.past_trg)
                # top-k and top-p sampling
                log_scores = top_k_top_p_filtering(outp[:,-1,:], top_k=self.top_k, top_p=self.top_p)
                probs = F.softmax(log_scores, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(1)
                # update list of tokens for next round
                new_inp += token.tolist()
                self.trg_inp = token.unsqueeze(0)
                self.trg_len+=1
        self.dialog_history.append(new_inp)
        #self.src_inp = to_var(torch.tensor([new_inp]))
        self.turn+=1
        
    def _agent_transition(self, action):
        # re-condition on new persona
        self.p1_ids.append(action)
        self.p1 = [self.i2p[pi] for pi in self.p1_ids][-5:]
        self.src_inp = tokenizer.encode(''.join(self.p1_descr + self.NA_token + self.sep_tok + self.p2_descr + self.p1 + self.sep_tok + ['<|start|>']))
        self.src_inp += flatten(self.dialog_history)
        self.src_inp, self.src_len, self.past_src = to_var(torch.tensor([self.src_inp])), len(self.src_inp), None
        # generate question, conditioning on new persona
        new_inp = []
        with torch.no_grad():
            while (tokenizer.eos_token_id not in new_inp) and (self.src_len + len(new_inp) < self.max_length):
                outp, self.past_src = model(self.src_inp, past=self.past_src)
                # top k sampling          
                log_scores = top_k_top_p_filtering(outp[:,-1,:], top_k=self.top_k, top_p=self.top_p)
                probs = F.softmax(log_scores, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(1)
                # update tokens for next output
                new_inp += token.tolist()
                self.src_inp = token.unsqueeze(0)
                self.src_len+=1
        self.dialog_history.append(new_inp)
        self.trg_inp = to_var(torch.tensor([new_inp]))

    def _update_state(self):
        with torch.no_grad():
            # dialog history -> 1024 convo state
            state_conv = model(to_var(flatten(self.dialog_history[1::2])).long(), 
                         output_hidden_states=True)[2][24].squeeze(0).mean(0)
            # predicted personality -> 1024 clf state
            state_clf = self.clf.clf(state_convo.unsqueeze(0))[0]
        self.state = torch.cat([state_conv, state_clf], dim=-1)
    
    def get_personas(self):
        return {'p1': self.p1, 'p2': self.p2, 'p1_ids': self.p1_ids, 'p2_ids': self.p2_ids}
    
    def get_state(self, numpy=False):
        return to_data(self.state) if numpy else self.state
    
    
    def reset_convo(self):
        # reset personas, pasts and dialog history
        self.p1_descr, self.p2_descr, self.sep_tok = ['<|p1|>']+ ['person 1: '], ['<|p2|>']+ ['person 2: '],  [tokenizer.sep_token]
        greet, eos_tok = ["hi! how's it going?"], [tokenizer.eos_token] # the intial prompt
        self.src_inp = tokenizer.encode(''.join(self.p1_descr + self.NA_token + self.sep_tok + self.p2_descr + self.p1 +self.sep_tok ), return_tensors='pt').to(device)
        self.trg_inp = tokenizer.encode(''.join(self.p1_descr + self.NA_token + self.sep_tok + self.p2_descr + self.p2 +self.sep_tok + ['<|start|>']), return_tensors='pt').to(device)
        # use PAST (src and trg) to keep track of (a) gradients, (b) history for separate decoding
        self.past_src, self.past_trg, self.src_len, self.trg_len = None, None, self.src_inp.size(1), self.trg_inp.size(1)
        self.dialog_history, self.turn = [], 0
        
        # turn 0: greet and initiate history
        # agent-side
        new_inp = tokenizer.encode(''.join(greet + eos_tok))
        self.dialog_history.append(new_inp)
        # env-side
        self.trg_inp = torch.cat([trg_inp, to_var(torch.tensor([new_inp]))], dim=-1)  
        self._env_transition()
        
'''