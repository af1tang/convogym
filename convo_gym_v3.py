#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:06:52 2020
    
v3.0 optimization:
    - PPO, clipping
    - importance sampling (off-policy)
    - database creation
    - max entropy regularization

reward shaping: 
    - get rid of "use loss"
    - (gym) track EMA and EMV on R, alpha = 1/(n+1), n = num. trajectories OR just raw R's
    - standardize returns (R - R.mean()) / (R.std() + 1e-5) over batch trajectories
    - update load_configs: clip, ppo_iters, gamma
    - report running Rt, loss@t, metrics@t

@author: af1tang
"""
import torch, os, pickle, random, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
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
        self.dialog_history, self.turn = [], 0
        
    def _update_persona(self, action):
        raise NotImplementedError("Update agent not specified for this user.")
        
    def _on_episode_end(self, last_msg, last_reward):
        self._update_dialog_hx(last_msg)
        return [], [], []
        
    
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
                 inp_size=1024, hidden_size=512, dropout=.2,
                 top_k=10, top_p = .92, max_length=1024, reward_shaping = False):
        super(Agent, self).__init__(i2p, persona_ids)

        self.i2p = dict([(k,v + tokenizer.eos_token) for k,v in i2p.items()])
        self.inp_size = inp_size
        self.persona_actions = persona_actions
        self.reward_shaping = reward_shaping
        action_size = len(self.persona_actions) if self.persona_actions is not None else len(self.i2p)
        # policy networks: old and new
        self.policy = ACNet(inp_size=inp_size, action_size=action_size, 
                            hidden_size=hidden_size, dropout=dropout).to(device)
        self.old_policy = ACNet(inp_size=inp_size, action_size=action_size,
                            hidden_size=hidden_size, dropout=dropout).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
    
    def _update_persona(self, action):
        if action is not None:
            #self.persona_ids.append(action)
            #TODO: persona id to list of persona traits
            self.persona_ids = self.persona_actions[action] if self.persona_actions is not None else [action]
            self.p2 = [self.i2p[pi] for pi in self.persona_ids]
            
    def __call__(self, inp, state=None, reward=None):
        # update state representation
        if state is None: 
            #TODO: better state initialization?
            state = to_var(torch.zeros((1,self.inp_size)))
        else: 
            if len(state.shape) < 2: state = state.unsqueeze(0)
            assert len(state.shape) == 2
            state = to_var(state)
        # track reward if not first step
        if reward is not None:
            self.episode['rewards'].append(reward)
            
        # act and update episodic memory
        # NOTE: currently acting w/ OLD policy 
        sampled_action = self.old_policy.act(state, self.episode)
        # update persona and action history
        self._update_persona(sampled_action)
        return self.step(inp)
    
    def reset_convo(self):
        # reset personas, pasts and dialog history
        self.p1_descr, self.p2_descr, self.sep_tok = ['<|p1|>']+ ['person 1: '], ['<|p2|>']+ ['person 2: '],  [tokenizer.sep_token]
        self.dialog_history, self.turn = [], 0
        self.persona_ids = []
        # episodic memory
        self.episode = {'states':[], 'actions': [], 'log_probs':[], 'rewards': [], 'returns': []}
        
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
    
    #TODO: calculate information gain as return
    def _calculate_shaped_returns(self, rewards):
        '''R - b:  R = future rT, b = past rt '''
        returns = np.ones_like(rewards) * rewards[-1]
        info_gain = [(lambda x: x - rewards[i - 1] if i >0 else 0.)(xx) for i, xx in enumerate(rewards)]
        #returns = np.array([.75 * trajectory_reward + .25* info_gain[i] for i in range(len(info_gain))])
        #Q_to_go = rewards - returns
        shaped_returns = returns + np.array(info_gain) #+ Q_to_go
        return shaped_returns
    
    def _on_episode_end(self, last_msg, last_reward):
        self._update_dialog_hx(last_msg)
        self.episode['rewards'].append(last_reward)
        #TODO: fix returns 
        if self.reward_shaping:
            self.episode['returns'] = self._calculate_shaped_returns(self.episode['rewards'])
        else:
            self.episode['returns'] = self._calculate_sparse_returns(self.episode['rewards'])
        return np.array(self.episode['actions']), np.array(self.episode['log_probs']), np.array(self.episode['returns'])
        
    def update(self, data):
        '''data: python dictionary of states, actions, log_probs, and returns
        calculates the batch rewards (losses) and performs update on actor and critic'''
        
        dataloader = DataLoader(data, batch_size = min(opts.ppo_bs, len(data)) )
        i=0; print(); print("training ... (%d samples) " %(len(data)))
        for _ in range(opts.ppo_iters):
            total_loss = []
            for batch in dataloader:
                states, actions, log_probs, returns = batch['states'], batch['actions'], batch['log_probs'], batch['returns']
                states, actions, log_probs, returns = map(to_var, (states, actions, log_probs, returns))

                # generate actions from NEW policy 
                curr_log_probs, values, entropies = self.policy.evaluate(states, actions.squeeze())
    
                # importance sampling: p/q, p = curr, q = old
                importance_ratios = torch.exp(curr_log_probs - log_probs.squeeze()) 
                
                ### actor loss ###
                #TODO: fix advantages w.r.t modified rewards for PPO
                #advantages = returns - values.detach() # don't backprop through value net for this 
                advantages = returns 
                
                # ppo losses
                # - R * log_p * importance sampling ratio
                loss1 = importance_ratios * advantages # * curr_log_probs
                # ppo constraint on kl
                loss2 = torch.clamp(importance_ratios, 1 - opts.clip, 1 + opts.clip) * advantages # * curr_log_probs
                actor_loss = - torch.min(loss1, loss2)
                
                ### critic loss ###
                # max entropy regularization on values
                critic_loss = .5 * F.smooth_l1_loss(values, returns) - .01 * entropies
                
                # backprop and step
                loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
                # logging
                total_loss.append(loss.mean().item()); i +=1;
            print("    training epoch %d, loss = %.2f"% (_, np.mean(total_loss)) )
                
        # update old policy w/ new params        
        self.policy.zero_grad()
        self.old_policy.load_state_dict(self.policy.state_dict())
        print("updated old policy net -> new policy"); print()
    
    def save(self):
        torch.save(self.old_policy.state_dict(), os.path.join(opts.authenticator_path,'old_policy.pt'))
        torch.save(self.policy.state_dict(), os.path.join(opts.authenticator_path,'policy.pt'))
    
    def load(self):
        policy_state_dict = torch.load(os.path.join(opts.authenticator_path, 'policy.pt'))
        self.policy.load_state_dict(policy_state_dict)
    
class ACNet(nn.Module):
    def __init__(self, inp_size, action_size, hidden_size=256, dropout=.2):
        super(ACNet, self).__init__()
        self.actor = nn.Sequential( 
                        nn.Linear(inp_size, hidden_size),
                        nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, action_size),
                        nn.Softmax(dim=-1)
            )
        
        self.critic = nn.Sequential(
                        nn.Linear(inp_size, hidden_size),
                        nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, 1),
            )
    
    def act(self, state, episode):
        '''input:  state: (1 x 1024 ) tensor, 
                    episode: python dict with keys states, actions, log_probs
        output: sampled action'''
        with torch.no_grad():
            action_probs = self.actor(state) # (1, 20)
            m = Categorical(action_probs) # softmax layer -> pmf, (1, 20)
            sampled_action = m.sample() # i ~ pmf, (1,)
            log_probs = m.log_prob(sampled_action) # log p @ index i, (1,)
            
            episode['states'].append(to_data(state))
            episode['actions'].append(to_data(sampled_action))
            episode['log_probs'].append(to_data(log_probs.detach()))
        return sampled_action.item()
    
    
    def evaluate(self, states, actions):
        '''states: bs x 1024, actions: bs x 1 '''
        value = self.critic(states) # (bs, 1)
        
        curr_probs = self.actor(states)
        m_curr = Categorical(curr_probs)
        log_probs = m_curr.log_prob(actions)
        entropy = m_curr.entropy()
        return log_probs, value.squeeze(), entropy
    
class ConvGym(object):
    def __init__(self, clf, i2p, agent, train_set = None, test_set = None,
                 num_convos = 10, epochs = 5, 
                 fat_tailed = False, title = None,
                 weaken_mode=False, k=100, alpha=1., beta=1., sigma=1.1,
                 length=8, top_k=10, top_p = .92, max_length=1024 ):
        self.i2p = i2p
        self.clf = clf
        self.agent = agent
        self.k, self.alpha, self.sigma, self.beta = k, alpha, sigma, beta
        self.epochs, self.weaken_mode, self.banned = epochs, weaken_mode, []
        self.num_convos = len(train_set) if train_set is not None else num_convos
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        self.train_set, self.test_set = train_set, test_set
        self.fat_tailed, self.title = fat_tailed, title
        self._reset_memory()
        
    def _reset_env(self):
        self.banned = []
    
    def _reset_memory(self):
        self.memory = {'states':[], 'actions': [], 'log_probs': [], 'returns':[]}
    
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
        return state_clf
    
    def _e_func(self, x, alpha=1., beta=None, sigma=1.):
        '''thin-tailed reward func'''
        return alpha * torch.exp( - x / sigma) + 1e-5
        #return - x
    
    def _r_func(self, x, alpha=1., beta=1., sigma=1.):
        '''fat-tailed reward func'''
        return torch.log(alpha / ( 1 + ( 1 + beta*x )**sigma ) + 1e-5)
        
    def _calculate_reward(self, state, labels):
        with torch.no_grad():
            pos, neg = self.clf._generate_candidates([labels], self.k)
            loss = self.clf.criterion(state.unsqueeze(0), pos[0], neg[0])
            # turn loss into a distribution q_t over actions
            reward = self._r_func(loss, self.alpha, self.beta, self.sigma) if self.fat_tailed else self._e_func(loss, self.alpha, self.beta, self.sigma)
            # r_t = log q_t for some "true" distribution q_t 
            # Note: R_t = sum(r_t) = sum(log q_t), so prod(q_t) = Q_t 
            #reward = torch.log(reward)
            return reward.item()

    def _generate_trajectory(self, persona_input = None): 
        '''generate a trajectory of 
            - dialog history
            - reward associated w/ each turn  '''
        self._reset_agents(persona_input)
        msg, rewards, states = None, [],[]
        state, reward = None, None
        # run for turns, updating rewards at each turn
        for turn in range(self.length):
            # get action
            msg = self.agent(msg, state, reward)
            msg = self.user(msg)
            # get next state
            state = self._get_state(self.user.dialog_history) #, self.agent.persona_ids)
            states.append(state)
            # get reward
            reward = self._calculate_reward(state, self.user.persona_ids)
            rewards.append(reward)
            
        dialog_hx = self.user.dialog_history
        states = to_data(torch.stack(states))
        #TODO: in PPO version, actions are detached 
        actions, log_probs, returns = self.agent._on_episode_end(msg, reward)
        # spot checks
        assert tokenizer.decode(flatten(self.agent.dialog_history)) == tokenizer.decode(flatten(self.user.dialog_history))
        assert len(dialog_hx[1::2]) == len(rewards)
        return dialog_hx, rewards, returns, states, actions, log_probs

    def _update_buffer(self, states, actions, log_probs, returns):
        self.memory['states'].extend(states)
        self.memory['actions'].extend(actions)
        self.memory['returns'].extend(returns)
        self.memory['log_probs'].extend(log_probs)
        
    def _normalize_returns(self, returns):
        gae_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-5)
        return gae_returns
    
    def _get_batch(self):
        # memory_size = len(self.memory['states'])
        # if memory_size < self.bs:
        #     indices = list(range(memory_size))

        # else:   
        #     indices = list(range(memory_size))[-10*self.length:] # get last 10 trajectories
        #     size = min(self.bs - len(indices), memory_size - len(indices))
        #     if size > 0:
        #         random_indices = random.sample(set(range(memory_size)) - set(indices), size)
        #         indices.extend(random_indices)
        # random.shuffle(indices)
        # try NO normalization
        normalized_returns = self._normalize_returns( np.array(self.memory['returns'])[indices] )
        batch = dict([(k, {'states': self.memory['states'][v], 
                        'actions': self.memory['actions'][v],
                        'log_probs': self.memory['log_probs'][v],
                        'returns': normalized_returns[k],
                        }) for k,v in enumerate(indices)])
        # batch = {'states': np.array(self.memory['states'])[indices], 
        #          'actions': np.array(self.memory['actions'])[indices],
        #          'log_probs': np.array(self.memory['log_probs'])[indices],
        #          'returns': self._normalize_returns( 
        #                          np.array(self.memory['returns'])[indices] )
        #          }
        return batch
        
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
                dialog_hx, rewards, returns, states, actions, log_probs= self._generate_trajectory(persona_inp)
                user_ids = self.user.persona_ids 
                # update memory
                self._update_buffer(states, actions, log_probs, returns)
                # track rewards 
                if (e==0) and (i==0): running_reward = rewards[-1] 
                else: running_reward = running_reward * .95 + (1-.05) * rewards[-1]
                total_rewards.append(running_reward)
                # calculate metrics 
                state = to_var(states[-1]).view(1,-1)
                prec1, prec5, rec5, rec10 = clf.evaluate(state, [user_ids])
                prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
                
                # Q function is off-policy, inputs are single actions rather than logits.
                data[e][i] = {'trajectory': dialog_hx, #'states': states,
                           'user_ids': self.user.persona_ids, 'agent_ids': actions,
                           #'actions': actions, 
                           'metrics': {'rewards': rewards, 'prec1': prec1, 'prec5': prec5, 
                                       'rec5': rec5, 'rec10': rec10}}

                print("[epoch %d, iter: %d / %d] rewards: %.2f" %(e+1, i+1, self.num_convos, running_reward))
                print("prec@1: %.1f, prec@5: %.1f, rec@5: %.1f, rec@10: %.1f "%(100*np.mean(prec1s), 
                                                                                100*np.mean(prec5s), 
                                                                                100*np.mean(rec5s),
                                                                                100*np.mean(rec10s)))
                print("user ids: ",self.user.persona_ids)
                print("agent actions:", actions.reshape(-1))
                print("-"*50)
                if (i+1) % opts.ppo_memory_size ==0:
                    batch = self._get_batch()
                    self.agent.update(batch)
                    self._reset_memory()
                # logging
                if (i+1) % 10 == 0:
                    plot_rewards(total_rewards, True, self.title)
                    mean_prec1, mean_prec5, mean_rec5, mean_rec10 = list(map(np.mean, (prec1s, prec5s, rec5s, rec10s)))
                    plot_ir_metrics( mean_prec1, mean_prec5, mean_rec5, mean_rec10, True, self.title)
            
            data[e]['prec@1'] = prec1s; data[e]['prec@5'] = prec5s; data[e]['rec@5'] = rec5s; data[e]['rec@10'] = rec10s
        
        data['total_rewards'] = total_rewards
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
    with open(os.path.join(opts.authenticator_path, 'actions_qs.pkl'), 'rb') as f: actions_qs = pickle.load(f)
    with open(os.path.join(opts.authenticator_path,'actions_descr.pkl'), 'rb') as f: actions_ps = pickle.load(f)
    #persona_actions = dict([(k+1,v) for k,v in enumerate(tr_personas) ])
    #persona_actions[0] = [0]
    action_dict = dict([(k,v) for k,v in enumerate(actions_qs)])
    agent = Agent(action_dict, reward_shaping=False)
    title = "20_qs_ppo_thin_sparse"
    
    gym = ConvGym(clf, i2p, agent, title=title, fat_tailed=False,
                   train_set = tr_personas, test_set = te_personas, epochs=5, k=100)
    data = gym.simulate_convos(True)
    # saving stats
    with open(os.path.join(opts.authenticator_path, 'stats_%s'%title), 'wb') as f: pickle.dump(data, f)
    
