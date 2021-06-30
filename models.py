#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:18:37 2020

@author: af1tang
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical
import random
from load_configs import model, tokenizer, opts, device, p1_tok, p2_tok, start_tok
from utils import *

### authentication models ###
class ACNet(nn.Module):
    def __init__(self, inp_size, action_size, hidden_size=256, dropout=.1):
        super(ACNet, self).__init__()
        self.actor = nn.Sequential( 
                        nn.Linear(inp_size, hidden_size),
                        nn.Tanh(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, action_size),
            )
        self.critic = nn.Sequential(
                        nn.Linear(inp_size, hidden_size),
                        nn.Tanh(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, 1),
            )
    
    def act(self, state, context, episode):
        '''input:  state: (1 x 1024 ) tensor, 
                    context: (1 x 1024) tensor,
                    episode: python dict with keys states, actions, log_probs
        output: sampled action'''
        #with torch.no_grad():
        logits = self.actor(state) # (1, 20)
        action_probs = F.softmax(logits, dim=-1)
        m = Categorical(action_probs) # softmax layer -> pmf, (1, 20)
        sampled_action = m.sample() # i ~ pmf, (1,)
        log_probs = m.log_prob(sampled_action) # log p @ index i, (1,)
        
        episode['states'].append(state.squeeze(0))
        episode['actions'].append(sampled_action.item())
        episode['log_probs'].append(log_probs.squeeze(0))
        episode['contexts'].append(context.squeeze(0))
        episode['logits'].append(logits.squeeze(0))
        return sampled_action.item()
    
    
    def evaluate(self, states, actions):
        '''states: bs x 1024, actions: bs x 1 '''
        value = self.critic(states) # (bs, 1)
        
        curr_probs = F.softmax(self.actor(states),-1)
        m_curr = Categorical(curr_probs)
        log_probs = m_curr.log_prob(actions)
        entropy = m_curr.entropy()
        return log_probs, value.squeeze(), entropy
    
class MetaNet(nn.Module):
    def __init__(self, inp_size, action_size, hidden_size=512, dropout=.1):
        super(MetaNet, self).__init__()
        self.policy = nn.Sequential( 
                        nn.Linear(inp_size, hidden_size),
                        nn.Tanh(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh(), nn.Dropout(dropout),
                        nn.Linear(hidden_size, action_size),
            )
        
    def forward(self, states, contexts):
        '''input:  state: (bs x 1024) dialog embedding, 
            context: (bs x 1024) persona embedding
        output:  probability distribution over actions (bs x action_size) '''
        inp = torch.cat([states, contexts], dim=-1)
        outp = self.policy(inp)
        return outp
    
    def act(self, state, context, episode=None):
        '''input:  state: (1 x 1024 ) tensor, 
                    episode: python dict with keys states, actions, log_probs
        output: sampled action'''
        with torch.no_grad():
            logits = self.forward(state, context) # (1, 20)
            action_probs = F.softmax(logits, -1)
            m = Categorical(action_probs) # softmax layer -> pmf, (1, 20)
            sampled_action = m.sample() # i ~ pmf, (1,)
            log_probs = m.log_prob(sampled_action) # log p @ index i, (1,)
            
            if episode is not None:
                episode['states'].append(state)
                episode['actions'].append(to_data(sampled_action))
                episode['log_probs'].append(log_probs)
                episode['logits'].append(logits)
                episode['contexts'].append(context)
        return sampled_action.item()
    
    
### identification models ####
class LSTM(nn.Module):
    def __init__(self, inp_size, hidden_size, outp_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(inp_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, outp_size)
        
    def forward(self, x):
        hidden, _ = self.rnn(x)
        outp = self.linear(hidden[:, -1, :])
        return outp
    
class MLP(nn.Module):
    def __init__(self, inp_size, hidden_size, outp_size, dropout=.2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(inp_size, hidden_size)
        self.act = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, outp_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        outp = self.linear2(self.dropout( self.act(self.linear1(x)) ))
        return outp
    
class ID(nn.Module):
    def __init__(self, inp_size, hidden_size, outp_size, dropout=.2):
        super(ID, self).__init__()
        self.x_linear1 = nn.Linear(inp_size, hidden_size)
        self.act = nn.Tanh()
        self.x_linear2 = nn.Linear(hidden_size, outp_size)
        self.dropout = nn.Dropout(dropout)
        
        self.p_linear1 = nn.Linear(inp_size, hidden_size)
        self.p_linear2 = nn.Linear(hidden_size, outp_size)
        
    def forward(self,x, pos=None, neg=None):
        x = self.x_linear2(self.dropout(self.act(self.x_linear1(x))))
        if opts.use_cca:
            pos = [self.p_linear2(self.dropout(self.act(self.p_linear1(pi)))) for pi in pos] if pos is not None else None
            neg = [self.p_linear2(self.dropout(self.act(self.p_linear1(ni)))) for ni in neg] if neg is not None else None
        return x, pos, neg
    
class CLF(nn.Module):
    def __init__(self, i2v, hidden_size=1024, outp_size=6737, mode = "BOW1", zsl=False):
        super().__init__()
        self.hidden_size, self.outp_size = hidden_size, hidden_size if zsl else outp_size
        self.mode, self.zsl = mode, zsl
        self.initialize_clf()
        self.i2v = i2v
        
    def initialize_clf(self):
        if self.mode == 'BOW1': 
            self.clf = MLP(inp_size = 300, hidden_size = 300, outp_size = self.outp_size)
            self.criterion = multilabel_loss() if not self.zsl else F.mse_loss
        elif self.mode == 'BOW2':
            self.clf = MLP(inp_size = 768, hidden_size=self.hidden_size, outp_size = self.outp_size )
            self.criterion = multilabel_loss() if not self.zsl else F.mse_loss
        elif self.mode == 'LSTM':
            self.clf = LSTM(inp_size = 300, hidden_size = 600, outp_size = self.outp_size)
            self.criterion = multilabel_loss() if not self.zsl else F.mse_loss
        elif self.mode == 'BERT':
            self.clf = MLP(inp_size = 768, hidden_size=self.hidden_size, outp_size = self.outp_size )
            self.criterion = multilabel_loss() if not self.zsl else F.mse_loss
        elif self.mode == 'GPT':
            self.clf = MLP(inp_size = 1024, hidden_size=self.hidden_size, outp_size = self.outp_size)
            self.criterion = multilabel_loss() if not self.zsl else F.mse_loss
        elif self.mode == 'ID':
            self.clf = ID(inp_size = 1024, hidden_size=self.hidden_size, outp_size = 1024)
            self.criterion = ranking_loss() 
        self.clf.to(device)
        
    def forward(self, inp, labels, k=100):
        if self.mode != 'ID':
            outp = self.clf(inp)
            if self.zsl:
                #indices = [(yy==1).nonzero().view(-1).tolist() for yy in labels]
                indices = labels
                targets = torch.stack([torch.stack([to_var(self.i2v[idx]) for idx in yi]).mean(0) for yi in indices])
                loss = self.criterion(outp,targets, reduction='sum')
            else:
                loss = self.criterion(outp, labels.float())
        else:
            pos_samples, neg_samples = self._generate_candidates(labels, k)
            anchor, pos, neg = self.clf(inp, pos=pos_samples, neg=neg_samples)
            if opts.use_mtl:
                r_loss = sum([self.criterion(anchor[i].unsqueeze(0), pos[i], neg[i]) for i in range(len(anchor))])
                m_loss = F.mse_loss(anchor, torch.stack([pi.mean(0) for pi in pos]), reduction='sum')
                loss = r_loss + 0.5 * m_loss
            else:
                loss = sum([self.criterion(anchor[i].unsqueeze(0), pos[i], neg[i]) for i in range(len(anchor))])
        return loss
    
    def evaluate(self, inp, labels, candidates=20):
        with torch.no_grad():
            prec1,prec10,rec5 = [],[],[]
            outp = self.clf(inp) if self.mode != "ID" else self.clf(inp)[0] # bs x dim
            if self.zsl:
                # bs x samples x dim
                positives = labels #[(yy==1).nonzero().view(-1).tolist() for yy in labels] 
                negatives = [random.sample(list(self.i2v.keys() - set(pos_set)), candidates) for pos_set in positives]
                pos_targets = [torch.stack([to_var(self.i2v[idx]) for idx in yi]) for yi in positives]
                neg_targets = [torch.stack([to_var(self.i2v[idx]) for idx in yi]) for yi in negatives]
                prec1 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=1)
                         for i in range(len(outp))]
                prec5 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=5)
                         for i in range(len(outp))]
                rec5 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], 
                        neg_targets[i], at_k=5, use_recall=True) for i in range(len(outp))]
                rec10 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], 
                        neg_targets[i], at_k=10, use_recall=True) for i in range(len(outp))]
            else:
                picks = outp.topk(10)[1]
                positives = labels #[(yy==1).nonzero().view(-1).tolist() for yy in labels] 
                hits = [[(lambda k,j: 1 if picks[j][k] in positives[j] else 0)(k,j)
                         for k in range(len(picks[j]))] for j in range(len(picks))]
                prec1 = [sum(hits[k][0]) / 1.0 for k in range(len(hits))]
                prec5 = [sum(hits[k][:5]) / 5.0 for k in range(len(hits))]
                rec5 = [sum(hits[k][:5]) / len(positives[k]) for k in range(len(hits))]
                rec10 = [sum(hits[k][:10]) / len(positives[k]) for k in range(len(hits))]
        return prec1, prec5, rec5, rec10
        
    def _generate_candidates(self, labels,k):
        positives = labels #[(yy==1).nonzero().view(-1).tolist() for yy in labels]
        negatives = [random.sample( list(self.i2v.keys() - set(pos_set)), k) for pos_set in positives]
        pos_samples = [ torch.stack([self.i2v[p] for p in persona]) for persona in positives]
        neg_samples = [ torch.stack([self.i2v[p] for p in persona]) for persona in negatives]
        return pos_samples, neg_samples
    
    def _score_triplet(self, x, pos,neg, at_k, use_recall=False):
        #dist = nn.PairwiseDistance(p=2)
        dist = nn.CosineSimilarity()
        scores = []
        for k,v in enumerate(pos):
            scores.append((k,1, dist(x,v.unsqueeze(0)).item()))
        for k,v in enumerate(neg):
            scores.append((k+len(pos), 0, dist(x,v.unsqueeze(0)).item()))
        scores = sorted(scores, key = lambda x: x[-1], reverse=True)
        picks = [s[1] for s in scores[0:at_k]]
        # precision @ k
        #score = sum(picks) / len(picks)
        # recall @ k
        score = sum(picks) / len(pos) if use_recall else sum(picks) / len(picks)
        return score
    
    def _get_knns(self, x, banned, at_k=5, reverse_picks=False, both_modes=False):
        ''' x: (1 x dim) '''
        dist = nn.CosineSimilarity()
        scores = []
        for k,v in self.i2v.items():
            scores.append((k, dist(x,v.unsqueeze(0)).item()))
        scores = sorted(scores, key = lambda x: x[-1], reverse= not reverse_picks)#True)
        scores = [s for s in scores if s[0] not in banned]
        if both_modes:
            picks = [s[0] for s in scores[:at_k]] + [s[0] for s in scores[-1*at_k:]]
        picks = [s[0] for s in scores[:at_k]]
        return picks

## Loss Functions ##
class multilabel_loss(nn.Module):
    def __init__(self, pos_weight=torch.tensor(2500.0)):
        super(multilabel_loss, self).__init__()
        self.pos_weight = pos_weight
    def forward(self,x,y):
        #y_hot = torch.stack([F.one_hot(y[i], self.config.vocab_size).max(0)[0] for i in range(y.size(0))]).float()
        #for idx in [self.config.pad_tok, self.config.eos_tok, self.config.sos_tok]:
        #    y_hot[:, idx] = 0
        return F.binary_cross_entropy_with_logits(x, y, pos_weight=self.pos_weight)
    
class ranking_loss(nn.Module):
    def __init__(self, swap=False, margin=1.0, reduction='sum'):
        super(ranking_loss, self).__init__()
        self.ranking_loss = nn.TripletMarginLoss(margin=margin, swap=swap, reduction=reduction)
                
    def forward(self, anchor, positive_samples, negative_samples, verbose=False):
        '''anchor: (bs x dim) e.g., 1 x 2400
        pos: (# personas x dim) e.g., 5 x 2400
        neg: (# candidates x dim) e.g., 100k x 2400'''
        num_pos, num_neg = positive_samples.size(0), negative_samples.size(0)
        total_loss = 0.
        count = 0
        for i in range(0, num_neg, num_pos):
            if positive_samples.size(0) == negative_samples[i:i+num_pos].size(0):
                total_loss = total_loss + self.ranking_loss(anchor, positive_samples, negative_samples[i:i+num_pos])
                count +=1
        if verbose: print ('ranked %d rounds of candidates (5 per round).' % (count))
        return total_loss