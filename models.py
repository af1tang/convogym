#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 21:20:40 2021

@author: af1tang
"""
import random, numpy as np
import torch, torch.nn as nn
from _configs import opts
from utils._device import to_var, to_device
from utils._reshape import chunker
from utils._visualization import plot_losses

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
    
class ID(nn.Module):
    '''identifier network '''
    def __init__(self, inp_size=1024, hidden_size=1024, dropout=.2):
        super(ID, self).__init__()
        self.x_linear1 = nn.Linear(inp_size, hidden_size)
        self.act = nn.Tanh()
        self.x_linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.p_linear1 = nn.Linear(inp_size, hidden_size)
        self.p_linear2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self,x):
        x = self.x_linear2(self.dropout(self.act(self.x_linear1(x))))
        return x
            
    
class Identifier(nn.Module):
    '''reward function'''
    def __init__(self, clf, p2v):
        super().__init__()
        self.clf = to_device(clf)
        self.criterion = ranking_loss()
        self.p2v = p2v
        
    def forward(self, inp, labels, k=20):
        pos, neg = self._generate_candidates(labels, k)
        anchor = self.clf(inp)
        loss = sum([self.criterion(anchor[i].unsqueeze(0), pos[i], neg[i]) for i in range(len(anchor))])
        return loss
    
    def score(self, inp, positives, candidates=20):
        with torch.no_grad():
            outp = self.clf(inp) 
            # bs x samples x dim
            negatives = [random.sample(list(self.p2v.keys() - set(pos_set)), candidates) for pos_set in positives]
            pos_targets = [torch.stack([to_var(self.p2v[idx]) for idx in yi]) for yi in positives]
            neg_targets = [torch.stack([to_var(self.p2v[idx]) for idx in yi]) for yi in negatives]
            prec1 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=1)
                     for i in range(len(outp))]
            prec5 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=5)
                     for i in range(len(outp))]
            rec5 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], 
                    neg_targets[i], at_k=5, use_recall=True) for i in range(len(outp))]
            rec10 = [self._score_triplet(outp[i].unsqueeze(0), pos_targets[i], 
                    neg_targets[i], at_k=10, use_recall=True) for i in range(len(outp))]

        return prec1, prec5, rec5, rec10
        
    def _generate_candidates(self, positives, k):
        negatives = [random.sample( list(self.p2v.keys() - set(pos_set)), k) for pos_set in positives]
        pos_samples = [ torch.stack([to_var(self.p2v[p]) for p in persona]) for persona in positives]
        neg_samples = [ torch.stack([to_var(self.p2v[p]) for p in persona]) for persona in negatives]
        return pos_samples, neg_samples
    
    def _score_triplet(self, x, pos,neg, at_k, use_recall=False):
        dist = nn.CosineSimilarity()
        scores = []
        for k,v in enumerate(pos):
            scores.append((k,1, dist(x,v.unsqueeze(0)).item()))
        for k,v in enumerate(neg):
            scores.append((k+len(pos), 0, dist(x,v.unsqueeze(0)).item()))
        scores = sorted(scores, key = lambda x: x[-1], reverse=True)
        picks = [s[1] for s in scores[0:at_k]]
        # precision or recall @ k
        score = sum(picks) / len(pos) if use_recall else sum(picks) / len(picks)
        return score
    
    def _get_knns(self, x, banned, at_k=5, reverse_picks=False, both_modes=False):
        ''' x: (1 x dim) '''
        dist = nn.CosineSimilarity()
        scores = []
        for k,v in self.p2v.items():
            scores.append((k, dist(x,v.unsqueeze(0)).item()))
        scores = sorted(scores, key = lambda x: x[-1], reverse= not reverse_picks)#True)
        scores = [s for s in scores if s[0] not in banned]
        if both_modes:
            picks = [s[0] for s in scores[:at_k]] + [s[0] for s in scores[-1*at_k:]]
        picks = [s[0] for s in scores[:at_k]]
        return picks
    
    def load_params(self, load_path):
        try: 
            state_dict = torch.load(load_path)
            self.clf.load_state_dict(state_dict)
            return True
        except:
            print('''No existing parameters for identifier model found. ''')
            return False

    
    def fit(self, train_data, save_path,
                epochs=1, lr=1e-3, bs=32, k=20, 
                logging_steps=10, save_steps=50):
        print()
        print("Training identifier ... ")
        print()
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)
        # training
        stats, iters, tr_loss, logging_loss = {}, 0, 0.0, 0.0
        self.clf.train()
        self.clf.zero_grad()
        iters = 0
        for epoch in range(epochs):
            random.shuffle(train_data)
            for minibatch in chunker(train_data, bs):
                # batching
                x1,x2,p1,p2 = zip(*minibatch)
                x1, x2 = map(to_var, (x1,x2))
                # forward
                loss1 = self.forward(x1, labels=p1, k=k)
                loss2= self.forward(x2, labels=p2, k=k)
                loss = loss1+loss2
                # backward
                loss.backward()
                tr_loss += loss.item()
                #if (step+1)% opts.gradient_accumulation_steps == 0:
                optimizer.step()
                self.clf.zero_grad()
                iters +=1
                # reporting
                if iters % logging_steps ==0:
                    stats[iters] = {'loss': (tr_loss - logging_loss) / opts.logging_steps}
                    logging_loss = tr_loss
                    print('Epoch: %d | Iter: %d | loss: %.3f ' %( 
                    epoch, iters, stats[iters]['loss']) )
                    
                if iters % save_steps==0:
                    print("Saving stuff ... ")
                    state_dict = self.clf.state_dict()
                    torch.save(state_dict, save_path)
                    plot_losses(stats, title='loss' )
                    print("Done.")
                    
    def evaluate(self, test_data, bs=32, k=20):
        # eval
        print("-"*50)
        print("Evaluating identifier ... ")
        eval_stats = {'prec@1':[], 'prec@5':[], 'rec@5':[], 'rec@10':[]}
        self.clf.eval()
        for minibatch in chunker(test_data, bs):
            # batching
            x1,x2,p1,p2 = zip(*minibatch)
            x1, x2 = map(to_var, (x1,x2))
            for xx,yy in [(x1,p1), (x2,p2)]:
                prec1, prec5, rec5, rec10 = self.score(xx, yy, candidates=k)
                eval_stats['prec@1'].extend(prec1)
                eval_stats['prec@5'].extend(prec5)
                eval_stats['rec@5'].extend(rec5)
                eval_stats['rec@10'].extend(rec10)
        print("prec1: %.1f | prec5: %.1f | rec5: %.1f | rec10: %.1f" % ( 
                                100*np.mean(eval_stats['prec@1']), 100*np.mean(eval_stats['prec@5']), 
                                100*np.mean(eval_stats['rec@5']), 100*np.mean(eval_stats['rec@10'])) )
        print("-"*50)
        return eval_stats
