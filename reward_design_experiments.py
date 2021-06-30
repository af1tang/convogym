#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:29:53 2020

@author: af1tang
"""
import torch, os, pickle, random, numpy as np
import torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## utils ##
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

### reward shaping ### 
def exp_tails():
    def standardize(x): return (x - np.mean(x))/(np.std(x) + 1e-5)
    def r_func(x, a=10,b=.5, c=.5): return a/(1+(1+ b*x)**c)
    def e_func(x,a=10, b=1e2): return a* np.exp(-x / b)
    # e.g., x as range of possible losses
    x = np.array([1., 5., 10., 20., 30., 40., 50., 60., 70., 80., 100., 150., 200., 250., 300.])
    thin_tails = [(x[i], 8.*x[i], j) for i,j in enumerate(standardize( 8.* e_func(x, a=1, b=1e2) )) ]
    fat_tails = [(x[i], 8.*x[i], j) for i,j in enumerate(standardize( 8.* r_func(x, a=1, b=1, c=1e-1) )) ]
    
    # visualization
    x = np.arange(12, 75)
    y = 8*r_func(x, a=1, b=1, c=1e-1)
    plt.plot(x,y)


### entropy exp ###
def entropy_exp():
    w = torch.autograd.Variable(torch.tensor([[5., 1.0, 3.0, 2.0, .01]]), requires_grad=True)
    
    for i in range(20):
        w.grad = None
        p = F.softmax(w,-1)
        print(p.detach())
        
        m = Categorical(p)
        entropy = m.entropy()
        H = -2.*entropy
        H.backward()
        
        print(entropy.item())
        w.data.add_(w.grad, alpha=-1)

### log p exp ###
def surrloss_exp():
    def reward_func(x):
        return np.exp( - np.linalg.norm(np.array([ x-2])) )
    
    data = {}; R_ = []
    
    w = torch.autograd.Variable(torch.tensor([[.2, .2, .2, .2, .2]]), requires_grad=True)
    log_old = torch.log(torch.tensor([[.2]]))
    
    for i in range(10):
        w.grad = None
        p = F.softmax(w, -1)
        
        # calculate distribution and sample
        m = Categorical(p)
        action = m.sample()
        log_p = m.log_prob(action)
        entropy = m.entropy()
        
        R = reward_func(action)
        # normalize reward 
        R_.append(R)
        R = (R-np.mean(R_))/(np.std(R_) + 1e-5)
        # importance sampling
        ratio = torch.exp(log_p - log_old)
        loss = - R * ratio 
        H = - 0.01 * entropy
        # backward
        total_loss = loss + H
        total_loss.sum().backward()
        print( loss.item(), entropy.item())
        
        w.data.add_(w.grad, alpha=-1 * 1e-2)
        data[i] = {'grad': w.grad.detach(), 'w': w.detach(), 'p': p.detach(),
                   'R': R, 'action': action.item() }
        
    actions = [data[i]['action'] for i in range(10)]
    rewards = [data[i]['R'] for i in range(10)]
    grads = [data[i]['grad'] for i in range(10)]

#### full exp's ###
def exp_1():
    X,y = make_classification(n_samples=int(1e5), n_features=1024, n_informative=512, 
                              n_redundant = 256, n_classes=20)
    X,y = to_var(X), to_var(y)
    dataset = TensorDataset(X,y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #data_iter = iter(dataloader)
    
    def loss_func(y_true, y_pred):
        dist = torch.abs(y_true - y_pred)
        return torch.exp( - dist)
    
    model = nn.Sequential( 
                        nn.Linear(1024, 256),
                        nn.Tanh(), nn.Dropout(.2), # tanh for probablity outputs
                        nn.Linear(256, 256), 
                        nn.Tanh(), nn.Dropout(.2), # tanh for prob outputs
                        nn.Linear(256, 20),
            ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    
    data = {}; R_ = []
    max_ent = torch.tensor([[2.]*20] * 32).to(device)
    for i, batch in enumerate(dataloader): # 25 iters for 5 classes
        # batch
        xx,yy = batch; xx, yy = map(to_var, (xx,yy))
        # forward pass
        with torch.no_grad():
            logits = model(xx)
            p = F.softmax(logits, -1)
    
            log_p_old = torch.log(p)
        
            # calculate distribution and sample
            m = Categorical(p)
            action = m.sample()
            log_p = m.log_prob(action)
            entropy = m.entropy()
            
            # calculate rewards
            raw_R = loss_func(yy, action)
            # normalize reward 
            R_.extend(raw_R.tolist())
            R = (raw_R-np.mean(R_))/(np.std(R_) + 1e-5)
        # # importance sampling
        for j in range(5):
            logits = model(xx)
            p = F.softmax(logits, -1)
            
            m = Categorical(p)
            log_p = m.log_prob(action)
            entropy = m.entropy()
            
            old_log_p = torch.stack([log_p_old[i][a] for i,a in enumerate(action)])
            ratio = torch.exp(log_p - old_log_p)
            loss1 = R * ratio 
            # # ppo constraint on kl
            loss2 = torch.clamp(ratio, .98, 1.02) * to_var(R) # * curr_log_probs
            loss = - torch.min(loss1, loss2)
            # log p
            #loss = - R * log_p
            H = (1e-4 * .9**j) * torch.sum((logits - max_ent)**2) # 1e-4, .99 for k =5 classes
            #H = - (.05 * .95**i) * entropy.sum()
            # backward
            optimizer.zero_grad()
            total_loss = loss + H
            total_loss.mean().backward()
            #nn.utils.clip_grad_norm_(model.parameters(), .25)

            #loss = F.cross_entropy(logits, to_var(y).long())
            #print(loss.item())
            #loss.backward()
            optimizer.step()
        
        #log_old = log_p.detach()
        print( loss.detach().mean(), entropy.detach().mean(), 
                  torch.eq(yy, p.argmax(1)).sum())
        
        data[i] = {'logits': logits.detach(), 'p': p.detach(),
                   'R': raw_R.mean().item(), 'action': action }
    
    actions = [data[i]['action'] for i in range(10)]
    rewards = [data[i]['R'] for i in range(10)]
    #grads = [data[i]['grad'] for i in range(10)]
    
