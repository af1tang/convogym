#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:30:29 2021
    TODO: 
        new agent
        - do kNN using context for p2 
        - do policy for p1

@author: af1tang
"""
import torch, torch.nn as nn, os, pickle, random
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from load_configs import model, tokenizer, stats, opts, device, create_dir, p1_tok, p2_tok, start_tok, act_tok
from utils import *
from models import *
from agents import *
from swa_utils import *

def convert_dialog_to_IL_data(clf,hx, actions, tr_personas):
    dataset = {}
    indices = list(range(29)) + [28, 29, 30] + list(range(30, 1282))
    print("building dataset ... ")
    for i, (history, action_seq) in enumerate(zip(hx,actions)):        
        assert len(history) == 16
        states, contexts, acts, rewards = [], [], [], []
        for turn in range(0,18,2):
            curr_hx = history[0:turn]
            if len(curr_hx) == 0:
                state, ctx = [0.0]*1025, [0.0]*1025
            else:
                state, ctx = _get_state(clf, curr_hx[1::2])
                # get reward
                reward = _calculate_reward(clf, ctx, tr_personas[indices[i]], k=20)
                state, ctx = state.tolist(), ctx.tolist()
                state.append(turn/2.0); ctx.append(turn /2.0)
                #_, embedding = _get_state(clf, curr_hx[1::2])
                rewards.append(reward)
            if turn // 2 <8:
                a = action_seq[turn //2 ]; acts.append(a)
            states.append(state); contexts.append(ctx)
        # get final reward
        #_, final_emb = _get_state(clf, history[1::2])
        #final_reward = _calculate_reward(clf, final_emb, tr_personas[indices[i]], k=20)
        #rewards.append(final_reward)
        # batching
        next_states = states[1:]; states = states[:-1]
        next_contexts = contexts[1:]; contexts = contexts[:-1]
        next_acts = acts[1:] + [0]
        batch = list(zip(states, contexts, acts, next_states, next_contexts, next_acts, rewards))
        dataset[i] = [list(tuples) for tuples in batch]
        if (i+1)%10 ==0:
            print("[%d / %d] ... " %(i+1, len(hx)))
    print("done.")
    return dataset

def pretrain_policy(dataset, gamma=1.0, num_train_epochs=500):
    policy = nn.Sequential(nn.Linear(2050, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, 11)).cuda()
    # polyak-avg / swa model
    # uncomment if polyak
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                0.99 * averaged_model_parameter + 0.01 * model_parameter
    swa_policy = AveragedModel(policy, avg_fn = ema_avg)
    flat_data = flatten([dataset[i] for i in dataset.keys()])
    dataloader = DataLoader(flat_data, batch_size=64, shuffle=True)
    # init optim and criteria
    t_total = len(dataloader) * num_train_epochs

    optimizer = AdamW(policy.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss()
    huber_loss = nn.SmoothL1Loss()
    stats = {}
    policy.train(); swa_policy.eval()
    for epoch in range(num_train_epochs):
        dataloader = DataLoader(flat_data, batch_size=64, shuffle=True)
        epoch_loss, epoch_q_loss = [], []
        for i, batch in enumerate(dataloader):
            # batching 
            x, c, y, x_next, c_next, y_next, r = batch
            x,c = torch.stack(x, dim=-1).type(torch.cuda.FloatTensor), torch.stack(c, dim=-1).type(torch.cuda.FloatTensor)
            x_next, c_next = torch.stack(x_next, dim=-1).type(torch.cuda.FloatTensor), torch.stack(c_next, dim=-1).type(torch.cuda.FloatTensor)
            y, y_next, r = y.to(device), y_next.to(device), r.type(torch.cuda.FloatTensor)
            xx = torch.cat((x,c), dim=-1)
            xx_next = torch.cat((x_next, c_next), dim=-1)
            # calculate q-values
            with torch.no_grad():
                # use target network to predict q-targets
                q_values = swa_policy(xx_next)
                #q_next = q_values.max(1)[0]
                
                dones = (x[:,-1] >= 7).long()
                #q_targets = r + (1-dones) * gamma * q_next
                q_targets = r + (1-dones) * gamma * q_values[torch.arange(len(y_next)), y_next]
            # fwd and bck
            policy.zero_grad()
            yhat = policy(xx)   # logits
            loss = criterion(yhat, y) # loss on softmax(yhat)
            # index logits by action index
            aux_loss = huber_loss(yhat[torch.arange(len(y)),y], q_targets)
            total_loss = loss + aux_loss
            total_loss.backward()
            optimizer.step()
            #scheduler.step()
            # take swa (polyak avg)
            swa_policy.update_parameters(policy)
            # tracking
            epoch_loss.append(loss.item()); epoch_q_loss.append(aux_loss.item())
            if i % 10 == 0:
                print("epoch: %d | iter: [%d / %d] | loss: %.2f | q_loss: %.2f | lr: %s" %(epoch, i, len(dataloader), loss.item(),
                                                                                           aux_loss.item(), str(scheduler.get_last_lr()[-1])))
        # on epoch end
        stats[epoch] = {'il_loss': np.mean(epoch_loss), 
                        'q_loss': np.mean(epoch_q_loss)}
        plot_losses(stats, title='il_loss' )
        plot_losses(stats, title='q_loss' )
    return policy, swa_policy

def train_policy(i2p, action_dict, clf, dataset, policy, tr_personas, stats=None,
                 gamma=1.0, iters = 3, num_train_epochs=3,
                 top_k = 10, top_p = .92):
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                0.99 * averaged_model_parameter + 0.01 * model_parameter
    swa_policy = AveragedModel(policy, avg_fn = ema_avg)
    swa_policy.eval()
    # init buffers 
    if (opts.imitation_path and os.path.isfile(os.path.join(opts.imitation_path, 'memory_buffer'))):
        with open(os.path.join(opts.imitation_path, 'memory_buffer'), 'rb') as f:
            memory_buffer = pickle.load(f)
            print("Loaded memory buffer.")
    else:
        memory_buffer = flatten([dataset[i] for i in dataset.keys()])
    #memory_buffer = []
    # init env
    agent = UserSim(i2p, None, reverse=True, top_k = top_k, top_p=top_p)
    # init optim and schedulers
    t_total = iters * len(tr_personas) * num_train_epochs
    optimizer = AdamW(policy.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss()
    huber_loss = nn.SmoothL1Loss()
    # tracking
    if (opts.imitation_path and os.path.isfile(os.path.join(opts.imitation_path, "optimizer.pt"))
                                and os.path.isfile(os.path.join(opts.imitation_path, "scheduler.pt")) ):
        # load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(opts.imitation_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(opts.imitation_path, "scheduler.pt")))
        print("Loaded optim and schedulers.")
    # track stats
    if stats is not None:
        prec1s, prec5s, rec5s, rec10s = stats['prec1s'], stats['prec5s'], stats['rec5s'], stats['rec10s']
        dialog_histories, p1s, p2s = stats['hx'], stats['p1s'], stats['p2s']
        global_epoch, global_step = stats['global_epoch'], stats['global_step']
        print("Resuming Training ... ")
    else:
        prec1s, prec5s, rec5s, rec10s = [], [], [], []
        dialog_histories, p1s, p2s = [], [], []    
        global_epoch, global_step, stats = 0, 0, {'losses': {}}
    # eps decay
    EPS_START, EPS_END, EPS_DECAY = 0.5, 0.05, 2048
    
    for num_iter in range(iters):
        for index in range(len(tr_personas)):
            ### 1. interact w/ policy ###
            p2 = tr_personas[index]
            ## reset env
            user = UserSim(i2p, p2, top_k = top_k, top_p = top_p)
            agent.reset_convo()
            ## run convo trajectory
            msg, state, context = None, torch.zeros(1024).to(device), torch.zeros(1024).to(device)
            states, contexts, rewards, acts = [state.tolist() + [0.0]], [context.tolist() + [0.0]], [], []
            print("="*20, " iter %d, env %d, buffer size %d "%(num_iter, index, len(memory_buffer)), "="*20)
            print("Conducting conversation ... ")
            policy.eval()
            for turn in range(8): 
                # p1 update
                with torch.no_grad():
                    turn_tensor = torch.ones(1,1).to(device) * turn
                    state, context = state.view(1,1024), context.view(1,1024)
                    state_t = torch.cat((state, turn_tensor), dim=-1)
                    context_t = torch.cat((context, turn_tensor), dim=-1)
                    x = torch.cat((state_t, context_t), dim=-1)
                    logits = policy(x)
                    p = F.softmax(logits,-1)
                    # eps-greedy
                    eps = random.random()
                    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * global_step / EPS_DECAY)
                    global_step +=1
                    if eps > threshold:
                        if turn == 0:
                            action = torch.multinomial(p, num_samples=1).item()
                        else:
                            action = p.max(1)[1].item()
                    else:
                        action = random.randrange(11)
                acts.append(action)
                agent.p1 = [action_dict[action]]
    
                # update msg rounds
                msg = agent(msg, act=False)
    
                # user response
                msg = user(msg)
                state, context = _get_state(clf, user.dialog_history[1::2])
                reward = _calculate_reward(clf, context, p2, k=20)
                rewards.append(reward)
                _state, _context = state.tolist(), context.tolist()
                _state.append(turn+1.0); _context.append(turn+1.0)
                # record into (s,a,r)
                states.append(_state); contexts.append(_context)
            ## convo statistics tracking 
            agent._update_dialog_hx(msg)
            # evaluate dialog
            prec1, prec5, rec5, rec10 = clf.evaluate(context.view(1,-1), [p2])
            # log data
            prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
            dialog_histories.append(agent.dialog_history)
            p2s.append(p2); p1s.append(acts)
            ## to memory 
            next_states = states[1:]; states = states[:-1]
            next_contexts = contexts[1:]; contexts = contexts[:-1]
            next_acts = acts[1:] + [0]
            batch = list(zip(states, contexts, acts, next_states, next_contexts, next_acts, rewards))
            memory_buffer.extend([list(tuples) for tuples in batch])
            print("prec@1: %.2f | prec@5: %.2f | rec@5: %.2f | rec@10: %.2f" % (prec1[0], prec5[0], rec5[0], rec10[0]))
            print("avg p@1: %.2f | avg p@5: %.2f | avg r@5: %.2f | avg. r@10: %.2f" % (np.mean(prec1s), np.mean(prec5s), 
                                                                                        np.mean(rec5s), np.mean(rec10s)))
            if prec1[0] < .5:
                print(" - " *20)
                display_dialog_history(agent.dialog_history)
                print(user.p2)
                print(" - "*20)
            print("Offline batch updates ... ")
            ## 2. batch updates
            policy.train()
            for epoch in range(num_train_epochs):
                #dataloader_old = DataLoader(flat_data, batch_size=64, shuffle=True)
                dataloader = DataLoader(memory_buffer, batch_size=64, shuffle=True)
                #data_iter = iter(dataloader_new)
                epoch_loss, epoch_q_loss = [], []
                for i, batch in enumerate(dataloader):#dataloader_old):
                    # get batch on expert trajectories 
                    xx, y, q_targets = _process_dqn_batch(batch, policy, swa_policy,gamma)
                    # get old batch loss
                    yhat = policy(xx)   # logits
                    #loss_old = criterion(yhat, y) # loss on softmax(yhat)
                    # index logits by action index
                    #aux_loss_old = huber_loss(yhat[torch.arange(len(y)),y], q_targets)
                    #total_loss_old = loss_old + aux_loss_old
                    loss = huber_loss(yhat[torch.arange(len(y)),y], q_targets)                
                    # get batch on new trajectories
                    # try: 
                    #     batch = data_iter.next()
                    # except:
                    #     data_iter = iter(dataloader_new)
                    #     batch = data_iter.next()
                    # xx, y, q_targets = _process_dqn_batch(batch, policy, swa_policy, gamma)
                    # yhat = policy(xx)
                    # loss_new = huber_loss(yhat[torch.arange(len(y)), y], q_targets)
                    # total_loss = total_loss_old + loss_new
                    # backward pass
                    policy.zero_grad()
                    #total_loss.backward()
                    loss.backward()
                    optimizer.step()
                    # take swa (polyak avg)
                    swa_policy.update_parameters(policy)
                    # tracking
                    #epoch_loss.append(loss_old.item()); epoch_q_loss.append(loss_new.item())
                    epoch_loss.append(loss.item())
                print("epoch: %d | loss: %.2f | lr: %s" %(epoch, np.mean(epoch_loss), str(scheduler.get_last_lr()[-1])))
                      #| q_loss: %.2f | lr: %s" %(epoch,  np.mean(epoch_loss), 
                                                                        # np.mean(epoch_q_loss), str(scheduler.get_last_lr()[-1])))
                # on epoch end
                scheduler.step()
                stats['losses'][global_epoch] = {'dqn_loss': np.mean(epoch_loss)}
                plot_losses(stats['losses'], title='dqn_loss' )
                global_epoch +=1
            # on dqn iter end
            torch.save(policy.state_dict(), os.path.join(imitation_path, 'dqn_policy.pt'))
            torch.save(swa_policy.state_dict(), os.path.join(imitation_path, 'target_policy.pt'))
            torch.save(optimizer.state_dict(), os.path.join(imitation_path, 'optimizer.pt'))
            torch.save(scheduler.state_dict(), os.path.join(imitation_path, 'scheduler.pt'))
            stats['global_step'] = global_step; stats['global_epoch'] = global_epoch
            stats['prec1s'] = prec1s; stats['prec5s'] = prec5s; stats['rec5s'] = rec5s
            stats['rec10s'] = rec10s; stats['hx'] = dialog_histories 
            stats['p1s'] = p1s; stats['p2s'] = p2s
            with open(os.path.join(imitation_path, 'dqn_stats'), 'wb') as f:
                pickle.dump(stats,f)
            with open(os.path.join(imitation_path, 'memory_buffer'), 'wb') as f:
                pickle.dump(memory_buffer, f)
            
    return policy, swa_policy, stats


def _process_dqn_batch(batch, policy, swa_policy, gamma):
    try:
        x, c, y, x_next, c_next, y_next, r = batch
        y_next = y_next.to(device)

    except:
        x, c, y, x_next, c_next, r = batch
    x,c = torch.stack(x, dim=-1).type(torch.cuda.FloatTensor), torch.stack(c, dim=-1).type(torch.cuda.FloatTensor)
    x_next, c_next = torch.stack(x_next, dim=-1).type(torch.cuda.FloatTensor), torch.stack(c_next, dim=-1).type(torch.cuda.FloatTensor)
    y, r = y.to(device), r.type(torch.cuda.FloatTensor)
    xx = torch.cat((x,c), dim=-1)
    xx_next = torch.cat((x_next, c_next), dim=-1)
    # calculate q-values
    with torch.no_grad():
        # use target network to predict q-targets
        q_values = policy(xx_next)
        idx = q_values.max(1)[1]
        q_values = swa_policy(xx_next)
        dones = (x[:,-1] >= 7).long()
        q_targets = r + (1-dones) * gamma * q_values[torch.arange(len(idx)), idx]
    return xx, y, q_targets

def _get_state(clf, dialog_hx):#, actions):
    # hx x action -> state
    with torch.no_grad():
        state_conv = model(to_var(flatten(dialog_hx)).long(), 
                     output_hidden_states=True)[2][24].squeeze(0).mean(0)
        # predicted personality -> 1024 clf state
        state_clf = clf.clf(state_conv.unsqueeze(0))[0].squeeze(0)
    return state_conv, state_clf

def _calculate_reward(clf,state, labels, k=20):#, sigma=100.):
    with torch.no_grad():
        pos, neg = clf._generate_candidates([labels], k)
        loss = clf.criterion(state.unsqueeze(0), pos[0], neg[0])
        # turn loss into a distribution q_t over actions
        #reward = torch.exp( - loss / sigma) + 1e-5
        #return reward.item()
        return - loss.item()
    
def _calculate_returns(rewards, gamma=1.0):
    '''calculates the RAW returns for each episode. '''
    R, returns = 0, []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0,R)
    return returns
    
def eval_agent(i2p, clf, action_dict, agent_setting, user_setting, 
               te_personas, tr_personas, policy = None, top_k=10, top_p=.92):
    if agent_setting == 'none':
        agent = UserSim(i2p, None, top_k = top_k, top_p = top_p)
    elif agent_setting == 'random':
        agent = UserSim(i2p, None, reverse=True, top_k = top_k, top_p = top_p)
    elif agent_setting == 'policy':
        agent = UserSim(i2p, None, reverse=True, top_k = top_k, top_p=top_p)
        assert policy is not None
    # reset stats
    prec1s, prec5s, rec5s, rec10s = [], [], [], []
    dialog_histories, p1s, p2s = [], [], []
    # cycle through eval set
    for i in range(len(te_personas)):            
        # reset user
        p2 = te_personas[i]
        if user_setting == 'transition':
            user = UserSim(i2p, [])
        elif user_setting == 'weak':
            user = UserSim(i2p, p2, top_k = 500, top_p = .85)
        else:
            user = UserSim(i2p, p2, top_k = top_k, top_p = top_p)
        # reset agent
        if agent_setting == 'random': action_seq = random.choices(list(action_dict.values()), k=8)
        elif agent_setting == 'persona':
            p1 = random.sample(tr_personas, 1)[0]
            agent = UserSim(i2p, p1, reverse=True, top_k = top_k, top_p = top_p)
        elif agent_setting == "policy": action_seq = []
        agent.reset_convo()
        # run convo trajectory
        msg, state, context = None, torch.zeros(1024).to(device), torch.zeros(1024).to(device)
        for turn in range(8):
            if agent_setting == 'random':
                agent.p1 = [action_seq[turn]]    
            if agent_setting == 'policy':
                # p1 update
                with torch.no_grad():
                    turn_tensor = torch.ones(1,1).to(device) * turn
                    state, context = state.view(1,1024), context.view(1,1024)
                    state_t = torch.cat((state, turn_tensor), dim=-1)
                    context_t = torch.cat((context, turn_tensor), dim=-1)
                    x = torch.cat((state_t, context_t), dim=-1)
                    logits = policy(x)
                    p = F.softmax(logits,-1)
                    if turn == 0:
                        action = torch.multinomial(p, num_samples=1).item()
                    else:
                        action = p.max(1)[1].item()
                action_seq.append(action)
                agent.p1 = [action_dict[action]]
                # nearest neighbor for p2
                #if turn >0: 
                    #nns = clf._get_knns(context.view(1,-1), [], at_k=5)
                    ##agent.p2 = [i2p[p] for p in nns]
                    #agent.p1.extend( [i2p[p] for p in nns] )
            # update msg rounds
            if agent_setting in ['none', 'persona']:
                msg = agent(msg, act=False)
            else:
                msg = agent(msg, act=True)
            # user response
            msg = user(msg)
            state, context = _get_state(clf, user.dialog_history[1::2])
        
        agent._update_dialog_hx(msg)
        # evaluate dialog
        prec1, prec5, rec5, rec10 = clf.evaluate(context.view(1,-1), [p2])
        # log data
        prec1s.extend(prec1); prec5s.extend(prec5); rec5s.extend(rec5); rec10s.extend(rec10)
        dialog_histories.append(agent.dialog_history)
        p2s.append(p2)
        if agent_setting in ['random', 'policy']: p1s.append(action_seq)
        elif agent_setting == 'persona': p1s.append(agent.p1)
        
        print( "[%d / %d ] prec1: %.1f, prec5: %.1f, rec5: %.1f, rec10: %.1f" % (i, 
                                len(te_personas), 100*np.mean(prec1s), 100*np.mean(prec5s), 
                                100*np.mean(rec5s), 100*np.mean(rec10s)) )
        
    return {'prec1': prec1s, 'prec5': prec5s, 'rec5': rec5s, 'rec10': rec10s, 
            'histories': dialog_histories, 'p1': p1s, 'p2': p2s}

if __name__ == '__main__':  
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

    # def action space
    action_space = [ 'ask about kids.', "ask about pets.", 'talk about work.', 
               'ask about marital status.', 'talk about travel.', 'ask about age and gender.',
        'ask about hobbies.', 'ask about favorite food.', 'talk about movies.', 
        'talk about music.', 'talk about politics.']
    
    action_dict = dict([(k,v) for k,v in enumerate(action_space)])
    
    imitation_path = os.path.join(save_path, 'checkpoint/imitation')
    opts.imitation_path = imitation_path
    # expert dataset
    try:
        with open(os.path.join(imitation_path, 'data'), 'rb') as f:
            dataset = pickle.load(f)
    except:
        with open(os.path.join(opts.active_learning_path, 'pi_data'), 'rb') as f: data=pickle.load(f)
        hx, actions = data['hx'], data['a']
        
        ## data set ###
        create_dir(imitation_path)
        dataset = convert_dialog_to_IL_data(clf, hx, actions)
        with open(os.path.join(imitation_path, 'data'), 'wb') as f: pickle.dump(dataset, f)
    
    # imitation learning
    try: 
        policy = nn.Sequential(nn.Linear(2050, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, 11)).cuda()
        state_dict = torch.load(os.path.join(imitation_path, 'policy.pt'))
        policy.load_state_dict(state_dict)
    except:
        policy, swa_policy = pretrain_policy(dataset)
        torch.save(policy.state_dict(), os.path.join(imitation_path, 'policy.pt'))
        torch.save(swa_policy.state_dict(), os.path.join(imitation_path, 'swa_policy.pt'))
    