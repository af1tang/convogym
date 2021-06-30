 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:32:37 2020
@author: af1tang
"""

import torch, os, pickle, random
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from load_configs import model, tokenizer, stats, opts, device, create_dir, p1_tok, p2_tok, start_tok, act_tok
from utils import *
from agents import UserSim
from train import train_loop, evaluate_loop

## model saving ##
def checkpoint(model, tokenizer):
    create_dir(opts.active_learning_path)
    model.save_pretrained(opts.active_learning_path)
    tokenizer.save_pretrained(opts.active_learning_path)
    torch.save(opts, os.path.join(opts.active_learning_path, "training_opts.bin"))
    
def split_by_index(seq, sep):
    g = []
    for el in seq:
        g.append(el)
        if el == sep:
            yield g
            g = []
    
def single_persona_fine_tuning(i2p, old_data, reverse=True, single_data= None,#max_iters = 100,
                               top_k=10, top_p=.92, num_train_epochs=1):
    if single_data is not None:
        data = single_data
    else:
        data = []
        print("building training set...")
        for i in range(len(old_data)):
            p1 = [tokenizer.decode(p) for p in old_data[i]['p_src']]
            p2 = [tokenizer.decode(p) for p in old_data[i]['p_trg']]
            # agent = UserSim(i2p, p1, top_k=top_k, top_p=top_p)
            # user = UserSim(i2p, p2, top_k=top_k, top_p=top_p)
            # msg, dialog_hx = None, []
            # for turn in range(8):
            #     if msg is not None: dialog_hx.append(msg)
            #     msg = agent(msg)
            #     dialog_hx.append(msg)
            #     msg = user(msg)
            # dialog_hx.append(msg)
            convo = old_data[i]['inp'] + old_data[i]['labels']
            dialog_hx = list(split_by_index(convo,tokenizer.eos_token_id))
            #if len(dialog_hx) < 30:
            dialog_hx = dialog_hx[:20] # limit by max len of convo
            if reverse:
                p1_ctx = tokenizer.encode(''.join(['<|p1|>', 'person 1: '] + p1 + ['<|sep|>'] + ['<|p2|>', 'person 2: '] + [] + ['<|sep|>'] + ['<|start|>']))
            else:
                p1_ctx = tokenizer.encode(''.join(['<|p1|>', 'person 1: '] + [] + ['<|sep|>'] + ['<|p2|>', 'person 2: '] + p1 + ['<|sep|>'] + ['<|start|>']))
            p2_ctx = tokenizer.encode(''.join(['<|p1|>', 'person 1: '] + [] + ['<|sep|>'] + ['<|p2|>', 'person 2: '] + p2 + ['<|sep|>'] + ['<|start|>']))
            for t in range(len(dialog_hx)):
                x = dialog_hx[:t]
                y = dialog_hx[t]
                if t == 0:
                    x = p1_ctx[:-1] 
                    y = [p1_ctx[-1]] + y
                elif t %2 ==0:
                    x = p1_ctx + flatten(x)
                else:
                    x = p2_ctx + flatten(x)
                data.append((x,y))
        print("done.")
    # fine tuning
    dataloader = DataLoader(data, batch_size=1, shuffle=True); del data
    ## optimizer and scheduler ##
    # larger batch size since more updates
    opts.gradient_accumulation_steps = 64
    t_total = len(dataloader) // opts.gradient_accumulation_steps * num_train_epochs
    with torch.no_grad():
        fast_group = flatten([[p[act_tok], p[start_tok], p[p1_tok], p[p2_tok]] for n,p in model.named_parameters() if n == 'transformer.wte.weight']) #['transformer.wte.weight']
        freeze_group = [p[:start_tok] for n,p in model.named_parameters() if n == 'transformer.wte.weight']#['transformer.wte.weight']
        slow_group = [p for n,p in model.named_parameters() if n == 'transformer.wpe.weight']
        normal_group = [p for n,p in model.named_parameters() if n not in ('transformer.wte.weight',                                                                           'transformer.wpe.weight')]
    
    optimizer_grouped_parameters = [{"params": fast_group, 'lr': 5e-4}, #5e-6
                                    {"params": freeze_group, 'lr': 1e-9}, 
                                    {"params": slow_group, 'lr': 1e-7}, # 1e-7
                                    {"params": normal_group, 'lr': 5e-6}] #1e-6
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=opts.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.warmup_steps, 
                                                num_training_steps=t_total)
    
    # track stats
    stats = {}
    global_step, epochs_trained, steps_trained_in_current_epoch = 0,0,0
    tr_loss, logging_loss = 0.0, 0.0
    # very important: set model to TRAINING mode!
    model.zero_grad(); model.train()
    print("Re-sizing model ... ")
    model.resize_token_embeddings(len(tokenizer))
    for epoch in range(num_train_epochs):
        data_iter= iter(dataloader)
        for step in range(len(dataloader)):
            ### step ###
            batch = data_iter.next()
            loss = fit_on_batch(batch); del batch
            # logging (new data only)
            tr_loss += loss.item()
            
            # gradient accumulation
            if (step+1) % opts.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # reporting 
                if global_step % opts.logging_steps ==0:
                    stats[global_step] = {'loss': (tr_loss - logging_loss) / opts.logging_steps, 
                                          'lr': scheduler.get_last_lr()[-1]}
                    logging_loss = tr_loss
                    
                    print('Epoch: %d | Iter: %d | loss: %.3f | lr: %s ' %( 
                    epoch, global_step, stats[global_step]['loss'],                             
                            str(stats[global_step]['lr'])) )
                    
                if global_step % opts.save_steps==0:
                    print("Saving stuff ... ")
                    #checkpoint(model, tokenizer)
                    plot_losses(stats, title='loss' )
                    plot_losses(stats, title='lr')
                    print("Done.")
                    
def eval_single_persona(i2p, eval_data, reverse=True, single_data = None):
    if single_data is not None: 
        data = single_data
    else:
        data = []
        print("building test set...")
        for i in range(len(eval_data)):
            p1 = [tokenizer.decode(p) for p in eval_data[i]['p_src']]
            p2 = [tokenizer.decode(p) for p in eval_data[i]['p_trg']]
            # agent = UserSim(i2p, p1, top_k=top_k, top_p=top_p)
            # user = UserSim(i2p, p2, top_k=top_k, top_p=top_p)
            # msg, dialog_hx = None, []
            # for turn in range(8):
            #     if msg is not None: dialog_hx.append(msg)
            #     msg = agent(msg)
            #     dialog_hx.append(msg)
            #     msg = user(msg)
            # dialog_hx.append(msg)
            convo = eval_data[i]['inp'] + eval_data[i]['labels']
            dialog_hx = list(split_by_index(convo,tokenizer.eos_token_id))
            #if len(dialog_hx) < 30:
            dialog_hx = dialog_hx[:30] # limit by max len of convo
            if reverse:
                p1_ctx = tokenizer.encode(''.join(['<|p1|>', 'person 1: '] + p1 + ['<|sep|>'] + ['<|p2|>', 'person 2: '] + [] + ['<|sep|>'] + ['<|start|>']))
            else:
                p1_ctx = tokenizer.encode(''.join(['<|p1|>', 'person 1: '] + [] + ['<|sep|>'] + ['<|p2|>', 'person 2: '] + p1 + ['<|sep|>'] + ['<|start|>']))
            p2_ctx = tokenizer.encode(''.join(['<|p1|>', 'person 1: '] + [] + ['<|sep|>'] + ['<|p2|>', 'person 2: '] + p2 + ['<|sep|>'] + ['<|start|>']))
            for t in range(len(dialog_hx)):
                x = dialog_hx[:t]
                y = dialog_hx[t]
                if t == 0:
                    x = p1_ctx[:-1] 
                    y = [p1_ctx[-1]] + y
                elif t %2 ==0:
                    x = p1_ctx + flatten(x)
                else:
                    x = p2_ctx + flatten(x)
                data.append((x,y))
        print("done.")
    print("Validating ... ")
    dataloader = DataLoader(data, batch_size=1, shuffle=True); del data
    data_iter = iter(dataloader)
    with torch.no_grad():
        eval_stats, total_steps, val_loss, val_f1_score = {}, 0, 0.0, 0.0
        model.eval()
        for i in range(len(dataloader)):
            batch = data_iter.next()
            xx,yy = batch
            try:
                xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
            except:
                xx, yy = to_var(xx), to_var(yy)
            ## forward on new data batch
            _, past = model(xx); del _
            outp = model(yy, past=past, labels=yy)
            loss = outp[0]
            ytrue=np.array( filter_turn_indices(to_data(yy[...,1:].contiguous().view(-1)) ) )
            #ypred = to_data(outp[1][..., :,:].contiguous().topk(1)[1].view(-1))
            ypred=np.array( filter_turn_indices(to_data( outp[1][..., :-1, :].contiguous().topk(1)[1].view(-1)) ) ) 
            min_len = min(len(ypred), len(ytrue))
            hits = [set(ypred[i]).intersection(set(ytrue[i])) for i in range(min_len)]#set(ytrue).intersection(set(ypred))
            prec = [len(hits[i])/len(ypred[i]) for i in range(min_len)]
            rec = [len(hits[i])/len(ytrue[i]) for i in range(min_len)]
            f1 = np.mean([2*(prec[i]*rec[i])/(prec[i] + rec[i]+1e-3) for i in range(min_len)])
            val_f1_score += f1
            val_loss += loss.mean().item()
            total_steps +=1 
            if total_steps%20 ==0: print(total_steps)
            
    val_loss = val_loss / total_steps 
    val_f1_score = val_f1_score / total_steps
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    eval_stats = {'perplexity': perplexity, 'loss': val_loss, 'f1': val_f1_score}
    print("Done.")
    return eval_stats

    
def collection_loop(i2p, tr_personas, action_space, data =None,
                    iters_per_action=50, top_k=10, top_p=.92):
    data = {} if data is None else data
    model.eval()
    for i, act in enumerate(action_space):
        if i in data.keys():
            print(len(data[i]['labels']), len(data[i]['input_ids']), data[i]['action'])
        else:
            data[i] = {'input_ids': [], 'labels': [], 'action': act}
            for j in range(iters_per_action):
                p1, p2 = random.sample(tr_personas, 2)
                length = random.sample(range(0,8), 1)[0]
                # reset convo agents 
                agent = UserSim(i2p, p1, top_k=top_k, top_p = top_p)
                user = UserSim(i2p, p2, top_k = top_k, top_p = top_p)
                msg = None
                # run for first N-1 turns
                for turn in range(length):
                    msg = agent(msg)
                    msg = user(msg)
                # update action as control code
                agent.p2 = [act]; agent._update_dialog_hx(msg)
                x = tokenizer.encode(''.join(agent.p1_descr + agent.p1 + agent.sep_tok + agent.act_descr + agent.p2 + agent.sep_tok + ['<|start|>']))
                x += flatten(agent.dialog_history)
                x = to_var(torch.tensor([x]))
                if turn ==0 : x = x[:, :-1]
                # get user input
                print(); print('-'*50)
                display_dialog_history(agent.dialog_history)
                print('-'*12, ' action %d, iter %d '%(i,j), '-'*12 )
                print("action: ", act)
                # set user input as labels for dataset
                y = tokenizer.encode(input("  >> user: ") + tokenizer.eos_token)
                # update dataset
                data[i]['input_ids'].append(x.tolist()); data[i]['labels'].append(y)
    return data

def fit_on_batch(batch):
    xx,yy = batch
    try:
        xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
    except:
        xx, yy = to_var(xx), to_var(yy)
    ## forward on new data batch
    _, past = model(xx); del _
    outp = model(yy, past=past, labels=yy)
    
    # backward
    loss = outp[0]; del outp
    if opts.gradient_accumulation_steps > 1:
        loss = loss / opts.gradient_accumulation_steps
    loss.backward()
    return loss

def pretrain(new_data, old_data, stats = None, num_train_epochs = 1):
    ## prep dataloaders ##
    # prep new data for batching
    X, y = new_data['X'], new_data['y']
    dataloader_new = DataLoader(list(zip(X,y)), batch_size=1, shuffle=True)
    # prep old data for batching
    ## OLD: original corpus ##
    # X = [[p1_tok] + tokenizer.encode("person 1: ") + flatten(old_data[i]['p_src']) + 
    #      [tokenizer.sep_token_id] + [p2_tok] + tokenizer.encode("person 2: ") + 
    #          flatten(old_data[i]['p_trg']) + [tokenizer.sep_token_id] 
    #          for i in range(len(old_data))]
    # y = [[start_tok] + old_data[i]['inp'] + old_data[i]['labels'] for i in range(len(old_data))]
    ## NEW: fine tuned corpus ##
    dataloader_old = DataLoader(old_data, batch_size=1, shuffle=True); del X, y
    
    ## optimizer and scheduler ##
    # calculate total steps
    opts.gradient_accumulation_steps = 64
    t_total = len(dataloader_old) // opts.gradient_accumulation_steps * num_train_epochs
    #t_total = max_steps
    #num_train_epochs = max_steps // (len(dataloader_old) // opts.gradient_accumulation_steps) + 1

    ## set up optimizers and schedulers ##
    # TODO: fine-tuning should serve as approx Fischer Information 
    #no_decay = ["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]   #LayerNorm.weight -> ln_1.weight, ln_2.weight
    with torch.no_grad():
        fast_group = flatten([[p[act_tok], p[start_tok], p[p1_tok], p[p2_tok]] for n,p in model.named_parameters() if n == 'transformer.wte.weight']) #['transformer.wte.weight']
        freeze_group = [p[:start_tok] for n,p in model.named_parameters() if n == 'transformer.wte.weight']#['transformer.wte.weight']
        slow_group = [p for n,p in model.named_parameters() if n == 'transformer.wpe.weight']
        normal_group = [p for n,p in model.named_parameters() if n not in ('transformer.wte.weight',
                                                                           'transformer.wpe.weight')]
    
    optimizer_grouped_parameters = [{"params": fast_group, 'lr': 5e-4}, #5e-4
                                    {"params": freeze_group, 'lr': 1e-9}, 
                                    {"params": slow_group, 'lr': 1e-7}, # 1e-8
                                    {"params": normal_group, 'lr': 1e-6}] #5e-6
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6, eps=opts.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.warmup_steps, 
                                                num_training_steps=t_total)
    # loading optimizer settings
    if (opts.active_learning_path and os.path.isfile(os.path.join(opts.active_learning_path, "optimizer.pt"))
                                and os.path.isfile(os.path.join(opts.active_learning_path, "scheduler.pt")) ):
        # load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(opts.active_learning_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(opts.active_learning_path, "scheduler.pt")))
    # track stats
    if stats is not None:
        global_step = max(stats.keys())
        epochs_trained = global_step // (len(dataloader_old) // opts.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(dataloader_old) // opts.gradient_accumulation_steps)
        print("Resuming Training ... ")
    else:
        stats = {}
        global_step, epochs_trained, steps_trained_in_current_epoch = 0,0,0
    tr_loss, logging_loss = 0.0, 0.0
    tr_loss_old, logging_loss_old = 0.0, 0.0
    model.zero_grad()
    print("Re-sizing model ... ")
    model.resize_token_embeddings(len(tokenizer))
    ## len(data_old) >> len(data_new), so just cycle through data_old w/o resetting per epoch
    
    # NOTE: set model to TRAINING mode
    model.train()
    data_iter_new = iter(dataloader_new)
    data_iter_old = iter(dataloader_old)
    for epoch in range(epochs_trained, num_train_epochs):
        for step in range(len(dataloader_old)): #dataloader_new
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                batch = data_iter_old.next()
                continue
            ### new data step ###
            #if step % 3 == 0:
            try:
                batch = data_iter_new.next()
            except:
                X, y = new_data['X'], new_data['y']
                dataloader_new = DataLoader(list(zip(X,y)), batch_size=1, shuffle=True); del X,y
                data_iter_new = iter(dataloader_new)
                batch = data_iter_new.next()
            new_loss = fit_on_batch(batch); del batch
            tr_loss += new_loss.item()
            
            ## old data step ###
            try:
                batch = data_iter_old.next()
            except:
                data_iter_old = iter(dataloader_old)
                batch = data_iter_old.next()
            old_loss = fit_on_batch(batch); del batch
            tr_loss_old += old_loss.item()
            
            # gradient accumulation
            if (step+1) % opts.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                # reporting 
                if global_step % opts.logging_steps ==0:
                    stats[global_step] = {'new_loss': (tr_loss - logging_loss) / opts.logging_steps, 
                                          'old_loss': (tr_loss_old - logging_loss_old) / opts.logging_steps,
                                          'lr': scheduler.get_last_lr()[-1]}
                    logging_loss = tr_loss
                    logging_loss_old = tr_loss_old
                    
                    print('Epoch: %d | Iter: [%d/%d] | new_loss: %.3f | old_loss: %.3f | lr: %s ' %( 
                    epoch, step, len(dataloader_old), stats[global_step]['new_loss'], 
                            stats[global_step]['old_loss'],
                            str(stats[global_step]['lr'])) )
                    
                if global_step % opts.save_steps==0:
                    print("Saving stuff ... ")
                    checkpoint(model, tokenizer)
                    plot_losses(stats, title='new_loss' )
                    plot_losses(stats, title='old_loss' )
                    plot_losses(stats, title='lr')
                    print("Done.")
                            
    return stats

def active_loop(old_data, i2p, tr_personas, action_space, randomize = False,
                total_iters = 100, top_k=10, top_p = .92):
    # X = [[p1_tok] + tokenizer.encode("person 1: ") + flatten(old_data[i]['p_src']) + 
    #      [tokenizer.sep_token_id] + [p2_tok] + tokenizer.encode("person 2: ") + 
    #          flatten(old_data[i]['p_trg']) + [tokenizer.sep_token_id] #+ [start_tok] 
    #          for i in range(len(old_data))]
    # y = [[start_tok] + old_data[i]['inp'] + old_data[i]['labels'] for i in range(len(old_data))]
    # dataloader = DataLoader(list(zip(X,y)), batch_size=1, shuffle=True); del X, y
    dataloader = DataLoader(old_data, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)
    ## set up optimizers and schedulers ##
    with torch.no_grad():
        fast_group = flatten([[p[act_tok], p[start_tok], p[p1_tok], p[p2_tok]] for n,p in model.named_parameters() if n == 'transformer.wte.weight']) #['transformer.wte.weight']
        freeze_group = [p[:start_tok] for n,p in model.named_parameters() if n == 'transformer.wte.weight']#['transformer.wte.weight']
        slow_group = [p for n,p in model.named_parameters() if n == 'transformer.wpe.weight']
        normal_group = [p for n,p in model.named_parameters() if n not in ('transformer.wte.weight',
                                                                           'transformer.wpe.weight')]
    
    optimizer_grouped_parameters = [{"params": fast_group, 'lr': 1e-6}, #1e-5
                                    {"params": freeze_group, 'lr': 1e-9}, 
                                    {"params": slow_group, 'lr': 1e-8}, #1e-8
                                    {"params": normal_group, 'lr': 5e-7}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-7, eps=opts.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.warmup_steps, 
                                                num_training_steps=total_iters)
    # track stats
    data = {'X': [], 'y': []}
    model.zero_grad()
    model.eval()
    for i in range(total_iters): # 53
        p1, p2 = random.sample(tr_personas, 2)
        # if i % 1 ==0:
        agent = UserSim(i2p, None, reverse=True, top_k=top_k, top_p = top_p)
        # else:
        #     agent = UserSim(i2p, p1, top_k=top_k, top_p = top_p)
        user = UserSim(i2p, p2, top_k = top_k, top_p = top_p)
        msg, dialog_hx = None, []
        action_seq = random.choices(action_space, k=8)
        # run for first N-1 turns
        turn = 0; done = False 
        while not done:
            if msg is not None: dialog_hx.append(msg)
            # reset action 
            # if i %2 ==0:
            if randomize:
                act = action_seq[turn]
            else:
                display_dialog_history(user.dialog_history)
                print()
                print(" actions: ")
                for k,v in enumerate(action_space): print(k,v)
                act = action_space[int(input(" input [0-11]: " ))]
            agent.p1 = [act]
            x = tokenizer.encode(''.join(agent.act_descr + agent.p1 + agent.sep_tok + agent.p2_descr + agent.p2 + agent.sep_tok + ['<|start|>']))
            # else:
            #     x = tokenizer.encode(''.join(agent.p1_descr + agent.p1 + agent.sep_tok + agent.p2_descr + agent.p2 + agent.sep_tok + ['<|start|>']))
            x += flatten(dialog_hx)
            x = to_var(torch.tensor([x]))
            
            # set inp as input_ids for dataset
            if turn == 0:
                x = x[:, :-1]
            
            # generate message from agent
            # if i%2 == 0:
            msg = agent(msg, act=True)
            # else:
            #     msg = agent(msg, act=False)
            dialog_hx.append(msg)
            # get user input
            print(); print('-'*50)
            display_dialog_history(agent.dialog_history)
            print('-'*12, ' iter %d, turn %d '%(i, turn), '-'*12 )
            # if i %2 ==0:
            print("action: ", act)
            # else:
            #     print(" personas: ")
            #     for p in agent.p2: print(p)
            # set user input as labels for dataset
            decision = input(" continue? [y/n] ")
            if decision == 'y':
                # append more turns to active data
                data['X'].extend(x.tolist()); data['y'].append(msg)
                x = tokenizer.encode(''.join(user.p1_descr + user.p1 + user.sep_tok + user.p2_descr + user.p2 + user.sep_tok + ['<|start|>']))
                x += flatten(dialog_hx)
                x = to_var(torch.tensor([x]))
                
                msg = user(msg, act=False)
                print(); print('-'*50)
                display_dialog_history(user.dialog_history)
                print('-'* 12, ' iter %d, turn %d ' %(i, turn), '-' * 12)
                print(" personas: ")
                for p in user.p2: print(p)
                decision = input( " continue? [y/n] " )
                if decision == 'y':
                    turn +=1
    
                else:
                    y = tokenizer.encode(input("  >> user: ") + tokenizer.eos_token, return_tensors='pt')
                    batch = (x,y)
                    data['X'].extend( x.tolist() ); data['y'].extend( y.tolist())
                    # fit on batch
                    print(); print("Fitting on batch ... ")
                    model.train()
                    new_loss = fit_on_batch(batch)
                    batch = data_iter.next()
                    old_loss = fit_on_batch(batch)
                    # step
                    if (i+1) % opts.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        
                        print('Iter: %d | new_loss: %.3f | old_loss: %.3f | lr: %s ' %( 
                                i, new_loss.item() * opts.gradient_accumulation_steps, 
                                old_loss.item() * opts.gradient_accumulation_steps,
                                str(scheduler.get_last_lr()[0])) )
                    done=True
                    model.eval()
            else:
                y = tokenizer.encode(input("  >> user: ") + tokenizer.eos_token, return_tensors='pt')
                if turn ==0:
                    y = torch.cat( (torch.tensor([start_tok]).unsqueeze(0), y), -1)
                batch = (x,y)
                data['X'].extend( x.tolist() ); data['y'].extend( y.tolist())
                # fit on batch
                print(); print("Fitting on batch ... ")
                model.train()
                new_loss = fit_on_batch(batch)
                batch = data_iter.next()
                old_loss = fit_on_batch(batch)
                # step
                if (i+1) % opts.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    
                    print('Iter: %d | new_loss: %.3f | old_loss: %.3f | lr: %s ' %( 
                            i, new_loss.item() * opts.gradient_accumulation_steps, 
                            old_loss.item() * opts.gradient_accumulation_steps,
                            str(scheduler.get_last_lr()[0])) )
                done =True
                model.eval()
                
            if turn >= 8:
                done=True

    return data

def get_expert_policy(i2p, tr_personas, action_space, top_k=10, top_p=.92): 
    def _reset(p2):
        agent = UserSim(i2p, None, reverse=True, top_k=10, top_p = top_p)
        user = UserSim(i2p, p2, top_k = 10, top_p = top_p)
        msg, dialog_hx, actions = None, [], []
        turn, done = 0, False
        return agent, user, msg, dialog_hx, actions, turn, done
    data = {'X': [], 'y': [], 'hx': [], 'a': []}
    model.eval()
    for i in range(59, len(tr_personas)):
        p2 = tr_personas[i]
        agent, user, msg, dialog_hx, actions, turn, done = _reset(p2)
        while not done:
            if msg is not None: dialog_hx.append(msg)
            # get action input
            display_dialog_history(user.dialog_history)
            print()
            print(" actions: ")
            for k,v in enumerate(action_space): print(k,v)
            int_act = int(input(" input [0-11]: " ))
            act = action_space[int_act]
            actions.append(int_act)
            agent.p1 = [act]
            x = tokenizer.encode(''.join(agent.act_descr + agent.p1 + agent.sep_tok + agent.p2_descr + agent.p2 + agent.sep_tok + ['<|start|>']))
            x += flatten(dialog_hx)
            x = to_var(torch.tensor([x]))
            
            # set inp as input_ids for dataset
            if turn == 0:
                x = x[:, :-1]
            # act
            msg = agent(msg, act=True)
            dialog_hx.append(msg)
            print(); print('-'*50)
            display_dialog_history(agent.dialog_history)
            print('-'*12, ' iter %d, turn %d '%(i, turn), '-'*12 )
            # if i %2 ==0:
            print("action: ", act)
            # else:
            #     print(" personas: ")
            #     for p in agent.p2: print(p)
            # set user input as labels for dataset
            decision = input(" continue? [y/n] ")
            if decision == 'y':
                # append more turns to active data
                data['X'].extend(x.tolist()); data['y'].append(msg)
                x = tokenizer.encode(''.join(user.p1_descr + user.p1 + user.sep_tok + user.p2_descr + user.p2 + user.sep_tok + ['<|start|>']))
                x += flatten(dialog_hx)
                x = to_var(torch.tensor([x]))
                
                msg = user(msg, act=False)
                print(); print('-'*50)
                display_dialog_history(user.dialog_history)
                print('-'* 12, ' iter %d, turn %d ' %(i, turn), '-' * 12)
                print(" personas: ")
                for p in user.p2: print(p)
                decision = input( " continue? [y/n] " )
                if decision == 'y':
                    turn +=1
                else:
                    y = tokenizer.encode(input("  >> user: ") + tokenizer.eos_token, return_tensors='pt')
                    data['X'].extend( x.tolist() ); data['y'].extend( y.tolist())
                    # retart convo
                    agent, user, msg, dialog_hx, actions, turn, done = _reset(p2) 
            else:
                y = tokenizer.encode(input("  >> user: ") + tokenizer.eos_token, return_tensors='pt')
                if turn ==0:
                    y = torch.cat( (torch.tensor([start_tok]).unsqueeze(0), y), -1)
                data['X'].extend( x.tolist() ); data['y'].extend( y.tolist())
                # retart convo
                agent, user, msg, dialog_hx, actions, turn, done = _reset(p2)      
                
            if turn >= 8: 
                done = True
        # final message
        dialog_hx.append(msg)
        data['hx'].append(dialog_hx); data['a'].append(actions)

if __name__ == '__main__':       
    # load datasets
    with open(opts.raw_data_path, 'rb') as f: old_data = pickle.load(f)
    #with open(opts.val_data_path, 'rb') as f: eval_data = pickle.load(f)
    with open(os.path.join(opts.identifier_path, 'i2p'), 'rb') as f: 
        i2p = pickle.load(f)
    with open(os.path.join(opts.authenticator_path, 'tr_personas'), 'rb') as f: tr_personas = pickle.load(f)

    # def action space
    action_space = [ 'ask about kids.', "ask about pets.", 'talk about work.', 
               'ask about marital status.', 'talk about travel.', 'ask about age and gender.',
        'ask about hobbies.', 'ask about favorite food.', 'talk about movies.', 
        'talk about music.', 'talk about politics.']
    
    # collect data
    try:
        with open(os.path.join(opts.active_learning_path, 'p1_active_data(extended)'), 'rb') as f: 
            new_data = pickle.load(f)
    except: 
        new_data = collection_loop(i2p, tr_personas, action_space, None)
    # keys = sorted(new_data.keys())
    # X, y = [flatten(new_data[i]['input_ids']) for i in keys], [new_data[i]['labels'] for i in keys]
    # X, y = map( flatten, (X,y))
    # new_data = {'X': X, 'y':y}
        
    # train on active batches
# =============================================================================
#     ## conversion from p2 to p1
#     data = {'X':[], 'y':[]}
#     for i in range(len(p2_data['X'])):
#         x,y = p2_data['X'][i], p2_data['y'][i]
#         p1 = x[:5]
#         jj = 5+x[5:].index(50257)+1
#         p2 = x[5:jj]
#         p1[0] = 50261; p1[2] = 362; p2[2] = 352
#         conv = p2 + p1 + x[jj:]
#         data['X'].append(conv); data['y'].append(y)
# =============================================================================
    