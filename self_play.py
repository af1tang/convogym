#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:48:43 2020

@author: af1tang
"""
import torch, os, pickle, random
from load_configs import model, tokenizer, opts, device, data_path, save_path
from utils import *
from models import *

## Self-Play ##
def self_play(model, tokenizer, clf, i2p, 
              mode='authentication', at_k=5, reverse_picks=False, both_modes=False,
              weaken_mode = False, #training_mode=False, authenticator = None, 
              length=8, top_k=10, top_p=.92, max_length=1024):
    p1_descr, p2_descr, sep_tok = p1_tok+ ['person 1: '], p2_tok+ ['person 2: '],  [tokenizer.sep_token]
    greet, eos_tok = ["hi! how's it going?"], [tokenizer.eos_token] # the intial prompt
    p1 = random.sample(i2p.keys(),5)
    p2_ids = random.sample(i2p.keys()-set(p1),5)    # this is the "input" chatbot
    p1 = [i2p[pi] for pi in p1]  
    p2 = [i2p[pi] for pi in p2_ids] 
    if mode != 'persona': p1 = ['Not Available.']
    if weaken_mode: p2 = ['Not Available.']
    #elif mode == 'authentication': p1 = ['1 2 3 4 5'] # indices 12-22
    # context: p1 = "user" -- possibly gradients, p2 = "bot" -- no gradients
    src_inp = tokenizer.encode(''.join(p1_descr + ['Not available.'] + sep_tok + p2_descr +p1 +sep_tok ), return_tensors='pt').to(device)
    trg_inp = tokenizer.encode(''.join(p1_descr + ['Not available.'] + sep_tok + p2_descr +p2 +sep_tok + [start_tok]), return_tensors='pt').to(device)
    # use PAST (src and trg) to keep track of (a) gradients, (b) history for separate decoding
    past_src, past_trg, src_len, trg_len = None, None, src_inp.size(1), trg_inp.size(1)
    dialog_history, banned = [], []
    for step in range(length):
        #################
        ## source side ##
        #################
        # 1. "source" asks question and "target" answers question
        if step ==0:
            new_inp = tokenizer.encode(''.join(greet + eos_tok))
        if (mode =='authentication'): #and (authenticator is not None):
            if step >0:
                with torch.no_grad():  # here, we are obtaining the persona-anchor based on curr history
                    x = model(to_var(torch.tensor(flatten(dialog_history[1::2]))), 
                              output_hidden_states=True)[2][24].squeeze(0).mean(0) # (dim, )
                    p_pred = clf.clf(x.unsqueeze(0))[0]
                # knn on persona-anchor
                picks = clf._get_knns(p_pred, banned = banned, at_k=at_k, 
                                      reverse_picks=reverse_picks, both_modes=both_modes)            
                knns = [i2p[pi] for pi in picks]
                banned.extend(knns)
                # re-condition on new persona
                src_inp = tokenizer.encode(''.join(p1_descr + ["Not available."] + sep_tok + p2_descr + knns + sep_tok + [start_tok]))
                src_inp += flatten(dialog_history)
                src_inp, src_len, past_src = to_var(torch.tensor([src_inp])), len(src_inp), None
            # output
            new_inp = []
            with torch.no_grad():
                while (tokenizer.eos_token_id not in new_inp) and (src_len + len(new_inp) < max_length):
                    outp, past_src = model(src_inp, past=past_src)
                    # top k sampling          
                    log_scores = top_k_top_p_filtering(outp[:,-1,:], top_k=top_k, top_p=top_p)
                    probs = F.softmax(log_scores, dim=-1)
                    token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    # update tokens for next output
                    new_inp += token.tolist()
                    src_inp = token.unsqueeze(0)
        else:
            new_inp = []
            with torch.no_grad():
                while (tokenizer.eos_token_id not in new_inp) and (src_len + len(new_inp) < max_length):
                    outp, past_src = model(src_inp, past=past_src)
                    # top k sampling          
                    log_scores = top_k_top_p_filtering(outp[:,-1,:], top_k=top_k, top_p=top_p)
                    probs = F.softmax(log_scores, dim=-1)
                    token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    # update tokens for next output
                    new_inp += token.tolist()
                    src_inp = token.unsqueeze(0)
                    src_len+=1
        # add new_inp -> trg_inp 
        dialog_history.append(new_inp)
        #################
        ## target side ##
        #################
        if step ==0:
            trg_inp = torch.cat([trg_inp, to_var(torch.tensor([new_inp]))], dim=-1)  
        else:
            trg_inp = to_var(torch.tensor([new_inp]))
        # 2. "target" asks question and "source" answers question
        new_inp = []
        with torch.no_grad():
            while (tokenizer.eos_token_id not in new_inp) and (trg_len + len(new_inp) < max_length):
                # fast decode
                outp, past_trg = model(trg_inp, past=past_trg)
                # top-k and top-p sampling
                log_scores = top_k_top_p_filtering(outp[:,-1,:], top_k=top_k, top_p=top_p)
                probs = F.softmax(log_scores, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(1)
                # update list of tokens for next round
                new_inp += token.tolist()
                trg_inp = token.unsqueeze(0)
                trg_len+=1
        dialog_history.append(new_inp)
        src_inp = to_var(torch.tensor([new_inp]))
    return p2_ids, (p1,p2, dialog_history)

def evaluate_authenticator(mode = 'authentication', batch_size=10):
    model, tokenizer, stats = load_from_pretrained()
    # persona dict helpers
    i2v = torch.load(os.path.join(opts.identifier_path, 'i2v'))
    with open(opts.i2p_path, 'rb') as f: i2p = pickle.load(f)
    # load clf
    clf = CLF(i2v, mode= "ID", zsl=True).cuda()
    id_save_path = os.path.join(opts.identifier_path, 'ID.pt')
    state_dict = torch.load(id_save_path)
    clf.load_state_dict(state_dict)
    print("Building dataset from self-play ... ")
    X,y, histories = [],[], []
    for i in range(100):
        p2, (p1,_,hx) = self_play(model,tokenizer,clf,i2p, 
                                  mode=opts.auth_mode, at_k=opts.at_k,
                                  reverse_picks=opts.reverse_picks,
                                  weaken_mode=opts.weaken_mode)
        with torch.no_grad():
            yi = p2 #torch.stack([i2v[pi] for pi in p2])
            x = model(to_var(torch.tensor(flatten(hx[1::2]))), output_hidden_states=True)[2][24].squeeze(0).mean(0)
        X.append(x); y.append(yi); histories.append(hx)
        if i%10==0: print(i)
    dataset = list(zip(X,y))
    print("Evaluating ... ")
    eval_stats = {'prec@1':[], 'prec@5':[], 'rec@5':[], 'rec@10': []}
    for step, minibatch in enumerate(chunker(range(len(dataset)), batch_size)):
        # batching
        xx= torch.stack([dataset[k][0] for k in minibatch])
        yy = [dataset[k][1] for k in minibatch]
        # scoring
        prec1, prec5, rec5, rec10 = clf.evaluate(xx, yy)
        eval_stats['prec@1'].extend(prec1)
        eval_stats['prec@5'].extend(prec5)
        eval_stats['rec@5'].extend(rec5)
        eval_stats['rec@10'].extend(rec10)
    print("Done.")
    eval_stats['trajectories'] = histories
    return dataset, eval_stats