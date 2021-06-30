#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:53:48 2020

@author: af1tang
"""
import torch, os, pickle
from load_configs import model, tokenizer, opts, device, data_path, save_path
from utils import *
from models import *
from preprocess_dataset import _build_persona_id_db

def run_experiment(clf, persona_dict, id_save_path, batch_size=64, persona_size=6737):
                   #    build_from_scratch=False, persona_size = 6737):
    # build dataset
    # print("Building dataset ... ")
    # if build_from_scratch: 
    #     print(" ... from scratch.")
    #     dataset, count={'tr':{}, 'te':{}}, 0
    #     tokenizer, transformer = _initialize_clf_tokenizer(opts.identifier_mode)
    #     for split in persona_dict.keys():
    #         data = persona_dict[split]
    #         for i, minibatch in enumerate(chunker(sorted(data.keys()), batch_size)):
    #             texts = [data[k]['x'] for k in minibatch]
    #             x1, x2 = _process_persona_dict_texts(texts, tokenizer=tokenizer,
    #                                                  transformer=transformer, mode=opts.identifier_mode)
    #             y1 = [data[k]['p1'] for k in minibatch]
    #             y2 = [data[k]['p2'] for k in minibatch]
    #             for k in range(len(minibatch)):
    #                 dataset[split][minibatch[k]] = {'x1': x1[k], 'x2': x2[k],'y1': y1[k], 'y2':y2[k]}
    #                 count+=1
    #                 if count%100==0: print(count)
    #     torch.save(dataset,os.path.join(opts.identifier_path, 'persona_dict_%s'%opts.identifier_mode))
    # else: 
    if clf.zsl:
        dataset=persona_dict
    else: 
        all_data, dataset, count = {},{'tr':{}, 'te':{}},0
        for split in persona_dict.keys():
            for k in sorted(persona_dict[split].keys()):
                all_data[count] = persona_dict[split][k]
                count+=1
        indices = sorted(all_data.keys())
        tr = random.sample(indices, int(len(indices)*.94)) 
        te = list(set(indices) - set(tr))
        for split, idx in [('tr', tr), ('te',te)]:
            count=0
            for k in idx:
                dataset[split][count] = all_data[k]
                count+=1
    #print("Done.")
                    
    # initialize optimizers
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    # training
    stats, iters, tr_loss, logging_loss = {}, 0, 0.0, 0.0
    clf.zero_grad()
    tot_epochs = 10 if clf.zsl else 1
    for epoch in range(tot_epochs):
        indices = sorted(list(dataset['tr']))
        random.shuffle(indices)
        for step, minibatch in enumerate(chunker(indices, batch_size)):
            # batching
            if clf.mode == 'LSTM':
                x1 = pad_sequence([to_var(dataset['tr'][k]['x1']) for k in minibatch], batch_first=True)
                x2 = pad_sequence([to_var(dataset['tr'][k]['x2']) for k in minibatch], batch_first=True)
            else:
                x1 = torch.stack([to_var(dataset['tr'][k]['x1']) for k in minibatch])
                x2 = torch.stack([to_var(dataset['tr'][k]['x2']) for k in minibatch])
            if clf.zsl:
                y1 = [dataset['tr'][k]['y1'] for k in minibatch]
                y2 = [dataset['tr'][k]['y2'] for k in minibatch]
            else:
                y1 = torch.stack([torch.eye(persona_size, dtype=int)[dataset['tr'][k]['y1']].sum(0) for k in minibatch]).to(device)
                y2 = torch.stack([torch.eye(persona_size, dtype=int)[dataset['tr'][k]['y2']].sum(0) for k in minibatch]).to(device)

            # forward
            loss1 = clf(x1, labels=y1, k=opts.k_cands)
            loss2= clf(x2, labels=y2, k=opts.k_cands)
            loss = loss1+loss2
            # backward
            loss.backward()
            tr_loss += loss.item()
            #if (step+1)% opts.gradient_accumulation_steps == 0:
            optimizer.step()
            clf.zero_grad()
            iters +=1
            # reporting
            if iters % opts.logging_steps ==0:
                stats[iters] = {'loss': (tr_loss - logging_loss) / opts.logging_steps}
                logging_loss = tr_loss
                print('Epoch: %d | Iter: %d | loss: %.3f ' %( 
                epoch, iters, stats[iters]['loss']) )
                
            if iters % opts.save_steps==0:
                print("Saving stuff ... ")
                state_dict = clf.state_dict()
                torch.save(state_dict, id_save_path)
                plot_losses(stats, title='loss' )
                print("Done.")
            
    # eval
    print("Evaluating ... ")
    eval_stats = {'prec@1':[], 'prec@5':[], 'rec@5':[], 'rec@10':[]}
    for step, minibatch in enumerate(chunker(sorted(dataset['te'].keys()), batch_size)):
        # batching
        if clf.mode == 'LSTM':
            x1 = pad_sequence([to_var(dataset['te'][k]['x1']) for k in minibatch], batch_first=True)
            x2 = pad_sequence([to_var(dataset['te'][k]['x2']) for k in minibatch], batch_first=True)
        else:
            x1 = torch.stack([dataset['te'][k]['x1'] for k in minibatch])
            x2 = torch.stack([dataset['te'][k]['x2'] for k in minibatch])
        
        if clf.zsl:
            y1 = [dataset['te'][k]['y1'] for k in minibatch]
            y2 = [dataset['te'][k]['y2'] for k in minibatch]
        else:
            y1 = torch.stack([torch.eye(persona_size, dtype=int)[dataset['te'][k]['y1']].sum(0) for k in minibatch]).to(device)
            y2 = torch.stack([torch.eye(persona_size, dtype=int)[dataset['te'][k]['y2']].sum(0) for k in minibatch]).to(device)

        for xx,yy in [(x1,y1), (x2,y2)]:
            prec1, prec5, rec5, rec10 = clf.evaluate(xx, yy)
            eval_stats['prec@1'].extend(prec1)
            eval_stats['prec@5'].extend(prec5)
            eval_stats['rec@5'].extend(rec5)
            eval_stats['rec@10'].extend(rec10)
    print("Done.")
    return stats, eval_stats

if __name__ == '__main__':  
    identifier_modes = ['ID']#["BOW1", "BOW2", "LSTM", "BERT", "GPT", "ID"]
    opts.zsl = True
    opts.k_cands = 20
    #opts.use_cca = True
    opts.use_mtl = False
    #k_cands_opts = [20, 100]
    #zsl_modes = [True, False]
    #cca_modes = [True, False]
    #mtl_modes = [True, False]
    meta_stats ={}
    for identifier_mode in identifier_modes: 
        opts.identifier_mode = identifier_mode
        
        persona_dict = torch.load(os.path.join(opts.identifier_path, 'persona_dict_%s'%opts.identifier_mode))
        # uncomment this line and later line if building from sratch
        #with open(os.path.join(opts.identifier_path, 'persona_dict'), 'rb') as f: persona_dict = pickle.load(f)
        i2v = torch.load(os.path.join(opts.identifier_path, 'i2v'))
        clf = CLF(i2v, mode= opts.identifier_mode, zsl=opts.zsl).cuda()
        id_save_path = os.path.join(opts.identifier_path, '%s.pt' % opts.identifier_mode)
        batch_size, persona_size = 32, len(i2v)
        # run exp
        try:
            state_dict = torch.load(id_save_path)
            clf.load_state_dict(state_dict)
        except Exception as e:
            print(e)
        #persona_dict = _build_persona_id_db(persona_dict, batch_size, opts)
        #torch.save(persona_dict, os.path.join(opts.identifier_path, 'persona_dict_%s'%opts.identifier_mode))
        stats, eval_stats = run_experiment(clf, persona_dict, id_save_path, batch_size=batch_size, persona_size=persona_size)
                       #build_from_scratch=True, persona_size = persona_size)
        meta_stats[opts.identifier_mode] = {'stats': stats, 'eval_stats': eval_stats}
        #save
        create_dir(os.path.join(opts.plot_path, 'identifier-exp'))
        with open(os.path.join(opts.plot_path, 'identifier-exp/%s-training_stats' % opts.identifier_mode), 'wb') as f: pickle.dump(stats,f)
        with open(os.path.join(opts.plot_path, 'identifier-exp/%s-validation_scores' % opts.identifier_mode), 'wb') as f: pickle.dump(eval_stats,f)
