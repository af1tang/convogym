#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 20:17:27 2021

@author: af1tang
"""
import os
import ast
import pandas as pd
import pickle
import torch
from _configs import opts
from utils._reshape import flatten
from utils._device import device, to_data

def prepare_persona_dict(model, tokenizer, data_path):
    df = pd.read_csv(data_path)
    tr_data, te_data = [], []
    for i, row in df.iterrows():
        convo, p1, p2,split = row['hx'], row['p1'], row['p2'], row['split']
        p1, p2 = ast.literal_eval(p1), ast.literal_eval(p2)

        x = [tokenizer.encode(line) for line in convo]
        if split == 'train':
            tr_data.append( [x, p1,p2])
        else:
            te_data.append( [x,p1,p2])
    
    _, tr_p1, tr_p2 = list(zip(*tr_data))
    _, te_p1, te_p2 = list(zip(*te_data))
    tr_personas = [pp for pp in tr_p1] + [pp for pp in tr_p2]
    te_personas = [pp for pp in te_p1] + [pp for pp in te_p2]
    persona_facts = list(set(flatten(tr_personas + te_personas)))
    model.eval()
    print("Making persona dictionary ... ")
    vectors = []
    with torch.no_grad():
        for i, line in enumerate(persona_facts):     
            outp = model(**tokenizer(line, return_tensors='pt').to(device), output_hidden_states=True)
            vectors.append(outp[2][24].squeeze(0).mean(0))
            if (i+1)%100 == 0:
                print("[ processed: %d | total personas: %d ]" %(i+1, len(persona_facts)))
        vectors = torch.stack(vectors)
    
    p2v = dict([ (persona_facts[_], to_data(vectors[_])) for _ in range(len(persona_facts))])
    with open(os.path.join(opts.example_path, 'p2v'), 'wb') as f:
        pickle.dump(p2v, f)
    return p2v