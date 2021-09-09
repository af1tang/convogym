#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 14:34:48 2021

@author: af1tang
"""
import os, pickle, ast
import pandas as pd
import torch
from tqdm import tqdm
from _configs import opts
from utils._reshape import flatten
from utils._device import to_var, to_data, device

def _df_to_array(tokenizer, data_path):
    '''
    converts dataframe -> array of [tokens, p1, p2] 
    input: 
        data_path: 
            path to pandas dataframe of history (hx), personas (p1 and p2), 
            and split (train or test) as columns
    output:
        tr_data: 
            list of rows in df, where hx is represented as token_ids, 
            p1 and p2 are list of strings. 
            
            rows where split == train.
            
        te_data:
            rows where split != train
    '''
    df = pd.read_csv(data_path)
    tr_data, te_data = [], []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        convo, p1, p2,split = row['hx'], row['p1'], row['p2'], row['split']
        convo, p1, p2 = ast.literal_eval(convo), ast.literal_eval(p1), ast.literal_eval(p2)

        x = [tokenizer.encode(line) for line in convo]
        if split == 'train':
            tr_data.append( [x, p1,p2])
        else:
            te_data.append( [x,p1,p2] )
    return tr_data, te_data

def _filter_personas(personas, uniques = None):
    '''
    filters out redundant profiles from a list of personas.
    input: 
        personas: 
            list of persona profiles, each profile consists of 3-5 list of strings
    outputs:
        uniques: 
            set of unique persona profiles
        filtered:
            original list of personas, with each profile replaced by 
            filtered personas. 
            
    example: 
        >>> personas = [['i like wine', 'i have 5 kids'], 
                        ['i hate wine', 'i like ice cream'],
                        ['i like wine', 'i've 5 kids]] 
        >>> uniques, filtered = _filtered_personas) 
        >>> print(filtered)
        
        [[['i like wine', 'i have 5 kids'], 
          ['i hate wine', 'i like ice cream'], 
          ['i like wine', 'i have 5 kids']]
    '''
    if not uniques: 
        uniques = []
    filtered = []
    
    for p in tqdm(personas, total=len(personas)):
        done, found = False, None
        i = 0
        while (not done) and len(uniques) > 0:
            p_cand_facts = set(uniques[i])
            num_overlap = sum([1 for fact in p if fact in p_cand_facts])
            if num_overlap > 1:
                done = True
                found = uniques[i]
            elif i+1 == len(uniques):
                done = True
            else:
                i += 1
        if found:
            filtered.append(found)
        else:
            uniques.append(p)
            filtered.append(p)
    print()
    print("%d unique persona profiles, %d unique persona facts found" %(len(uniques), 
                                                len(set(flatten(uniques)))
                                                                        )
          )
        
    return uniques, filtered

def _build_identifier_dataset(model, tokenizer, data):
    '''
    formats personachat dataframe into an array of samples.
    
    inputs:
        model: 
            transformer model (BERT, GPT, etc.) that embeds the tokens 
            corresponding to the relevant turns in the dialog history. 
            (default = personaGPT)
            
        tokenizer:
            associated tokenizer for the transformer. 
            (default = personaGPT)
            
        data_path: 
            file path for personachat data
            
    outputs:
        array of samples, each sample consists of a tuple,
            x1: 
                token embedding inputs for person 1 responses during dialog
            x2: 
                token embedding inputs for person 2 responses during dialog
            p1: 
                list of strings of person 1 personas
            p2: 
                list of strings of person 2 personas
    '''
    model.eval()    
    with torch.no_grad():
        x,p1,p2 = list(zip(*data))
        x1 = [[tokenizer.encode(xxx) for xxx in xx[::2]] for xx in x]
        x2 = [[tokenizer.encode(xxx) for xxx in xx[1::2]] for xx in x]
        x1 = [to_data(model(to_var(flatten(xx)).long(), 
                 output_hidden_states=True)[2][24].squeeze(0).mean(0)) for xx in x1]
        x2 = [to_data(model(to_var(flatten(xx)).long(), 
                 output_hidden_states=True)[2][24].squeeze(0).mean(0)) for xx in x2]
    return list(zip(x1,x2, p1,p2))


def prepare_persona_dataset(model, tokenizer, data_path):
    '''
    hash map of persona facts -> vector embeddings of persona facts
    
    inputs:
        model: 
            transformer model (BERT, GPT, etc.) that embeds tokens from facts. 
            (default = personaGPT)
            
        tokenizer:
            associated tokenizer for the transformer. 
            (default = personaGPT)
            
        data_path: 
            file path for personachat data
            
    outputs:
        p2v: 
            dictionary with unique persona facts as keys and 
            embeddings of persona facts in R^n as values. (default n = 1024)
            
        vec_train_data:
            training set of (x1, x2, p1, p2) samples to train identifier
            
        vec_test_data:
            test set of (x1, x2, p1, p2) samples to test identifier
    '''
    print()
    print("*"*50)
    print("Extracting data from personachat dataframe...")
    tr_data, te_data = _df_to_array(tokenizer, data_path)
    # build p2v
    x_tr, tr_p1, tr_p2 = list(zip(*tr_data))
    x_te, te_p1, te_p2 = list(zip(*te_data))
    # filter out redundant personas
    tr_uniques, tr_p1_filtered = _filter_personas(tr_p1)
    tr_uniques, tr_p2_filtered = _filter_personas(tr_p2, tr_uniques)
    te_uniques, te_p1_filtered = _filter_personas(te_p1)
    te_uniques, te_p2_filtered = _filter_personas(te_p2, te_uniques)
    persona_facts = list(set(flatten(tr_uniques + te_uniques)))
    # remake datasets w/ filtered personas 
    tr_data = list(zip(x_tr, tr_p1_filtered, tr_p2_filtered))
    te_data = list(zip(x_te, te_p1_filtered, te_p2_filtered))
    model.eval()
    print("Making persona dictionary ... ")
    vectors = []
    with torch.no_grad():
        for line in tqdm(persona_facts):     
            outp = model(**tokenizer(line, return_tensors='pt').to(device), output_hidden_states=True)
            vectors.append(outp[2][24].squeeze(0).mean(0))
        vectors = torch.stack(vectors)
    # save p2v to local dir
    p2v = dict([ (persona_facts[_], to_data(vectors[_])) for _ in range(len(persona_facts))])
    with open(os.path.join(opts.example_path, 'p2v'), 'wb') as f:
        pickle.dump(p2v, f)
        
    # build identifier datasets     
    print()
    print("Preparing identifier training set ... ")
    vec_train_data = _build_identifier_dataset(model, tokenizer, tr_data)
    print()
    print("Preparing identifier test set ... ")
    vec_test_data = _build_identifier_dataset(model, tokenizer, te_data)
    print("Done.")
    print('*'*50)
    print()
    # saving 
    with open(os.path.join(opts.example_path, 'vec_train_data'), 'wb') as f:
        pickle.dump(vec_train_data, f)
    with open(os.path.join(opts.example_path, 'vec_test_data'), 'wb') as f:
        pickle.dump(vec_test_data, f)

    return p2v, vec_train_data, vec_test_data