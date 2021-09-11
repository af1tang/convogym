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
    """
    converts dataframe -> array of [tokens, p1, p2] 
    
    Parameters
    -------
    data_path : String or os.path
        path to pandas dataframe of history (hx), personas (p1 and p2), 
            and split (train or test) as columns
    Results
    -------
    tr_data : List of tuples
        - list of rows in df, where hx is represented as token_ids, 
        - p1 and p2 are list of strings. 
        - selected rows where split == train.
            
    te_data: List of tuples
        - selected rows where split != train
    """
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
    """
    filters out redundant profiles from a list of personas.

    Parameters
    ----------
    personas : List of list of string
        List of persona profiles, each profile consists of 3-5 list of strings.
        
    uniques : List of string, optional
        Set of unique persona profiles. The default is None.

    Returns
    -------
    uniques : List of string, optional
        Set of unique persona profiles. The default is None..
        
    filtered : List of list of string, optional
        original list of personas, with each profile replaced by filtered personas.

    Examples
    -------
    >>> personas = [['i like wine', 'i have 5 kids'], 
                        ['i hate wine', 'i like ice cream'],
                        ['i like wine', 'i've 5 kids]] 
    >>> uniques, filtered = _filtered_personas) 
    >>> print(filtered)
    [[['i like wine', 'i have 5 kids'], 
      ['i hate wine', 'i like ice cream'], 
      ['i like wine', 'i have 5 kids']]
     
    """
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

def _build_persona_batches(model, tokenizer, data):
    """
    Formats (dialog history, person 1 facts, person 2 facts) into turn-level batches for training.

    Parameters
    ----------
    model : A transformer decoder model e.g., GP2LMHeadModel 
        Transformer model (BERT, GPT, etc.) that embeds the tokens corresponding to the relevant turns in the dialog history. 
        (default = af1tang/personaGPT).
        
    tokenizer : A transformer tokenizer e.g., GPT2Tokenizer
        Associated tokenizer for the transformer. 
        (default = af1tang/personaGPT)        
        
    data : List of tuples
        List of training batches in (tokens, persona 1, persona 2) format, where persona 1 and persona 2 are list of strings.

    Returns
    -------
    result : List of tuples
        Each sample is a (x,y) pair:
            - x: token IDs of persona facts + dialog history up to current turn
            - y: token IDs of response at current turn.
        
        A dialog history of 8 turns (1 response from person 1 + 1 response from person 2 per turn) is broken up into 16 input output pairs. 
        
    Examples
    -------
    Suppose we have the following convo (a tuple) from PersonaChat:
    >>> dialog_tokens = ["hi how is it going?", # pre-tokenized
                         "not bad, how are you?",
                         "i'm having fun on fifa, you?",
                         "i'm tired from farming."]
    >>> dialog_hx = [tokenizer.encode(line) for line in dialog_tokens]
    >>> print(dialog_hx)
    [[5303, 703, 318, 340, 1016, 30],
     [1662, 2089, 11, 703, 389, 345, 30],
     [72, 1101, 1719, 1257, 319, 5515, 64, 11, 345, 30],
     [72, 1101, 10032, 422, 16035, 13]]
    >>> p1 = ["i like to play video games.", 
              "i want a cat someday."]
    >>> p2 = ["i work on a large farm.",
              "i don't have health insurance"]
    
    This convo gets converted to the following input-output pairs (x,y):
    >>> for i, (x,y) in enumerate(convo):
    >>>     print('-'*10+'turn %d'%i + '-' *10)
    >>>     print('input (x): ' , tokenizer.decode(x))
    >>>     print('output (y): ', tokenizer.decode(y))
    >>>     print()
    ----------turn 0----------
    input (x):  <|p1|> i like to play video games.i want a cat someday. <|sep|>
    output (y):  <|start|> hi how is it going?
    
    ----------turn 1----------
    input (x):  <|p2|> i work on a large farm.i don't have health insurance <|sep|> <|start|> hi how is it going?
    output (y):  not bad, how are you?
    
    ----------turn 2----------
    input (x):  <|p1|> i like to play video games.i want a cat someday. <|sep|> <|start|> hi how is it going?not bad, how are you?
    output (y):  i'm having fun on fifa, you?
    
    ----------turn 3----------
    input (x):  <|p2|> i work on a large farm.i don't have health insurance <|sep|> <|start|> hi how is it going?not bad, how are you?i'm having fun on fifa, you?
    output (y):  i'm tired from farming.        
    """
    result = []
    for (dialog_hx, p1, p2) in data:
        dialog_hx = dialog_hx[:20] # limit by max len of convo
        p1_ctx = tokenizer.encode(''.join(['<|p1|>'] + p1 + ['<|sep|>'] + ['<|start|>']))
        p2_ctx = tokenizer.encode(''.join(['<|p2|>'] + p2 + ['<|sep|>'] + ['<|start|>']))
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
            result.append((x,y))
    return result

def prepare_personachat_dataset(model, tokenizer, data_path):
    """
    Converts persona dataset into language model task format.

    Parameters
    ----------
    model : A transformer decoder model e.g., GP2LMHeadModel 
        Transformer model (BERT, GPT, etc.) that embeds the tokens corresponding to the relevant turns in the dialog history. 
        (default = af1tang/personaGPT).
        
    tokenizer : A transformer tokenizer e.g., GPT2Tokenizer
        Associated tokenizer for the transformer. 
        (default = af1tang/personaGPT)   
        
    data_path : String or os.path
        File path for personachat data.

    Returns
    -------
    tr_batches : List of tuples
        List of training set samples.
        Each sample is a (x,y) pair:
            - x: token IDs of persona facts + dialog history up to current turn
            - y: token IDs of response at current turn.
        
        A dialog history of 8 turns (1 response from person 1 + 1 response from person 2 per turn) is broken up into 16 input output pairs. 
    
    te_batches : List of tuples
        List of test set samples.
        Each sample is a (x,y) pair:
            - x: token IDs of persona facts + dialog history up to current turn
            - y: token IDs of response at current turn.
        
        A dialog history of 8 turns (1 response from person 1 + 1 response from person 2 per turn) is broken up into 16 input output pairs. 
    
    """
    print()
    print("*"*50)
    print("Extracting data from personachat dataframe...")
    tr_data, te_data = _df_to_array(tokenizer, data_path)
    # build training and testing batches
    print("Building traing and testing batches from conversations...")
    tr_batches = _build_persona_batches(model, tokenizer, tr_data)
    te_batches = _build_persona_batches(model, tokenizer, te_data)
    # saving
    with open(os.path.join(opts.example_path, 'train_decoder_data'), 'wb') as f:
        pickle.dump(tr_batches, f)
    with open(os.path.join(opts.example_path, 'test_decoder_data'), 'wb') as f:
        pickle.dump(te_batches, f)
    print("done!")
    print()
    return tr_batches, te_batches

def prepare_identifier_dataset(model, tokenizer, data_path):
    """
    Formats personachat dataframe into an array of training samples for unsupervised learning. 
    
    In unsupervised learning, embedding functions are learned for
        - the dialog history
        - the persona facts.
        
    Embeddings of the dialog history and persona facts can be used for state and context representations.


    Parameters
    ----------
    model : A transformer decoder model e.g., GP2LMHeadModel 
        Transformer model (BERT, GPT, etc.) that embeds the tokens corresponding to the relevant turns in the dialog history. 
        (default = af1tang/personaGPT).
        
    tokenizer : A transformer tokenizer e.g., GPT2Tokenizer
        Associated tokenizer for the transformer. 
        (default = af1tang/personaGPT)   
        
    data_path : String or os.path
        File path for personachat data.
    
    Saves
    -------
    tr_data : List of tuples
        List of training set samples. 
        Each sample is a tuple (dialog history, persona 1, persona 2):
            - dialog history : list of token ids for each conversational turn
            - persona 1 : list of strings of person 1 persona facts
            - persona 2 : list of strings of person 2 persona facts.
        
    te_data : List of tuples
        List of training set samples. 
        Each sample is a tuple (dialog history, persona 1, persona 2):
            - dialog history : list of token ids for each conversational turn
            - persona 1 : list of strings of person 1 persona facts
            - persona 2 : list of strings of person 2 persona facts.

    Returns
    -------
    persona_facts : List of strings
        List of unique persona facts in personachat.
    """
    print()
    print("*"*50)
    print("Extracting data from personachat dataframe...")
    tr_data, te_data = _df_to_array(tokenizer, data_path)
    # build filter personas to get unique persona facts
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
    # saving
    df_facts = pd.DataFrame(persona_facts, columns=['Facts'])
    df_facts.to_csv(os.path.join(opts.example_path, 'persona_facts.csv'), index=False)
    with open(os.path.join(opts.example_path, 'train_state_estim_data'), 'wb') as f:
        pickle.dump(tr_data, f)
    with open(os.path.join(opts.example_path, 'test_state_estim_data'), 'wb') as f:
        pickle.dump(te_data, f)
    return persona_facts
    