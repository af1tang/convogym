#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:19:04 2021

@author: af1tang
"""
import os
import warnings
import random
import pandas as pd
from _configs import opts
from _tokenizer import tokenizer


# load train and test personas
tr_df = pd.read_csv(os.path.join(opts.data_path, 'train_personas.csv'))
te_df = pd.read_csv(os.path.join(opts.data_path, 'test_personas.csv'))

train_personas = []
for i, row in tr_df.iterrows():
    train_personas.append([fact + tokenizer.eos_token for fact in row.dropna().tolist()])

test_personas = []
for i, row in te_df.iterrows():
    test_personas.append([fact + tokenizer.eos_token for fact in row.dropna().tolist()])
    
try:
    df_facts = pd.read_csv(os.path.join(opts.example_path, 'persona_facts.csv'))
    persona_facts = df_facts['Facts'].tolist()
    
except Exception as e:
    warnings.warn(str(e))
    from _prepare_persona_data import prepare_identifier_dataset
    from _tokenizer import tokenizer
    from _decoder import model
    print()
    print("Building unique persona facts from personachat dataset.")
    persona_facts = prepare_identifier_dataset(model, tokenizer,
                                               os.path.join(opts.data_path, 'personachat.csv'))
    
# persona sampling functions 
def get_custom_persona(*args, **kwargs):
    """
    Get a set of persona facts from user inputs (default = 3).
    """
    personas = []
    for i in range(opts.num_personas):
        response = ""
        while len(response) <1:
            response = input(">> Fact %d: "%(i+1))+ tokenizer.eos_token
        personas.append(response)
    return personas

def get_random_persona(persona_list):
    """
    Samples a random persona profile (3-5 facts) from a list of input personas. 
    (Default = _personas.train_personas)

    Parameters
    ----------
    persona_list : List of list of strings.
        List of persona profiles from which to sample from.

    Returns
    -------
    List of strings.
        A single set of persona facts (3-5) from a list of personas.

    """
    if not persona_list:
        return random.sample(train_personas, 1)[0]
    else:
        return random.sample(persona_list, 1)[0]
        

def get_sequence_personas(persona_list):
    """
    Yields a set of persona facts (3-5) from a list of personas.
    (Default = _personas.train_personas)

    Parameters
    ----------
    persona_list : List of list of strings.
        List of persona profiles from which to sample from.

    Yields
    ------
    List of strings.
        A single set of persona facts (3-5) from a list of personas.

    Example 
    -----
    >>> from convogym._personas import train_personas
    >>> next(get_sequence_personas())
    
    ['i design video games for a living.<|endoftext|>',
     'i hate broccoli.<|endoftext|>',
     'i am afraid of the dark.<|endoftext|>',
     'my mom is my best friend.<|endoftext|>']
    """
    random.shuffle(persona_list)
    p_iter = iter(persona_list)
    while p_iter:
        try:
            yield next(p_iter)
        except:
            return