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
from convogym._configs import opts

# default special tokens
eos_token = '<|endoftext|>'

# load train and test personas
tr_df = pd.read_csv(os.path.join(opts.data_path, 'train_personas.csv'))
te_df = pd.read_csv(os.path.join(opts.data_path, 'test_personas.csv'))

train_personas = []
for i, row in tr_df.iterrows():
    train_personas.append([fact + eos_token for fact in row.dropna().tolist()])

test_personas = []
for i, row in te_df.iterrows():
    test_personas.append([fact + eos_token for fact in row.dropna().tolist()])

    
# persona sampling functions 
def load_persona_facts(save_path=None):
    """
    Load a set of unique persona facts from save file.

    Parameters
    ----------
    save_path : os.path or string, optional
        Loads .csv format file with "Facts" as the column name for persona facts. 
        If none provided, constructs a list of unique persona facts from the PersonaChat dataset. The default is None.

    Returns
    -------
    persona_facts : List of strings
        DESCRIPTION.

    """
    if save_path:
        try:
            df_facts = pd.read_csv(save_path)
            persona_facts = df_facts['Facts'].tolist()
        except: 
            try: 
                import pickle
                with open(save_path, 'rb') as f:
                    persona_facts = pickle.load(f)
            except Exception as e:
                warnings.warn( e )
                print(""" 
                      Unable to load person facts from file. 
                      
                      If file is .csv, make sure the persona facts column is labeled "Facts".
                      Otherwise try serializing the list of persona facts and saving it as a pickle object.
                      """)
    else:
        try:
            df_facts = pd.read_csv(os.path.join(opts.example_path, 'persona_facts.csv'))
            persona_facts = df_facts['Facts'].tolist()
        
        except Exception as e:
            warnings.warn(str(e))
            from load_data import prepare_state_estim_dataset
            from decoders import model, tokenizer
            print()
            print("Building unique persona facts from personachat dataset.")
            persona_facts = prepare_state_estim_dataset(model, tokenizer,
                                                       os.path.join(opts.data_path, 'personachat.csv'))
    return persona_facts

def get_custom_persona(n=3, save_path=None, *args, **kwargs):
    """
    Get a set of persona facts from user inputs.

    Parameters
    ----------
    n : TYPE, optional
        Number of persona facts to input. The default is 3.
        
    save_path : os.path or string, optional
        Path to save custom persona set to. If none, personas will not be saved. The default is None.

    Returns
    -------
    personas : List of strings.
        A single set of persona facts (3-5) from a list of personas.

    """
    personas = []
    for i in range(n):
        response = ""
        while len(response) <1:
            response = input(">> Fact %d: "%(i+1))+ eos_token
        personas.append(response)
    if save_path:
        try:
            df = pd.DataFrame(personas, columns=['Facts'])
            df.to_csv(save_path, index=False)
        except Exception as e:
            warnings.warn(e)
            print("Could not save new persona list to save file.")
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