#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:19:04 2021

@author: af1tang
"""
import os
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