#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:21:16 2021

@author: af1tang
"""
from itertools import groupby
from _tokenizer import tokenizer

def filter_turn_indices(x):
    filtered = [[t[1] for t in list(g)] for k,g in groupby(list(enumerate(x)), lambda x: x[1]==tokenizer.eos_token_id) if not k]
    return filtered

def to_tokens(dialog_history):
    return [tokenizer.decode(line) for line in dialog_history]