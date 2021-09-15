#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:21:16 2021

@author: af1tang
"""
from itertools import groupby

def filter_turn_indices(x, eos_token_id):
    filtered = [[t[1] for t in list(g)] for k,g in groupby(list(enumerate(x)), lambda x: x[1]==eos_token_id) if not k]
    return filtered