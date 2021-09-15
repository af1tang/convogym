#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 01:58:57 2021

@author: af1tang
"""

import pytest
from convogym import decoders
from convogym.utils._device import to_var

def test_loading_model_tokenizer():
    assert len(decoders.tokenizer) == 50263
    assert ( decoders.act_tok == 50262 ) and ( decoders.start_tok == 50259)
    assert ( decoders.p1_tok == 50260 ) and ( decoders.p2_tok == 50261)
    
def test_default_tokenizer():
    goal = "ask about hobbies."
    inp = decoders.tokenizer.encode("<|act|>" + goal + "<|p1|><|sep|><|start|>")        
    assert inp == [50262, 2093, 546, 45578, 13, 50260, 50257, 50259]
    
    