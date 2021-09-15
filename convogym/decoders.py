#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:16:09 2021

@author: af1tang
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from convogym._configs import opts
from convogym.utils._device import to_device

try: 
    print("*"*50)
    print("Loading decoder from pretrained")
    model = GPT2LMHeadModel.from_pretrained(opts.model_path)
    print("*"*50)
except Exception as e:
    print(e)
    print("*"*50)
    print("Downloading decoder... ")
    print("*"*50)
    # download decoder
    model = GPT2LMHeadModel.from_pretrained(opts.download_name)
    # save locally
    model.save_pretrained(opts.model_path)
    
model = to_device(model)

try: 
    print("*"*50)
    print("Load tokenizer from pretrained")
    tokenizer = GPT2Tokenizer.from_pretrained(opts.model_path, 
                                            pad_token='<|endoftext|>', cls_token='<|cls|>',
                                            sep_token='<|sep|>')
    print("*"*50)
except Exception as e:
    print(e)
    print("*"*50)
    print("Downloading tokenizer ... ")
    print("*"*50)
    # download personaGPT
    tokenizer = GPT2Tokenizer.from_pretrained(opts.download_name, 
                                        pad_token='<|endoftext|>', cls_token='<|cls|>',
                                        sep_token='<|sep|>')
    # save to local
    tokenizer.save_pretrained(opts.model_path)

# special tokens    
p1_tok, p2_tok, start_tok = tokenizer.encode('<|p1|>')[0], tokenizer.encode('<|p2|>')[0], tokenizer.encode('<|start|>')[0]

# action token
act_tok = tokenizer.encode('<|act|>')[0]

