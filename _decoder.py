#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:16:09 2021

@author: af1tang
"""
from transformers import GPT2LMHeadModel
from _configs import opts
from utils._device import to_device

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