#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:17:28 2020

@author: af1tang
"""
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(model):
    return model.to(device)

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

            
