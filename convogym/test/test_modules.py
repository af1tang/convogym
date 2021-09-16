#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:05:11 2021

@author: af1tang
"""

import pytest
from convogym.policies import Policy # default policy
from convogym.utils._device import to_var
import torch, torch.nn as nn


def test_policy():
    action_space = [ 
        'talk about work.', 
        'ask about hobbies.',
        'talk about movies.'
        ]
    policy_net = nn.Sequential(nn.Linear(100, len(action_space))
                                )
    policy = Policy( policy = policy_net,
                        action_space = action_space) 
    action = policy.act( inp=to_var(torch.rand((1,100))), turn=1 ) 
    assert action in range(len(action_space))
    