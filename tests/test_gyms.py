#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:42:59 2021

@author: af1tang
"""

import pytest
from convogym.decoders import model, tokenizer
from convogym.gyms import Gym
from convogym.prefixes import get_custom_persona, get_random_persona


def test_self_play():
    gym = Gym(model=model, tokenizer=tokenizer, interactive=False, 
              reset_persona_func=get_random_persona, length=4)
    gym.sim_convos(num_convos=3)
    assert len(gym.data['hx']) == 3
    assert sum([ len(gym.data['hx'][i]) for i in range(3)]) /2 ==12
    
