#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:32:07 2021

@author: af1tang
"""
from convogym.gyms import Gym
# default persona model is af1tang/personaGPT from huggingface
from convogym.decoders import model, tokenizer
# get persona function, in this case randomly samples from personachat dataset
from prefixes import get_random_persona 

gym = Gym(model = model, tokenizer=tokenizer, interactive = False,
          reset_persona_func=get_random_persona, length=10)
# simulate 3 conversations
gym.sim_convos(3)