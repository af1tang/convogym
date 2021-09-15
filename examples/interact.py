#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:00:13 2021

@author: af1tang
"""
from convogym.gyms import Gym
# default persona model is af1tang/personaGPT from huggingface
from convogym.decoders import model, tokenizer
# get persona function, in this case user inputs persona 
from prefixes import get_custom_persona 

# interact with a persona model, using custom persona inputs
gym = Gym(model = model, tokenizer=tokenizer, interactive = True, 
          reset_persona_func=get_custom_persona, length=8)
gym.sim_convos(1)
