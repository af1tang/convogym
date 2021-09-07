#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:02:24 2021

@author: af1tang
"""

## reward func helpers 
default_reward_func = lambda state, context, personas: 0.0

class Reward:
    def __init__(self, reward_func=default_reward_func):
        self._calculate_reward = reward_func
    
    def calculate_reward(self, scb, mcb):
        scb.reward = self._calculate_reward(state=scb.state, context=scb.context, 
                                            personas=scb.personas)
        scb.rewards.append(scb.reward)
        return scb, mcb

