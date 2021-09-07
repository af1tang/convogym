#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:02:05 2021

@author: af1tang
"""

## logging metric helpers
default_scoring_func = lambda states, contexts, personas: 0.0

class Logger:
    def __init__(self, score_func=default_scoring_func):
        self.evaluate = score_func
        self.metrics = {'losses': []}
        
    def __call__(self, scb, mcb):
        '''update metrics'''
        self.metrics['losses'].append(self.evaluate(scb.states, scb.contexts, scb.personas))
        return scb, mcb