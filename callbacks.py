#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 00:21:08 2021

@author: af1tang
"""

## state and dialog trackers 
class StateCb:
    def __init__(self, state=None, context=None, personas = None,
                 action=None, reward=None, act=False):
        self.state, self.context, self.action, self.reward, self.act = state, context, action, reward, act
        self.states, self.contexts, self.actions, self.rewards = [], [], [], []
        self.personas = personas
        self.turn, self.done = 0, False

class MessageCb:
    def __init__(self, msg=None, x=None, y=None):
        self.msg, self.x, self.y, self.dialog_hx = msg, x, y, []