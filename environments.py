#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:00:29 2021

@author: af1tang
"""
import torch
from utils._device import to_var

class Env:        
    """
    A helper object for callbacks to estimate the state information at each turn during a dialog. 
    
    Parameters
    -------
    scb : StateCb object
        A callback object that tracks the states, contexts, actions objects that are updated during conversation.
        
    mcb : MessageCb object
        A callback object that tracks raw text, turn count and status of the conversation.
        
    state_estimator : object or function
        A custom state estimation object or function that maps from dialog history tokens (or token embeddings) to state vectors.
    
    state_size : int, optional
        Number of dimensions in the state vector.
    """
    def __init__(self, state_estimator, state_size = 1024):
        self.state_estimator = state_estimator
        self.state_size = state_size
        
    def reset(self, scb):
        """
        Initializes the state and context vectors to the 0 vectors. 
        """
        scb.state = to_var(torch.zeros(self.state_size))
        scb.states, scb.rewards, scb.actions = [scb.state.tolist() + [0.0]], [], []
        return scb
    
    def get_policy_inp(self, scb):
        """
        Formats the turn count and state into a single vector input to the policy.
        """
        turn_tensor = to_var(torch.ones(1,1)) * scb.turn
        state = scb.state.view(1,-1)
        state_t = torch.cat((state, turn_tensor), dim=-1)
        return state_t
    
    def get_curr_state(self, scb, mcb):
        """
        Use state estimator to obtain state and context vectors from dialog history text.
        
        Notes
        -----
        Only the person 2 responses are used to obtain the state and context vectors corresponding to person 2.
        """
        with torch.no_grad():
            scb.state = self.state_estimator(dialog_history=mcb.dialog_hx[1::2])
        return scb, mcb
        
    def get_next_state(self, scb, mcb):
        """
        Not implemented. Passes new state estimates to state callbacks.
        """
        _state = scb.state.tolist()
        _state.append(scb.turn+1.0)
        scb.states.append(_state)
        return scb, mcb