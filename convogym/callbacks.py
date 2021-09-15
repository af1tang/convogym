#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 00:21:08 2021

@author: af1tang
"""

## state and dialog trackers 
class StateCb:
    """
    Callback object for tracking conversational state information such as (state, action, reward, next_state) at each turn of the conversation.

    Parameters
    ----------
    state : torch.Tensor, optional
        The feature vector embedding of the dialog history. Shape should be (1 x n) for a vector in R^n.
        In Markov Decision Process terms, this is the state input to the policy. The default is None.
        
        
    personas : List of strings, optional
        List of persona facts to track for scoring (if applicable). The default is None.
        
    actions : List of ints, optional
        List of action indices corresponding to actions taken per turn. The default is None.
        
    reward : float, optional
        Reward at a given turn. The default is None.
        
    act : string, optional
        Turn-level goal at a given turn in text form. The default is False.

    Attributes
    -------
    turn : int
        Tracks number of turns elapsed in current conversation.        
    
    done : bool
        Whether to end conversation. 
        
    rewards : List of floats
        Tracks reward received at each turn.

    """
    def __init__(self, state=None, personas = None,
                 actions=None, reward=None, act=False):

        self.state, self.action, self.reward, self.act = state, actions, reward, act
        self.states, self.actions, self.rewards = [], [], []
        self.personas = personas
        self.turn, self.done = 0, False

class MessageCb:
    """
    Tracks text information that flows throughout the conversation between the user and agent.

    Parameters
    ----------
    msg : List of ints, optional
        Input string message (tokenized) to a list of integers. The default is None.
        
    x : torch.LongTensor, optional
        Tracks the formatted prefix tokens (context / prompt) to the decoder. Used during ActiveGym to format the training data inputs. The default is None.
        
    y : torch.LongTensor, optional
        Tracks the formatted output tokens provided by human users. Used during ActiveGym to format the training data labels. The default is None.

    Attributes
    -------
    dialog_hx : List of list of ints
        List of token ids, each corresponding to a dialog response.
    """
    def __init__(self, msg=None, x=None, y=None):
        self.msg, self.x, self.y, self.dialog_hx = msg, x, y, []