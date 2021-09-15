#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:32:22 2021

@author: af1tang
"""
import torch
from convogym.utils._device import device, to_var

def fit_on_batch(model, batch, gradient_accumulation_steps=8):
    """
    Calculate loss on batch, update gradient and backpropagate.

    Parameters
    ----------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT model card.
        
    batch : tuple of torch.LongTensors
        Format of a batch tuple is (xx,yy): 
            - xx: List of int
                Input tokens (encoded by tokenizer) of action prefix + dialog history.
            - yy: List of int
                Encoded human-provided response (ground truth).
        
    gradient_accumulation_steps : int, optional
        Number of gradient accumulation steps. The default is 8.

    Returns
    -------
    loss : float
        Loss on the current batch.

    """
    xx,yy = batch
    try:
        xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
    except:
        xx, yy = to_var(xx).long(), to_var(yy).long()
    ## forward on new data batch
    _outp = model(xx)
    past = _outp.past_key_values
    outp = model(yy, past_key_values=past, labels=yy)
    
    # backward
    loss = outp[0]; del outp
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss

def _process_dqn_batch(batch, policy, swa_policy, gamma):
    """
    Processes a batch of (state, context, action, next_state, next_action) -> (inputs, actions, Q-targets).

    Parameters
    ----------
    batch : tuple of torch.Tensor objects
        (state, context, action, next_state, next_action), each a torch.Tensor object
        
    policy : Callable OR nn.Module
        The policy network that maps from (state, context) -> action.
        
    swa_policy : Callable or nn.Module
        The target network used to calculate the Q-target.
        
    gamma : float (0,1.0)
        Discount factor when calculating Q-values.

    Returns
    -------
    xx : torch.Tensor
        Policy input.
        
    y : torch.LongTensor
        Index of the action sampled from action space. The Q-value corresponding to the action index is used to calculate the gradient for backpropagation.
        
    q_targets : torch.Tensor
        Scalar Q-values are calculated by the target network across each sample in the batch. These are treated as labels (no gradients through the Q-targets) for temporal difference learning.

    """
    try:
        x, y, x_next, y_next, r, dones = batch
        y_next = to_var(y_next)

    except:
        x, y, x_next, y_next, r, dones = batch
    xx = to_var(torch.stack(x, dim=-1).type(torch.FloatTensor))# torch.stack(c, dim=-1).type(torch.cuda.FloatTensor)
    xx_next = to_var(torch.stack(x_next, dim=-1).type(torch.FloatTensor)) #torch.stack(c_next, dim=-1).type(torch.cuda.FloatTensor)
    y, r, dones = to_var(y), to_var(r.type(torch.FloatTensor)), to_var(dones.long())
    #xx = torch.cat((x,c), dim=-1)
    #xx_next = torch.cat((x_next, c_next), dim=-1)
    # calculate q-values
    with torch.no_grad():
        # use target network to predict q-targets
        q_values = policy(xx_next)
        idx = q_values.max(1)[1]
        q_values = swa_policy(xx_next)
        q_targets = r + (1-dones) * gamma * q_values[torch.arange(len(idx)), idx]
    return xx, y, q_targets
