#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:22:53 2020

@author: af1tang
"""
import torch, torch.nn.functional as F
from convogym.load_data import default_action_space as action_space
from convogym.utils._reshape import flatten
from convogym.utils._device import to_var
from convogym.utils._sampling import top_k_top_p_filtering

#### Baseline Agent ####
class Agent(object):
    """
    Wrapper object for the decoder model to do conditional decoding (no gradients). 
    
        - An Agent object can take on either the role of person 1 (user / policy) or person 2 (agent). 
        - The decoder model is assumed to be a transformer decoder (e.g., GPT2LMHeadModel), so top-k / nucleus sampling is used during token generation. 

    Parameters
    ----------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT.
        
    tokenizer : Huggingface transformer tokenizer (default = GPT2Tokenizer).
        The corresponding tokenizer for the transformer model. 
        The default tokenizer is af1tang/personaGPT.
        
    personas : List of strings
        List of persona facts to be included in the prefix tokens inputs to the decoder model.
        
    reverse : bool, optional
        If True, the Agent object takes on the role of person 1 (user / policy). The default is False.
    
    top_k : int, optional
        Number of top candidates to consider (sample from) the conditional distribution at each decoding step. The default is 10.
        
    top_p : float, optional
        Nucleus sampling parameter. Top fraction (0-1.0) of the conditional probability mass function to sample from at each decoding step. The default is .92.
        
    max_length : int, optional
        Max number of tokens allowed to decode in a given conversation. The default is 1024.

    Attributes
    -------
    p1 : List of strings
        List of persona facts corresponding to person 1.
        
    p2: List of strings
        List of persona facts corresponding to person 2.
        
    reversed : bool
        If True, the Agent object takes on the role of person 1 (user / policy). The default is False.
    
    inp : torch.LongTensor
        Tensor (1 x num tokens) of input tokens to the decoder model:
            - persona token <|p1|> or <|p2|> indicating the person identity OR
                action token <|act|> followed by the turn-level goal
            - persona facts separated by <|endoftext|> tokens
            - <|sep|> token separates persona facts or turn-level goal from dialog history
            - <|start|> token denotes start of dialog history
            - dialog history up to current turn.
        
        The input tokens can be thought of as the prompt used by the decoder (language model) to autoregressively decode the next set of tokens that comprise the response.
    
    past : torch.Tensor
        Cached hidden states of each layer in the transformer for quicker inference.
    
    turn : int 
        Turn count in the current conversation.
        
    curr_len : int
        Total number of tokens so far in the current conversation.
        
    dialog_history : List of list of ints
        List of token IDs for each response in the conversation.

    """
    def __init__(self, model, tokenizer, persona, reverse = False,
                 top_k=10, top_p = .92, max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k, self.top_p, self.max_length = top_k, top_p, max_length
        self.reversed = reverse
        if not self.reversed:
            self.p1 = [] # self.NA_token # don't know your partner's persona
            self.p2 = persona
        else:
            self.p2 = [] #self.NA_token
            self.p1 = persona
        self.reset_convo()
        
    def __call__(self, inp, act=False):
        """
        Calls self.step(inp, act).

        Parameters
        ----------
           
        inp : List of ints
            Token IDs of the input message.
            
        act : bool, optional
            Whether the input prefix is an action code (True) or a set of persona facts (False). The default is False.

        Returns
        -------
        List of ints
            Token IDs of the output message.

        """
        return self.step( inp, act)
        
    def reset_convo(self):
        """
        Resets the dialog history and turn count. Used at beginning of conversations.
        """
        # reset dialog history
        self.dialog_history, self.turn = [], 0
        
    def _update_prefix(self, action):
        """
        Updates the prefix tokens with the input turn-level goal description.

        Parameters
        ----------
        action : List of strings
            A list with a single string representing the turn-level goal.

        Raises
        ------
        NotImplementedError
            Raises an error if the input action is not in the support set of actions.

        Returns
        -------
        None.

        """
        if action not in action_space:
            raise NotImplementedError("this action is not currently in the set of learned action space")
        else:
            if self.reversed:
                self.p2 = [action]
            else:
                self.p1 = [action]
    
    def _reset_inp(self, act = False):
        """
        Formats the input to the agent. If self.reversed=True, use person 1 facts, otherwise use person 2 facts. 
        
        If act=True, self.p1 or self.p2 represents the turn-level goal rather than a list of persona facts. 

        Parameters
        ----------
        act : bool, optional
            Whether the input prefix is an action code (True) or a set of persona facts (False). The default is False.

        Returns
        -------
        None.

        """
        if not act:
            if self.reversed:
                self.inp = self.tokenizer.encode(''.join(['<|p1|>'] + self.p1 + ['<|sep|>'] + ['<|start|>']))
            else:
                self.inp = self.tokenizer.encode(''.join(['<|p2|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
        else:
            if self.reversed:
                self.inp =  self.tokenizer.encode(''.join(['<|act|> '] + self.p2 + ['<|sep|>'] + ['<|p1|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
            else:
                self.inp =  self.tokenizer.encode(''.join(['<|act|> '] + self.p1 + ['<|sep|>'] + ['<|p1|>'] + self.p2 + ['<|sep|>'] + ['<|start|>']))
        # incorporate dialog dx
        self.inp += flatten(self.dialog_history)
        self.inp, self.curr_len, self.past = to_var(torch.tensor([self.inp])), len(self.inp), None

    def _update_dialog_hx(self, new_inp):
        """
        Updates the dialog history with new input message.

        Parameters
        ----------
        new_inp : List of ints
            List of tokens corresponding to the input message.

        Returns
        -------
        None.

        """
        if new_inp is not None:
            self.dialog_history.append(new_inp)
        
    def step(self, inp, act=False):
        """
        Generates a response message based on input message.

        Parameters
        ----------

        inp : List of ints
            Token IDs of the input message.
            
        act : bool, optional
            Whether the input prefix is an action code (True) or a set of persona facts (False). The default is False.

        Returns
        -------
        List of ints
            Token IDs of the output message.
            
        """
        self._update_dialog_hx(inp)
        self._reset_inp(act)
        outp = []
        self.model.eval()
        with torch.no_grad():
            while (self.tokenizer.eos_token_id not in outp) and (self.curr_len + len(outp) < self.max_length):
                outputs = self.model(self.inp, past_key_values=self.past)
                logits, self.past = outputs.logits, outputs.past_key_values
                # top k sampling          
                log_scores = top_k_top_p_filtering(logits[:,-1,:], top_k=self.top_k, top_p=self.top_p)
                probs = F.softmax(log_scores, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(1)
                # update tokens for next output
                outp += token.tolist()
                self.inp = token.unsqueeze(0) 
                self.curr_len+=1
        self.dialog_history.append(outp)
        self.turn+=1
        return outp
    