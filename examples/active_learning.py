#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:30:06 2021

@author: af1tang
"""
# import default language models for decoding
from convogym.decoders import model, tokenizer
# gym for active learning
from convogym.gyms import ActiveGym

# build an old training data (default is personachat training set) 
# used during active learning to prevent catastrophic forgetting
from convogym._configs import opts
from convogym.load_data import prepare_personachat_dataset
train_data, _ = prepare_personachat_dataset(model, tokenizer)

# define new turn-level goals to learn
new_goals = ['talk about pokemon.', 'ask about favorite anime.']
# when `train_mode=True`, performs gradient descent on each user correction
gym = ActiveGym(model=model, tokenizer=tokenizer, action_space=new_goals,
								 training_data=train_data, train_model=True)
gym.sim_convos(1)

# save data
gym.save_data(opts.example_path)
