#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:14:47 2021

@author: af1tang
"""
# import default language models for decoding
from convogym.decoders import model, tokenizer
# gym for reinforcement learning
from convogym.gyms import RLGym
# import modules for RL
from convogym.environments import Env
# base state estimator aggregates hidden state info of input tokens from decoder
from convogym.states import BaseStateEstimator
# manual reward gets user input to rate episode
from convogym.rewards import ManualReward
# DQN policy uses batch q learning to update policy
from convogym.policies import DQNPolicy
# define an action space for the policy (default is designed for personachat)
from convogym.load_data import default_action_space

state_estimator = BaseStateEstimator(model=model, tokenizer=tokenizer)
gym = RLGym( model=model, tokenizer=tokenizer, 
				 policy=DQNPolicy(action_space=default_action_space),
				 env=Env(state_estimator),
				 reward_obj=ManualReward(state_estimator),
		  )
# simulate and rate 3 convos
gym.sim_convos(num_convos=3, training=True)

# save data
import os
from convogym._configs import opts
gym.save_data(os.path.join(opts.example_path, 'rl_data.csv'))