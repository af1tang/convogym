#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 01:47:23 2021

@author: af1tang
"""
import os
from environments import Env
from policies import Verifier 
from rewards import IR_Reward
from gyms import Gym, ActiveGym, RLGym
from states import RankingStateEstimator

from _configs import opts
from _decoder import model
from _tokenizer import tokenizer
from _personas import ( 
                        train_personas, test_personas,
                        get_custom_persona, get_random_persona, 
                        get_sequence_personas
                        )            


# TODO: saved / augment training data from various gym environments
# TODO: make unit tests for 
    
    
# full interactive gym
# gym = Gym(model, tokenizer, interactive=True, 
#           reset_persona_func=get_custom_persona, length=3)

# self-play with custom persona 
# gym = Gym(model = model, tokenizer = tokenizer, interactive = False, 
#           reset_persona_func=get_custom_persona, length=8)

# self-play with random persona
# gym = Gym(model = model, tokenizer = tokenizer, interactive = False, 
#           reset_persona_func=get_random_persona, length=8)


# active learning gym (collecting new episodes open)
gym = ActiveGym(model = model, tokenizer = tokenizer, length = 4)

# active learning with fine tuning
# from _prepare_persona_data import prepare_personachat_dataset
# train_data, test_data = prepare_personachat_dataset(model, tokenizer, 
#                                          os.path.join(opts.data_path, 'personachat.csv'))
# gym = ActiveGym(model=model, tokenizer=tokenizer, training_data=train_data, 
#                 train_mode=True, length = 4)

# RL gym 
# state_estimator = RankingStateEstimator(model=model, tokenizer=tokenizer)
# policy = Verifier()
# env = Env(state_estimator)              # uses state stimator when obtaining state transitions
# reward = IR_Reward(state_estimator)     # uses state estimator when calculating rewards and metrics

# gym = RLGym(model=model, tokenizer=tokenizer, 
#             policy=policy, env=env, reward_obj=reward)

# simulate convos
gym.sim_convos(3)