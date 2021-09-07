#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:21:35 2020

@author: af1tang
"""
from dotenv import load_dotenv
import os, torch, pickle

load_dotenv(verbose=True)
# paths and configs
base_path = os.path.join(os.environ['HOME'], 'convogym/')
save_path = os.path.join(base_path, 'checkpoint')
model_path = os.path.join(base_path, 'checkpoint/model/')
data_path = os.path.join( os.path.abspath(os.getcwd()), 'data/')
example_path = os.path.join(save_path, 'example/')
# learning
lr = os.getenv("learn_rate")
gradient_accumulation_steps = os.getenv("gradient_accumulation_steps")
bs = os.getenv("batch_size")
epochs = os.getenv("epochs")
weight_decay = os.getenv("weight_decay")
logging_steps = os.getenv("logging_steps")
save_steps = os.getenv("save_steps")
num_cands = os.getenv("num_cands")
# convo params
num_personas = os.getenv("num_personas")

def create_dir(directory):
    """
    Create directory if not exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# initialize save folder
create_dir(base_path)
create_dir(save_path)
create_dir(model_path)
create_dir(example_path)

class Configs():
    def __init__(self):
        # saving and loading paths
        self.model_path = model_path
        self.data_path = data_path
        self.save_path = save_path
        self.example_path = example_path
        self.plot_path = os.path.join(save_path,'samples/')
        self.download_name = 'af1tang/personaGPT'
        # persona params
        self.num_personas = int(num_personas)
        # identifier training params
        self.num_epochs = int(epochs)
        self.save_steps = int(save_steps)
        self.loggin_steps = int(logging_steps)
        self.lr = float(lr)
        self.logging_steps = int(logging_steps)
        self.bs = int(bs)
        self.num_cands = int(num_cands)
        
opts = Configs()

action_space = [ 'ask about kids.', "ask about pets.", 'talk about work.', 
           'ask about marital status.', 'talk about travel.', 'ask about age and gender.',
    'ask about hobbies.', 'ask about favorite food.', 'talk about movies.', 
    'talk about music.', 'talk about politics.']


