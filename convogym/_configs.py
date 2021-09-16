#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:21:35 2020

@author: af1tang
"""
from dotenv import load_dotenv
import pkg_resources
import os, torch, pickle

load_dotenv(verbose=True)
# paths and configs
base_path = os.path.join(os.environ['HOME'], 'convogym/')
save_path = os.path.join(base_path, 'checkpoint')
model_path = os.path.join(base_path, 'checkpoint/model/')
data_path = pkg_resources.resource_filename('convogym', 'data/')
example_path = os.path.join(save_path, 'example/')


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

opts = Configs()


