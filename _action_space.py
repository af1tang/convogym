#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:16:42 2021

@author: af1tang
"""
import os
import pandas as pd
from _configs import opts

action_space = pd.read_csv(os.path.join(opts.data_path, 'action_space.csv'))['actions'].tolist()