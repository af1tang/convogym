#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:20:35 2021

@author: af1tang
"""
import os
import matplotlib.pyplot as plt
from _configs import opts, create_dir
from _tokenizer import tokenizer

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()
            
### plotting ###    
def plot_losses(stats, title='loss'):
    create_dir(opts.plot_path)
    x = list(sorted(stats.keys()))
    loss = [stats[i][title] for i in x]
    plt.plot(x, loss, label= title)
    plt.legend()
    plt.title("%s" %title)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.plot_path,'%s.png'%title))
    plt.close()