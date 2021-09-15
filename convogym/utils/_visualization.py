#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:20:35 2021

@author: af1tang
"""
import os
import matplotlib.pyplot as plt
from convogym._configs import opts, create_dir

def display_dialog_history(dialog_hx, tokenizer):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()

def to_tokens(dialog_history, tokenizer):
    return [tokenizer.decode(line) for line in dialog_history]

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