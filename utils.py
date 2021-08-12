#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:17:28 2020

@author: af1tang
"""
import torch, os, matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import groupby
from load_configs import tokenizer, opts, create_dir

action_space = [ 'ask about kids.', "ask about pets.", 'talk about work.', 
           'ask about marital status.', 'talk about travel.', 'ask about age and gender.',
    'ask about hobbies.', 'ask about favorite food.', 'talk about movies.', 
    'talk about music.', 'talk about politics.']

## Utils ##
flatten = lambda l: [item for sublist in l for item in sublist]
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def process_conv(row, tokenizer, eos = True, make_flat=True):
    if eos:
        conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row])
    else: conv = list([tokenizer.encode(x) for x in row])
    if make_flat: conv = flatten(conv)
    return conv

def split_by_index(seq, sep):
    result = []
    for el in seq:
        result.append(el)
        if el == sep:
            yield result
            result = []
            
def filter_turn_indices(x):
    filtered = [[t[1] for t in list(g)] for k,g in groupby(list(enumerate(x)), lambda x: x[1]==tokenizer.eos_token_id) if not k]
    return filtered

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()
            
## nucleus and top k sampling ##
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0,  top_p: float = 1.0, filter_value: float = -float("Inf"),
                          min_tokens_to_keep: int = 1) -> torch.Tensor:
    '''
    adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py
    '''
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

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