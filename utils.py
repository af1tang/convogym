#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:17:28 2020

@author: af1tang
"""
import torch, os, pickle, matplotlib.pyplot as plt
import torch.nn as nn, torch.nn.functional as F
import numpy as np, random
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from load_configs import tokenizer, p1_tok, p2_tok, start_tok, opts, device, create_dir

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
        conv = list([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row])#reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    else: conv = list([tokenizer.encode(x) for x in row])
    if make_flat: conv = flatten(conv)
    return conv


def collate(examples):
    '''single instance padding: (prompt, labels, p1, p2)'''
    x= pad_sequence([examples[i] for i in range(len(examples))], batch_first=True, padding_value=tokenizer.pad_token_id)
    lx = [len(examples[i]) for i in range(len(examples))]; mask_x = torch.ones_like(x)
    for k in range(len(examples)):
        mask_x[k, lx[k]:] = 0
    p_x = torch.tensor([list(range(x.shape[1])) for i in range(x.shape[0])])
    for i, p_slice in enumerate(p_x): p_slice[lx[i]-1:] = p_slice[lx[i]-1]
    t_x = torch.full_like(p_x, start_tok)
    return x, mask_x, p_x, t_x

def _get_pos_ids(lengths, masks):
    '''make contiguous positional_ids, varying on mask (padding) size per sample. '''
    pos, max_id, last_id = [], masks.shape[1], []
    for i,mask in enumerate(masks):
        p_ids, count=[],lengths[i]
        for m in mask: 
            if m==1: 
                p_ids.append(count)
                count+=1
        for m in range(p_ids[-1], max_id+lengths[i]-1):
            p_ids.append(p_ids[-1])
        pos.append(torch.tensor(p_ids, dtype=torch.long))
        last_id.append(p_ids[-1]+1)
    pos= torch.stack(pos)
    return pos, last_id

def collate_tuple(examples):
    '''input: array of (prompt, labels, persona1, persona2) tuples.
    output: (x, y, p1, p2, mx, my, mp1, mp2, px, py, pp1, pp2) -- tensors, masks, positional_ids'''
    # padding uneven sequences
    x= pad_sequence([examples[i][0] for i in range(len(examples))], batch_first=True, padding_value=tokenizer.pad_token_id) 
    y= pad_sequence([examples[i][1] for i in range(len(examples))], batch_first=True, padding_value=tokenizer.pad_token_id) 
    p1=  pad_sequence([examples[i][2] for i in range(len(examples))], batch_first=True, padding_value=tokenizer.pad_token_id)
    p2=  pad_sequence([examples[i][3] for i in range(len(examples))], batch_first=True, padding_value=tokenizer.pad_token_id)
    # calculate attention_masks
    lx = [len(examples[i][0]) for i in range(len(examples))]; mask_x = torch.ones_like(x)
    ly = [len(examples[i][1]) for i in range(len(examples))]; mask_y = torch.ones_like(y)
    lp1 = [len(examples[i][2]) for i in range(len(examples))]; mask_p1 = torch.ones_like(p1)
    lp2 = [len(examples[i][3]) for i in range(len(examples))]; mask_p2 = torch.ones_like(p2)
    for k in range(len(examples)):
        mask_x[k, lx[k]:] = 0
        mask_y[k, ly[k]:] = 0
        mask_p1[k, lp1[k]:] = 0
        mask_p2[k, lp2[k]:] = 0
    # fix positional_ids after masking
    p_p1, last_id = _get_pos_ids([0]*x.shape[0], mask_p1)
    p_p2, last_id = _get_pos_ids(last_id, mask_p2)
    p_x, last_id = _get_pos_ids(last_id, mask_x)
    p_y, last_id = _get_pos_ids(last_id, mask_y)

    if opts.use_token_ids:
        #TODO: changed token_ids to [0,1] 
        t_x, t_y, t_p1, t_p2 = torch.full_like(p_x, 1), torch.full_like(p_y, 1), torch.full_like(p_p1, 0), torch.full_like(p_p2, 0)
        return (x ,y, p1,p2, # text
            mask_x, mask_y, mask_p1, mask_p2, # mask ids
            p_x, p_y, p_p1, p_p2, t_x, t_y, t_p1, t_p2 ) # position and token ids
    else:
         return (x ,y, p1,p2, # text
            mask_x, mask_y, mask_p1, mask_p2, # mask ids
            p_x, p_y, p_p1, p_p2)

def filter_turn_indices(x):
    '''depreciated: originally used for preparing persona_dict. '''
    filtered = [[t[1] for t in list(g)] for k,g in groupby(list(enumerate(x)), lambda x: x[1]==tokenizer.eos_token_id) if not k]
    return filtered

## nucleus and top k sampling ##
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0,  top_p: float = 1.0, filter_value: float = -float("Inf"),
                          min_tokens_to_keep: int = 1) -> torch.Tensor:
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


## construct pytorch dataset ##
class ConvDataset(Dataset):
    def __init__(self, convos):
        self.examples = convos
    def __len__(self):
        return len(self.examples)
    def __getitem__(self,item):
        if type(self.examples[item]) == tuple:
            return (torch.tensor(self.examples[item][0], dtype=torch.long), # prompt
                    torch.tensor(self.examples[item][1], dtype=torch.long), # convo
                    torch.tensor(self.examples[item][2], dtype=torch.long), # persona 1
                    torch.tensor(self.examples[item][3], dtype=torch.long) )# persona 2
        else:
            return torch.tensor(self.examples[item],dtype=torch.long)
        
### plotting ###    
def plot_losses(stats, title='loss'):
    create_dir(opts.plot_path)
    x = list(sorted(stats.keys()))
    loss = [stats[i][title] for i in x]
    plt.plot(x, loss, label= title)
    plt.legend()
    plt.title("Training %s" %title)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.plot_path,'training_%s.png'%title))
    plt.close()

def plot_rewards(rewards, training=True, title=None):
    if title is None:
        title = "training" if training else "test"
    plt.plot(rewards, label="episode reward")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(opts.plot_path,'authenticator-exp/%s_rewards.png' %title))
    plt.close()
    
def plot_ir_metrics(prec1, prec5, rec5, rec10, training=True, title=None):
    if title is None:
        title = "training" if training else "test"
    plt.bar(np.arange(4), [prec1, prec5, rec5, rec10] )
    plt.xticks(np.arange(4), ['prec@1', 'prec@5', 'rec@5', 'rec@10'])
    plt.ylim([0., 1.])
    plt.savefig(os.path.join(opts.plot_path,'authenticator-exp/%s_metrics.png' %title))
    plt.close()
    
def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print("p1: "+ msg)
        else:
            print("p2: "+msg)
            print()