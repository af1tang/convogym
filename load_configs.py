#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:21:35 2020

@author: af1tang
"""
from dotenv import load_dotenv
import os, torch, pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_dotenv(verbose=True)
# paths and configs
save_path = os.getenv("save_path")
model_path = os.path.join(save_path, 'checkpoint/model/')
data_path = os.getenv("data_path")
# learning
lr = os.getenv("learn_rate")
gradient_accumulation_steps = os.getenv("gradient_accumulation_steps")
bs = os.getenv("batch_size")
epochs = os.getenv("epochs")
weight_decay = os.getenv("weight_decay")
logging_steps = os.getenv("logging_steps")
save_steps = os.getenv("save_steps")
# convo params
num_personas = os.getenv("num_personas")

def create_dir(directory):
    """create directory if not exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# initialize save folder
create_dir(save_path)

class Configs():
    def __init__(self):
        # saving and loading paths
        self.model_path = model_path
        self.data_path = data_path
        self.save_path = save_path
        self.plot_path = os.path.join(save_path,'samples/')
        self.download_name = 'af1tangt/personaGPT'
        self.i2p_path = os.path.join(save_path, 'i2p')
        self.num_personas = int(num_personas)
        self.num_epochs = int(epochs)
        
opts = Configs()

# global pretrained model and tokenizer
def load_from_pretrained():
    try: 
        print("*"*50)
        print("Load from pretrained")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, 
                                                pad_token='<|endoftext|>', cls_token='<|cls|>',
                                                sep_token='<|sep|>')
        model = GPT2LMHeadModel.from_pretrained(model_path)
        print("*"*50)
    except Exception as e:
        print(e)
        print("*"*50)
        print("Downloading ... ")
        print("*"*50)
        # download dialogpt
        tokenizer = AutoTokenizer.from_pretrained(opts.download_name, 
                                            pad_token='<|endoftext|>', cls_token='<|cls|>',
                                            sep_token='<|sep|>')
        model = AutoModelWithLMHead.from_pretrained(opts.download_name)
        # save to dialogpt
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)

    return model.to(device), tokenizer


model, tokenizer = load_from_pretrained()
p1_tok, p2_tok, start_tok = tokenizer.encode('<|p1|>')[0], tokenizer.encode('<|p2|>')[0], tokenizer.encode('<|start|>')[0]

# action token
act_tok = tokenizer.encode('<|act|>')[0]

# load train and test personas
with open(os.path.join(data_path, 'train_personas'), 'rb') as f:
    train_personas = pickle.load(f)
with open(os.path.join(data_path, 'test_personas'), 'rb') as f:
    test_personas = pickle.load(f)