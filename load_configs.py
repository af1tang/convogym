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
tokenizer_path = os.path.join(save_path, 'checkpoint/tokenizer/')
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
        self.raw_data_path = os.path.join(save_path, 'train_convai_gpt')
        self.val_data_path = os.path.join(save_path, 'valid_convai_gpt')
        self.output_dir = os.path.join(save_path, 'checkpoint/model/')
        self.model_name_or_path = os.path.join(save_path,'checkpoint/model/')
        self.plot_path = os.path.join(save_path,'samples/')
        self.identifier_path = os.path.join(save_path, 'checkpoint/identifier')
        self.authenticator_path = os.path.join(save_path, 'checkpoint/authenticator')
        self.active_learning_path = os.path.join(save_path, 'checkpoint/active_learning')
        self.download_name = 'microsoft/DialoGPT-medium'
        self.i2p_path = os.path.join(save_path, 'i2p')
        # eval
        self.do_eval = True
        self.evaluate_during_training = False
        # batching
        self.batch_size = int(bs)
        self.eval_batch_size = 1
        # optimization
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.eps = float(1e-8)
        self.max_grad_norm = 1.0
        self.num_train_epochs = int(epochs)
        self.max_steps = -1
        self.warmup_steps = 0
        # logging
        self.logging_steps = int(logging_steps)
        self.save_steps = int(save_steps)
        # fp16
        self.use_token_ids = False
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        # sampling params
        self.top_k = 20
        self.top_p = .92
        # identifier configs
        self.zsl = True
        self.k_cands = 100
        self.use_cca = False
        self.use_mtl = False
        self.identifier_mode = "ID"
        # RL params
        self.gamma = .77
        self.ppo_iters = 5
        self.clip = .2
        # ppo fit hyperparams
        self.ppo_bs = 64
        self.ppo_memory_size = 2000
        # meta policy hyperparams
        self.meta_policy_iters = 30
        self.meta_bs = 64
        
opts = Configs()

# global pretrained model and tokenizer
def load_from_pretrained():
    try: 
        print("*"*50)
        print("Load from checkpoint")
        tokenizer = GPT2Tokenizer.from_pretrained(opts.active_learning_path, #opts.model_name_or_path,  
                                                pad_token='<|endoftext|>', cls_token='<|cls|>',
                                                sep_token='<|sep|>')
        model = GPT2LMHeadModel.from_pretrained(opts.active_learning_path)#opts.model_name_or_path)
        try:
            with open(os.path.join(opts.output_dir, 'stats.pkl'), 'rb') as f:
                stats = pickle.load(f)
        except: 
            print("Can't find training stats...")
            stats = None
        print("*"*50)
    except Exception as e:
        print(e)
        try:
            # from dialogpt pretrained
            print("*"*50)
            print("Load from pretrained")
            print("*"*50)
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, 
                                                pad_token='<|endoftext|>', cls_token='<|cls|>',
                                                sep_token='<|sep|>')
            model =  GPT2LMHeadModel.from_pretrained(model_path)
        except:
            print("*"*50)
            print("Downloading ... ")
            print("*"*50)
            # download dialogpt
            tokenizer = AutoTokenizer.from_pretrained(opts.download_name, 
                                                pad_token='<|endoftext|>', cls_token='<|cls|>',
                                                sep_token='<|sep|>')
            model = AutoModelWithLMHead.from_pretrained(opts.download_name)
            # save to dialogpt
            tokenizer.save_pretrained(tokenizer_path)
            model.save_pretrained(model_path)
        stats = None
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|start|>', '<|p1|>', '<|p2|>', '<|act|>']})
    model.resize_token_embeddings(len(tokenizer))
    return model.to(device), tokenizer, stats


model, tokenizer, stats = load_from_pretrained()
p1_tok, p2_tok, start_tok = tokenizer.encode('<|p1|>')[0], tokenizer.encode('<|p2|>')[0], tokenizer.encode('<|start|>')[0]

# new, action token
act_tok = tokenizer.encode('<|act|>')[0]