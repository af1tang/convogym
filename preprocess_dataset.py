#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:40:29 2020

@author: af1tang
"""
from tqdm import tqdm
import torch, os, pickle, nltk
import torch.nn as nn, torch.nn.functional as F
import numpy as np, random

from load_configs import model, tokenizer, opts, device, data_path, save_path
from utils import *

## Persona Preprocess ##
def preprocess_convai(filename):
    raw_data = open(filename).read().strip().split('\n')
    data, count = {}, 0
    curr_convo, curr_ps, curr_pt = [], [], []
    indices = []
    
    person_a = 'your persona'
    person_b = "partner's persona"
    with tqdm(total = len(raw_data)) as pbar:
        turn_count, ctx_count = 1,0 #init cycle
        for idx, line in enumerate(raw_data):
            if person_a in line[0:20]:
                if (turn_count != 0) and (len(curr_ps)>1 and len(curr_pt)>1 and len(curr_convo)>1):
                    if idx > 1:
                        if curr_convo[0] == '__SILENCE__' :
                            p1 = curr_ps; p2 = curr_pt; curr_convo = curr_convo[1:]
                        else:
                            p1 = curr_pt; p2 = curr_ps
                        data[count] = { 'inp': process_conv([curr_convo[0]], tokenizer),
                                        'labels': process_conv(curr_convo[1:],tokenizer), #to_data(torch.cat(curr_convo,dim=-1)[0]), 
                                       'p_src': process_conv(p1, tokenizer,make_flat=False), #to_data(torch.cat(curr_ps,dim=-1)[0]),
                                       'p_trg': process_conv(p2, tokenizer, make_flat=False)}#to_data(torch.cat(curr_pt,dim=-1)[0])}
                        count+=1
                    curr_convo, curr_ps, curr_pt = [], [], []
                    turn_count=0

                words = line.split()
                turn_id, words = int(words[0]), ' '.join(words[3:])
                curr_ps.append(words)

                ctx_count +=1
                assert ctx_count == turn_id
                
            elif person_b in line[0:20]:
                if (turn_count != 0) and (len(curr_ps)>1 and len(curr_pt)>1 and len(curr_convo)>1):
                    if idx > 1:
                        if curr_convo[0] == '__SILENCE__' :
                            p1 = curr_ps; p2 = curr_pt; curr_convo = curr_convo[1:]
                        else:
                            p1 = curr_pt; p2 = curr_ps
                        data[count] = { 'inp': process_conv([curr_convo[0]], tokenizer),
                                        'labels': process_conv(curr_convo[1:],tokenizer), #to_data(torch.cat(curr_convo,dim=-1)[0]), 
                                       'p_src': process_conv(p1, tokenizer,make_flat=False), #to_data(torch.cat(curr_ps,dim=-1)[0]),
                                       'p_trg': process_conv(p2, tokenizer, make_flat=False)}#to_data(torch.cat(curr_pt,dim=-1)[0])}
                        count+=1
                    curr_convo, curr_ps, curr_pt = [], [], []
                    turn_count=0
                words = line.split()
                turn_id, words = int(words[0]), ' '.join(words[3:])
                curr_pt.append(words)

                ctx_count +=1
                assert ctx_count == turn_id

                
            else:
                if ctx_count !=0:
                    turn_count = ctx_count *1 
                    ctx_count =0
                    indices.append(idx)
                        
                src_line, trg_line = line.split('\t')
                src_words = src_line.split()
                src_idx, src_line = src_words[0], ' '.join(src_words[1:])

                curr_convo.append(src_line) 
                curr_convo.append(trg_line)#turn)
                
                turn_count +=1
                assert turn_count == int(src_idx)
                
            pbar.update(1)
        
    return data

def _filter_persona_list(personas, total=1255):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from word2number import w2n
    import re
    # from gensim import downloader as api
    # w2v = api.load('fasttext-wiki-news-subwords-300')
    
    def one_hot(arr, size):
        onehot = np.zeros((len(arr),size), dtype = int)
        for i in range(len(arr)):
            if not np.isnan(arr[i]):            
                onehot[i, int(arr[i])]=1
        #onehot[np.arange(len(arr)), arr] =1
        return onehot

    def num_filter(x):
        filter_func = lambda x: str(w2n.word_to_num(x[0])) if x[1] == 'CD' else x[0]
        r = re.compile("([0-9]+)([a-zA-Z]+)")
        try:
            return filter_func(x)
        except:
            if x[0] == 'live':
                return x[0]
            else:
                return filter_func((r.match(x[0]).group(1), 'CD'))
        
    #bow_personas = np.array([[pp for pp in tokenizer.encode(p) if pp != tokenizer.eos_token_id] 
    #                         for p in personas])
    final_lst = {}
    cc = SmoothingFunction()

    for i, text in enumerate(personas):
        text = text.strip(tokenizer.eos_token)
        word_tags = nltk.pos_tag(nltk.word_tokenize(text))
        text = ' '.join([num_filter( word_tags[j] )
                             for j in range(len(word_tags))])
        if text not in final_lst.keys():
            best_bleu, best_match = 0., None
            for key, val in final_lst.items():
                curr_bleu = sentence_bleu([text], key, weights=(.25,.25, .25, .25), 
                                              smoothing_function = cc.method2)
                if curr_bleu > best_bleu:#len(set(val) - set(p)) / min(len(val), len(p)) < .1:
                    best_bleu = curr_bleu 
                    best_match = key
            
            #best_cos = w2v.wv.similarity(text, best_match)
            if (best_bleu > .75):# and (best_cos>.5):
                num_text = [xx for xx in word_tags if xx[1] == 'CD' ]
                if (len(num_text) >0): 
                    num_key = [yy for yy in nltk.pos_tag(nltk.word_tokenize(best_match)) if yy[1] == 'CD']
                    if (len(num_key) > 0) and (int(num_filter(num_text[0])) - int(num_filter(num_key[0])) == 0):
                        del final_lst[best_match]
                        print("-"*50)
                        print(best_bleu, text, best_match)
                        print(i, len(final_lst))
                        print("-"*50)
                else:
                    del final_lst[best_match]
                    print("-"*50)
                    print(best_bleu, text, best_match)
                    print(i, len(final_lst))
                    print("-"*50)
                
            final_lst[text] = best_match
            if i %100 == 0: print(i)
    return final_lst

## persona processing ##
def _initialize_clf_tokenizer(mode):
    import spacy
    from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2Model 
    if mode == 'BOW1': 
        tokenizer = spacy.load('en_core_web_lg')
        transformer = None

    elif mode == 'BOW2':
        tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        transformer = BertModel.from_pretrained('bert-base-uncased')

    elif mode == 'LSTM':
        tokenizer  = spacy.load('en_core_web_lg')
        transformer = None

    elif mode == 'BERT':
        tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
        transformer = BertModel.from_pretrained('bert-base-uncased')

    elif mode == 'GPT':
        tokenizer  = GPT2Tokenizer.from_pretrained('gpt2-medium')
        transformer =  GPT2Model.from_pretrained('gpt2-medium')

    elif mode == 'ID':
        tokenizer = GPT2Tokenizer.from_pretrained(opts.model_name_or_path, 
                                                pad_token='<|endoftext|>', cls_token='<|cls|>',
                                                sep_token='<|sep|>')
        transformer = GPT2Model.from_pretrained(opts.model_name_or_path)

    if transformer is not None: transformer.to(device)
    return tokenizer, transformer

def _process_persona_dict_texts(texts, tokenizer, transformer=None, mode="BOW1"):
    if mode == 'BOW1': 
        x1 = np.array([np.array([tokenizer(line).vector for line in text[::2]]).mean(0) for text in texts ])
        x2 = np.array([np.array([tokenizer(line).vector for line in text[1::2]]).mean(0) for text in texts ])
        x1, x2 = map(to_var, (x1,x2))
    elif mode == 'LSTM': 
        x1 = pad_sequence([pad_sequence(to_var([tokenizer(line).vector for line in text[::2]]), batch_first=True) for text in texts ], batch_first=True)
        x2 = pad_sequence([pad_sequence(to_var([tokenizer(line).vector for line in text[1::2]]), batch_first=True) for text in texts ], batch_first=True)
    else:
        inp1 = [tokenizer(' '.join(text[::2]), return_tensors='pt').to(device) for text in texts]
        inp2 = [tokenizer(' '.join(text[1::2]), return_tensors='pt').to(device) for text in texts]
        with torch.no_grad():
            outp1 = [transformer(**inputs, output_hidden_states=True) for inputs in inp1 ]
            outp2 = [transformer(**inputs, output_hidden_states=True) for inputs in inp2 ]
        if mode == 'BOW2':
            x1 = torch.stack([ xi[2][0].squeeze(0).mean(0) for xi in outp1])
            x2 = torch.stack([ xi[2][0].squeeze(0).mean(0) for xi in outp2])    
        else:
            x1 = torch.stack([ xi[0].squeeze(0).mean(0) for xi in outp1])
            x2 = torch.stack([ xi[0].squeeze(0).mean(0) for xi in outp2])
    return x1, x2

def _transform_i2p(i2p, tokenizer, transformer):
    inputs = dict([(k, tokenizer(v, return_tensors='pt').to(device)) for k,v in i2p.items()])
    with torch.no_grad():
        outputs = dict([(k, transformer(**v)[0].squeeze(0).mean(0) ) for k,v in inputs.items()])
    return outputs

def _build_persona_id_db(persona_dict, batch_size, opts):
    dataset, count={'tr':{}, 'te':{}}, 0
    tokenizer, transformer = _initialize_clf_tokenizer(opts.identifier_mode)
    for split in persona_dict.keys():
        data = persona_dict[split]
        for i, minibatch in enumerate(chunker(sorted(data.keys()), batch_size)):
            texts = [data[k]['x'] for k in minibatch]
            x1, x2 = _process_persona_dict_texts(texts, tokenizer=tokenizer,
                                                 transformer=transformer, mode=opts.identifier_mode)
            y1 = [data[k]['p1'] for k in minibatch]
            y2 = [data[k]['p2'] for k in minibatch]
            for k in range(len(minibatch)):
                dataset[split][minibatch[k]] = {'x1': x1[k], 'x2': x2[k],'y1': y1[k], 'y2':y2[k]}
                count+=1
                if count%100==0: print(opts.identifier_mode, str(count))
    torch.save(dataset,os.path.join(opts.identifier_path, 'persona_dict_%s'%opts.identifier_mode))
    return dataset
        
def prepare_persona_dict():
    from sklearn.cluster import KMeans
    #from transformers import BertModel, BertTokenizer

    with open(opts.raw_data_path, 'rb') as f: tr_data = pickle.load(f)
    with open(opts.val_data_path, 'rb') as f: val_data = pickle.load(f)
    persona_dict = {'tr':{}, 'te':{}}
    personas = []
    # 1x build dictionary of personas
    for key, data in (('tr', tr_data), ('te', val_data)):
        for k in data.keys():
            personas.extend([tokenizer.decode(np.array(line)) for line in data[k]['p_src']])
            personas.extend([tokenizer.decode(np.array(line)) for line in data[k]['p_trg']])
    personas = sorted(list(set(personas)))

    #bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #bert_model = BertModel.from_pretrained('bert-base-uncased')
    print("Making persona dictionary ... ")
    vectors = []
    with torch.no_grad():
        for line in personas:     
            outp = model(**tokenizer(line, return_tensors='pt').to(device), output_hidden_states=True)
            #outp = bert_model(**bert_tokenizer(line, return_tensors='pt'), output_hidden_states=True)
            vectors.append(outp[2][24].squeeze(0).mean(0))
            #vectors.append(outp[1].squeeze(0))
        vectors = torch.stack(vectors)
    # kmeans 
# =============================================================================
#     print("Kmeans clustering ... ")
#     kmeans = KMeans(n_clusters = 1155)
#     kmeans.fit(to_data(vectors))
#     cluster_ids = kmeans.predict(to_data(vectors))
# =============================================================================
    
    # making dictionaries
    #i2p = dict([(cluster_ids[k], v) for k,v in enumerate( personas )]) 
    i2p = dict([(k,v) for k,v in enumerate( personas )])
    #i2v = dict([(cluster_ids[k], v) for k,v in enumerate( vectors )]) 
    i2v = dict([(k,v) for k,v in enumerate( vectors )])
    #p2i = dict([(v, cluster_ids[k]) for k,v in enumerate( personas )])
    p2i = dict([(v,k) for k,v in enumerate( personas )])
    # 2x map personas -> ids
    for key, data in (('tr', tr_data), ('te', val_data)):
        for k in data.keys():
            query1 = [tokenizer.decode(np.array(line)) for line in data[k]['p_src']]
            query2 = [tokenizer.decode(np.array(line)) for line in data[k]['p_trg']]
            p1 = [p2i[q] for q in query1]
            p2 = [p2i[q] for q in query2]
            x = tokenizer.decode(data[k]['inp'] + data[k]['labels']).split(tokenizer.eos_token)[:-1]
            persona_dict[key][k] = {'x': x, 'p1': p1, 'p2': p2}
            for j in range(min(len(p1), len(p2))):
                p1j = tokenizer.encode(i2p[p1[j]])
                #p1j = bert_tokenizer.encode(i2p[p1[j]])
                p2j = tokenizer.encode(i2p[p2[j]])
                #p2j = bert_tokenizer.encode(i2p[p2[j]])
                if len(p1j) != len(data[k]['p_src'][j]):
                    print(tokenizer.decode(p1j), tokenizer.decode(data[k]['p_src'][j]))
                if len(p2j) != len(data[k]['p_trg'][j]):
                    print(tokenizer.decode(p2j) , tokenizer.decode(data[k]['p_trg'][j]))
    return persona_dict, i2v, i2p #, kmeans
    
    
if __name__ == '__main__':        
    print("Preprocessing dataset ... ")
    train_data = preprocess_convai(os.path.join(data_path, 'train_both_original_no_cands.txt'))
    val_data = preprocess_convai(os.path.join(data_path, 'valid_both_original_no_cands.txt'))
    with open(opts.raw_data_path, 'wb') as f: pickle.dump(train_data, f)
    with open(opts.val_data_path, 'wb') as f: pickle.dump(val_data, f)
    #persona_dict, i2v, i2p, kmeans = prepare_persona_dict()
    persona_dict, i2v, i2p = prepare_persona_dict()
    print("Making persona dictionary ...")
    create_dir(opts.identifier_path)
    with open(os.path.join(opts.identifier_path, 'i2p'), 'wb') as f: pickle.dump(i2p, f)
    with open(os.path.join(opts.identifier_path, 'persona_dict'), 'wb') as f: pickle.dump(persona_dict, f)
    #with open(os.path.join(opts.identifier_path, 'kmeans'), 'wb') as f: pickle.dump(kmeans,f)
    torch.save(i2v, os.path.join(opts.identifier_path, 'i2v'))
    print(" ... done.")

    