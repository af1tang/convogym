#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 21:20:40 2021

@author: af1tang
"""
import os
import warnings
import random
import torch, torch.nn as nn
import numpy as np
from rewards import RankingLoss
from utils._device import to_var, to_device, device
from utils._reshape import chunker, flatten
from utils._visualization import plot_losses
from _configs import opts

# embedding models
class StateEstimator(nn.Module):
    """
    Embedding neural network model that converts dialog history strings -> an embedding vector.
    
    The embedding of the dialog history is a "state representation".
    In Markov Decision Process terms, this is the state input to the policy
    The output shape is (1 x n) for a vector in R^n.
    
    Parameters
    -------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT model card.
        
    tokenizer : Huggingface transformer tokenizer (default = GPT2Tokenizer).
        The corresponding tokenizer for the transformer model. 
        The default tokenizer is af1tang/personaGPT.
            
    checkpoint_path : string or os.path, optional
        Save and loading path to the embedder network state_dict (weights).
    
    inp_size : int, optional
        Size of the state space. The default is 1024.
    
    hidden_size : int, optional
        Number of hidden units per layer in the embedder model. The default is 1024.
        
    dropout : float, optional
        Dropout (0-1.0) between hidden layers for the embedder model. The default is 0.2.
        
    Attributes
    -------
    embedder : Callable, nn.Module or nn.Sequential
        Embedding neural network model for dialog history embeddings:
            - takes as input the averaged dialog history tokens 
            - outputs an embedding (feature vector) in R^n.
        
    Notes
    -----  
    The fit() and evaluate() methods train the identifier network on persona data (e.g., PersonaChat). 
    The score() method can potentially handle arbitruary dialog trajectories and personas 
    so long as the persona facts are found in self.p2v. 
    """
    def __init__(self, model, tokenizer, 
                 checkpoint_path=os.path.join(opts.example_path, 'embedder.pt'),
                 inp_size=1024, hidden_size=1024, dropout=.2):
        super(StateEstimator, self).__init__()
        # decoder model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        # network layers
        self.embedder = nn.Sequential( 
                nn.Linear(inp_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            )     
        self.checkpoint_path = checkpoint_path
        self.load_params(self.checkpoint_path)
        
    def forward(self, dialog_history):
        """
        Converts a dialog history of a conversation into a single embedding vector in R^n.

        Parameters
        ----------
        dialog_hx : List of list of ints
            List of token ids, each corresponding to a dialog response.

        Returns
        -------
        x : torch.Tensor
            Outputs an embedding vector for the entire dialog history.

        """
        self.model.eval()   # fixes model to eval mode
        with torch.no_grad():
            inp_embedding = self.model(to_var(flatten(dialog_history)).long(), 
                         output_hidden_states=True)[2][24].squeeze(0).mean(0)
            inp_embedding = inp_embedding.unsqueeze(0)
        x = self.embedder(inp_embedding)
        return x

    def _get_persona_embeddings(self, persona_facts):
        """
        Converts a set of persona facts from string -> embedding vectors in R^n.

        Parameters
        ----------
        persona_facts : List of strings
            List of persona facts in string form.

        Returns
        -------
        vectors : torch.Tensor
            Outputs an embedding vector for each persona fact.

        """
        self.model.eval()   # fixes model to eval mode
        vectors = []
        with torch.no_grad():
            for line in persona_facts:     
                outp = self.model(**self.tokenizer(line, return_tensors='pt').to(device), output_hidden_states=True)
                vectors.append(outp[2][24].squeeze(0).mean(0))
            vectors = torch.stack(vectors)
        return vectors
    
    def load_params(self):
        """
        Try to load the parameters of the identifier model using load_path. Raises an exception if fails.
        """
        try: 
            state_dict = torch.load(self.checkpoint_path)
            self.embedder.load_state_dict(state_dict)
        except Exception as e:
            warnings.warn(str(e) + '''
                          
                          This state estimator model is NOT trained yet.
                          
                          ''')

    def fit(self, *args, **kwargs):
        """
        Not Implemented in the base class.
        """
        raise NotImplementedError
        
    def evaluate(self, *args, **kwargs):
        """
        Not Implemented in the base class.
        """
        raise NotImplementedError

class RankingStateEstimator(StateEstimator):
    """
    A wrapper object over the base StateEstimator class that trains the base estimator using the RankingLoss object.
    
    Parameters
    -------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT model card.
        
    tokenizer : Huggingface transformer tokenizer (default = GPT2Tokenizer).
        The corresponding tokenizer for the transformer model. 
        The default tokenizer is af1tang/personaGPT.
            
    checkpoint_path : string or os.path, optional
        Save and loading path to the embedder network state_dict (weights).
    
    inp_size : int, optional
        Size of the state space. The default is 1024.
    
    hidden_size : int, optional
        Number of hidden units per layer in the embedder model. The default is 1024.
        
    dropout : float, optional
        Dropout (0-1.0) between hidden layers for the embedder model. The default is 0.2.
        
    k : int, optional
        Number of candidates to use for triplet loss. The default is 20.
        
    Attributes
    -------
    embedder : Callable, nn.Module or nn.Sequential
        Embedding neural network model for dialog history embeddings:
            - takes as input the averaged dialog history tokens 
            - outputs an embedding (feature vector) in R^n.
            
    criterion : RankingLoss
        Loss function used to train the state estimator network.
        
    """
    def __init__(self, model, tokenizer, 
                 checkpoint_path=os.path.join(opts.example_path, 'embedder.pt'),
                 inp_size=1024, hidden_size=1024, dropout=.2, k=20):
        super().__init__(model=model, tokenizer=tokenizer, 
                         checkpoint_path=checkpoint_path,
                         inp_size=inp_size, hidden_size=hidden_size,
                         dropout=dropout)
        self.criterion = RankingLoss(self, k=k)
        
    def fit(self, train_data, candidates,
                epochs=1, lr=1e-3, bs=32, 
                logging_steps=10, save_steps=50):
        """
        Trains the identifier model on a training set of (dialog token embeddings, persona facts) pairs.

        Parameters
        ----------
        train_data : List of tuples
            Batches of training samples of the form (dialog_history, persona 1, persona 2):
                - dialog_history: List of list of ints representing token ids for responses in conversation.
                - persona 1: list of strings corresponding to person 1 persona facts
                - persona 2: list of strings corresponding to person 2 persona facts.
                
        candidates : List of strings
            List of possible persona facts from which to do negative sampling.
            
        save_path : String or os.path
            Path to save identifier paramters to.
            
        epochs : int, optional
            Number of training epochs. The default is 1.
            
        lr : float, optional
            Learning rate. The default is 1e-3.
            
        bs : int, optional
            Batch size. The default is 32.
            
        logging_steps : int, optional
            Print training results every logging_steps. The default is 10.
            
        save_steps : int, optional
            Save model checkpoint every save_steps. The default is 50.

        Returns
        -------
        None.

        """
        print()
        print("Training identifier ... ")
        print()
        optimizer = torch.optim.Adam(self.embedder.parameters(), lr=lr)
        # training
        stats, iters, tr_loss, logging_loss = {}, 0, 0.0, 0.0
        self.embedder.train()
        self.embedder.zero_grad()
        iters = 0
        for epoch in range(epochs):
            random.shuffle(train_data)
            for minibatch in chunker(train_data, bs):
                # batching
                convos, p1, p2 = zip(*minibatch)
                convos, p1, p2 = map(to_var, (convos, p1, p2))
                # forward
                x1 = self.forward([history[::2] for history in convos])
                x2 = self.forward([history[1::2] for history in convos])
                loss1 = self.criterion(inp=x1, positives=p1, candidates=candidates)
                loss2= self.criterion(inp=x2, positives=p2, candidates=candidates)
                loss = loss1+loss2
                # backward
                loss.backward()
                tr_loss += loss.item()
                #if (step+1)% opts.gradient_accumulation_steps == 0:
                optimizer.step()
                self.embedder.zero_grad()
                iters +=1
                # reporting
                if iters % logging_steps ==0:
                    stats[iters] = {'loss': (tr_loss - logging_loss) / logging_steps}
                    logging_loss = tr_loss
                    print('Epoch: %d | Iter: %d | loss: %.3f ' %( 
                    epoch, iters, stats[iters]['loss']) )
                    
                if iters % save_steps==0:
                    print("Saving stuff ... ")
                    state_dict = self.embedder.state_dict()
                    torch.save(state_dict, self.checkpoint_path)
                    plot_losses(stats, title='loss' )
                    print("Done.")
                    
    def evaluate(self, test_data, candidates, bs=32):
        """
        Evaluates identifier model on test persona data.

        Parameters
        ----------
        test_data : List of tuples
            Batches of training samples of the form (dialog_history, persona 1, persona 2):
                - dialog_history: List of list of ints representing token ids for responses in conversation.
                - persona 1: list of strings corresponding to person 1 persona facts
                - persona 2: list of strings corresponding to person 2 persona facts.
                
        bs : int, optional
            Batch size. The default is 32.
        k : int, optional
            Number of candidates to use for triplet loss. The default is 20.

        Returns
        -------
        eval_stats : dict
            Dictionary of precision@k and recall@k stats.

        """
        # eval
        print("-"*50)
        print("Evaluating identifier ... ")
        eval_stats = {'prec@1':[], 'prec@5':[], 'rec@5':[], 'rec@10':[]}
        self.embedder.eval()
        for minibatch in chunker(test_data, bs):
            # batching
            convos, p1, p2 = zip(*minibatch)
            convos, p1, p2 = map(to_var, (convos, p1, p2))
            with torch.no_grad():
                x1 = self.forward([history[::2] for history in convos])
                x2 = self.forward([history[1::2] for history in convos])
            for xx,yy in [(x1,p1), (x2,p2)]:
                prec1, prec5, rec5, rec10 = self.criterion.score(inp=xx, positive=yy, candidates=candidates)
                eval_stats['prec@1'].extend(prec1)
                eval_stats['prec@5'].extend(prec5)
                eval_stats['rec@5'].extend(rec5)
                eval_stats['rec@10'].extend(rec10)
        print("prec1: %.1f | prec5: %.1f | rec5: %.1f | rec10: %.1f" % ( 
                                100*np.mean(eval_stats['prec@1']), 100*np.mean(eval_stats['prec@5']), 
                                100*np.mean(eval_stats['rec@5']), 100*np.mean(eval_stats['rec@10'])) )
        print("-"*50)
        return eval_stats