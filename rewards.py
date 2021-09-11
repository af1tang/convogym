#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:02:24 2021

@author: af1tang
"""
import random
import numpy as np
import torch, torch.nn as nn
from _personas import persona_facts

class RankingLoss(nn.Module):
    """
    Implements triplet loss with negative sampling.
    
    Parameters
    -------
    state_estimator : StateEstimator object or Callable
        A state estimator model that maps from dialog history text -> embedding vector to represent state information.

    swaps : bool, optional
        Whether to swap anchor and positive sample roles during triplet loss.
        
    margin : float, optional
        Margin parameter used for triplet loss.
    
    k : int, optional
        Number of total candidates to use during negative sampling. The default is 20.
        
    reduction : string, optional
        See https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html for details.
        > Specifies the reduction to apply to the output:
        >    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        >    ``'mean'``: the sum of the output will be divided by the number of
        >    elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        >    and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        >    specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """
    def __init__(self, state_estimator, swap=False, margin=1.0, k = 20, reduction='sum'):
        super(RankingLoss, self).__init__()
        self.ranking_loss = nn.TripletMarginLoss(margin=margin, swap=swap, reduction=reduction)
        self.state_estimator = state_estimator
        self.k = k
                
    def forward(self, inp, positives, candidates):
        '''inp: (bs x dim) e.g., 1 x 1024
        positives: (# personas x dim) e.g., 5 x 1024
        candidate_facts: List of strings'''
        # generate candidates using negative sampling
        positive_samples, negative_samples = self._generate_candidates(candidates, positives)
        # calculate ranking loss over candidates
        num_pos, num_neg = positive_samples.size(0), negative_samples.size(0)
        total_loss = 0.
        count = 0
        for i in range(0, num_neg, num_pos):
            if positive_samples.size(0) == negative_samples[i:i+num_pos].size(0):
                total_loss = total_loss + self.ranking_loss(inp, positive_samples, negative_samples[i:i+num_pos])
                count +=1
        return total_loss
    
    def score(self, inp, positives, candidates):
        """
        Calculates the dialog history embeddings and outputs the triplet loss on batch. 

        Parameters
        ----------
        inp : torch.Tensor
            Embedding of an input dialog trajectory. 
            
        positives : List of list strings
            List of person 2 persona facts corresponding to each conversation in the batch.    
            
        candidates : List of strings
            List of supporting set of persona facts to generate candidates from.

        Returns
        -------
        prec1 : float
            # hits with the top guess / # samples in the batch. Measures quality of the top ranked guess.
            
        prec5 : float
            Average precsion@5 score, defined as # hits in top 5 guesses / 5. 
            
        rec5 : float
            Average recall@5 score, defined as # hits in top 5 guesses / # positives. Measures diversity of guesses.
            
        rec10 : float
            Average recall@5 score, defined as # hits in top 10 guesses / # positives. Measures diversity of guesses.

        Examples
        -------
        >>> from convogym.representers import StateEstimator
        >>> from convogym._decoder import model
        >>> dialog_history = ["hi how are you doing?",  # corresponds to person 2
                              "good, just got back from playing baseball.",
                              "i like it. i also like to play the piano."]
        >>> persona = ["i play the piano.", "i'm a little league all-star."]
        >>> state_estimator = StateEstimator(model)
        >>> ranking_loss = RankingLoss()
        >>> inp = state_estimator(dialog_history) # outputs a (1 x 1024) tensor
        >>> prec1, prec5, rec5, rec10 = ranking_loss.score(inp, persona) 
        >>> print("precision@5: %.2f, recall@5: %.2f" %(prec5, rec5))
        precision@5: 0.40, recall@5: 1.00
        
        """
        pos_targets, neg_targets = self._generate_candidates(candidates, positives)
        # calculate precision@k and recall@k metrics
        prec1 = [self._score_triplet(inp[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=1)
                 for i in range(len(inp))]
        prec5 = [self._score_triplet(inp[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=5)
                 for i in range(len(inp))]
        rec5 = [self._score_triplet(inp[i].unsqueeze(0), pos_targets[i], 
                neg_targets[i], at_k=5, use_recall=True) for i in range(len(inp))]
        rec10 = [self._score_triplet(inp[i].unsqueeze(0), pos_targets[i], 
                neg_targets[i], at_k=10, use_recall=True) for i in range(len(inp))]

        return prec1, prec5, rec5, rec10
    
        
    def _generate_candidates(self, candidates, positives):
        """
        Generates negative samples for triplet rankings.
        """
        negatives = random.sample( set(candidates) - set(positives), self.k)
        pos_samples = self.state_estimator._get_persona_embeddings(positives)
        neg_samples = self.state_estimator._get_persona_embeddings(negatives)
        return pos_samples, neg_samples
    
    def _score_triplet(self, x, pos,neg, at_k, use_recall=False):
        """
        Scores the ranking model based on the ranked list of generated candidates. 
        
        If use_recall then recall@k is used. Otherwise precision@k is used.
        """
        dist = nn.CosineSimilarity()
        scores = []
        for k,v in enumerate(pos):
            scores.append((k,1, dist(x,v.unsqueeze(0)).item()))
        for k,v in enumerate(neg):
            scores.append((k+len(pos), 0, dist(x,v.unsqueeze(0)).item()))
        scores = sorted(scores, key = lambda x: x[-1], reverse=True)
        picks = [s[1] for s in scores[0:at_k]]
        # precision or recall @ k
        score = sum(picks) / len(pos) if use_recall else sum(picks) / len(picks)
        return score

# interface with gym callback objects
class Reward:
    """
    A helper object for callbacks that outputs turn-level rewards based on state information at each turn.
    
    Also tracks various performance metrics during evaluation of a dialog policy.
    
    Parameters
    -------
    state_estimator : StateEstimator object or Callable
        A state estimator model that maps from dialog history text -> embedding vector to represent state information.
    
    criterion : Callable
        Maps from state vector to a scalar reward. If None, initiates a RankingLoss object. The default is None.
    
    candidates : List of strings
        List of possible persona facts to sample candidates from. Only used if RankingLoss is used for criterion.
    """
    def __init__(self, state_estimator, criterion=None, candidates=persona_facts):
        if not criterion:
            self.criterion = RankingLoss(state_estimator)
        self.candidates = candidates
        self.metrics = {'prec1s': [], 'prec5s': [],
                        'rec5s': [], 'rec10s': [], 'rewards': []}
    
    def calculate_reward(self, scb, mcb):
        """
        Obtains reward from given persona and state information and passes it to the state callback.
    
        Parameters
        ----------
        identifier : ID object (nn.Module or Callable)
            Wrapper object for the Identifier neural network model:
            - takes as input the averaged dialog history tokens 
            - outputs an embedding (feature vector) in R^n.
            Used to score trajectories based on persona inputs.
            
        state : torch.Tensor
            The feature vector embedding of the dialog history. Shape should be (1 x n) for a vector in R^n.
            In Markov Decision Process terms, this is the state input to the policy. The default is None.
            
        context : torch.Tensor
            An auxiliary feature vector embedding of the dialog history. Shape should be (1 x n) for a vector in R^n.
            This parameter is only relevant if a contextual policy is used. The default is None.
            
        personas : List of list strings
            List of person 2 persona facts corresponding to each conversation in the batch.
            
        k : int, optional
                Number of total candidates to use during negative sampling. The default is 20.
    
        Attributes
        -------
        scb.reward : float
            A scalar reward for current turn.
            
        scb.rewards: List of floats
            List of reward for each turn.
    
        """
        loss = self.criterion(inp=scb.state.view(1,-1), 
                              positives=scb.personas, 
                              candidates=self.candidates)
        scb.reward = - loss.item()
        scb.rewards.append(scb.reward)
        return scb, mcb
    
    def score_trajectory(self, scb, mcb):
        """
        Not implemented in base class.
        """
        raise NotImplementedError

class IR_Reward(Reward):
    """
    Implements a wrapper on the Reward base object. Calculates various precision@k and recall@k metrics based on the RankingLoss criterion.
    """
    def __init__(self, state_estimator, criterion=None, candidates=persona_facts):
        super().__init__(state_estimator, None, candidates)

    def score_trajectory(self, scb, mcb):
        """
        Calculates precision@k and recall@k for current dialog trajectory.
    
        Parameters
        ----------            
        scb : StateCb object
            State callback object.
    
        Metrics
        -------
        prec1 : float
            # hits with the top guess / # samples in the batch. Measures quality of the top ranked guess.
            
        prec5 : float
            Average precsion@5 score, defined as # hits in top 5 guesses / 5. 
            
        rec5 : float
            Average recall@5 score, defined as # hits in top 5 guesses / # positives. Measures diversity of guesses.
            
        rec10 : float
            Average recall@5 score, defined as # hits in top 10 guesses / # positives. Measures diversity of guesses.
    
        """
        prec1, prec5, rec5, rec10 = self.criterion.score(inp=scb.state.view(1,-1), 
                                                        positives=scb.personas, 
                                                        candidates=self.candidates)
        # log data
        self.metrics['prec1s'].extend(prec1)
        self.metrics['prec5s'].extend(prec5)
        self.metrics['rec5s'].extend(rec5)
        self.metrics['rec10s'].extend(rec10)
        self.metrics['rewards'].append(scb.rewards)
        print("prec@1: %.2f | prec@5: %.2f | rec@5: %.2f | rec@10: %.2f" % (prec1[0], prec5[0], rec5[0], rec10[0]))
        print("avg p@1: %.2f | avg p@5: %.2f | avg r@5: %.2f | avg. r@10: %.2f" % (np.mean(self.metrics['prec1s']),
                                                                                   np.mean(self.metrics['prec5s']), 
                                                                                    np.mean(self.metrics['rec5s']), 
                                                                                    np.mean(self.metrics['rec10s'])))
        print()
        
