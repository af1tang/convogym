#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:02:24 2021

@author: af1tang
"""
import random
import numpy as np
import torch, torch.nn as nn
from convogym.utils._visualization import display_dialog_history
from convogym.prefixes import load_persona_facts

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
    
    def _calculate_loss(self, inp, pos_cands, neg_cands):
        """
        Calculates the triplet loss for a single sample.

        Parameters
        ----------
        inp : torch.Tensor
            A single state embedding.
            
        pos_cands : torch.Tensor
            The set of persona embeddings that co-occur with the dialog.
            
        neg_cands : torch.Tensor
            A set of persona embeddings negatively sampled from a list of candidates.

        Returns
        -------
        total_loss : torch.Tensor
            Loss on single sample over candidate comparisons.

        """
        num_pos, num_neg = pos_cands.size(0), neg_cands.size(0)
        total_loss = 0.
        count = 0
        for i in range(0, num_neg, num_pos):
            if pos_cands.size(0) == neg_cands[i:i+num_pos].size(0):
                total_loss = total_loss + self.ranking_loss(inp, pos_cands, neg_cands[i:i+num_pos])
                count +=1
        return total_loss
        
    def forward(self, states, positives, candidates):
        """
        Calculates the dialog history embeddings and outputs the triplet loss on batch. 
        
        Parameters
        ----------
        states : torch.Tensor
            Embeddings of input dialog trajectories. 
            
        positives : List of list strings
            List of person 2 persona facts corresponding to each conversation in the batch.    
            
        candidates : List of strings
            List of supporting set of persona facts to generate candidates from.
            
        Returns
        -------
        loss : float
            Triplet loss on batch.

        """
        # generate candidates using negative sampling
        pos_targets, neg_targets = self._generate_candidates(candidates, positives)
        # calculate ranking loss over candidates
        loss = sum([self._calculate_loss(states[i].unsqueeze(0), pos_targets[i], neg_targets[i])
                                for i in range(len(states))])
        return loss
    
    def score(self, states, positives, candidates):
        """
        Calculates precision and recall metrics based on embeddings of dialog trajectories.
        
        Parameters
        ----------
        states : torch.Tensor
            Embeddings of input dialog trajectories. 
            
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
        >>> states = state_estimator(dialog_history) # outputs a (1 x 1024) tensor
        >>> prec1, prec5, rec5, rec10 = ranking_loss.score(states, [persona]) 
        >>> print("precision@5: %.2f, recall@5: %.2f" %(prec5, rec5))
        precision@5: 0.40, recall@5: 1.00
        
        """
        pos_targets, neg_targets = self._generate_candidates(candidates, positives)
        # calculate precision@k and recall@k metrics
        prec1 = [self._score_triplet(states[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=1)
                 for i in range(len(states))]
        prec5 = [self._score_triplet(states[i].unsqueeze(0), pos_targets[i], neg_targets[i], at_k=5)
                 for i in range(len(states))]
        rec5 = [self._score_triplet(states[i].unsqueeze(0), pos_targets[i], 
                neg_targets[i], at_k=5, use_recall=True) for i in range(len(states))]
        rec10 = [self._score_triplet(states[i].unsqueeze(0), pos_targets[i], 
                neg_targets[i], at_k=10, use_recall=True) for i in range(len(states))]

        return prec1, prec5, rec5, rec10
    
        
    def _generate_candidates(self, candidates, positives):
        """
        Generates negative samples for triplet rankings.
        """
        negatives = [random.sample( set(candidates) - set(pos_set), self.k) for pos_set in positives]
        pos_samples = [self.state_estimator._get_persona_embeddings(p) for p in positives]
        neg_samples = [self.state_estimator._get_persona_embeddings(n) for n in negatives]
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
    
    In the base Reward, a custom criterion function is 
    
    Parameters
    -------
    state_estimator : StateEstimator object or Callable
        A state estimator model that maps from dialog history text -> embedding vector to represent state information.
    
    criterion : nn.Module or Callable
        Maps from state vector to a scalar reward. The default is None.
    
        If want (state, action) -> reward, simply concatenate action to state vector as input.

    """
    def __init__(self, state_estimator, criterion):
        self.state_estimator = state_estimator
        self.criterion = criterion
    
    def calculate_reward(self, scb, mcb):
        """
        Obtains reward from given persona and state information and passes it to the state callback.
    
        Attributes
        -------
        scb.reward : float
            A scalar reward for current turn.
            
        scb.rewards: List of floats
            List of reward for each turn.
    
        """
        loss = self.criterion(scb.state.view(1,-1))
        scb.reward = - loss.item()
        scb.rewards.append(scb.reward)
        return scb, mcb
    
    def score_trajectory(self, scb, mcb):
        """
        Not implemented in base class.
        """
        raise NotImplementedError
        
class ManualReward(Reward):
    """
    Implements a wrapper on the Reward base object. 
    
    At each turn, the user is manually prompted to score the dialog up to the current turn. 
    (state, action, next_state, reward) samples collected this way can be used to learn reward function that approximates user scoring (i.e., a criterion). 
    """
    def __init__(self, state_estimator, *args, **kwargs):
        super().__init__(state_estimator=state_estimator, criterion=None)

    def calculate_reward(self, scb, mcb):
        """
        Obtains reward from given persona and state information and passes it to the state callback.
    
        Attributes
        -------
        scb.reward : float
            A scalar reward for current turn.
            
        scb.rewards: List of floats
            List of reward for each turn.
    
        """
        reward = None
        while reward is None:
            display_dialog_history(mcb.dialog_hx, self.state_estimator.tokenizer)
            try: 
                reward = float(input( "Enter a reward (float) for current dialog: " ))
            except:
                reward = None
        scb.reward = reward
        scb.rewards.append(scb.reward)
        return scb, mcb
    
    def score_trajectory(self, scb, mcb):
        """
        No additional metrics for this wrapper.
        """
        return scb, mcb

class RankingReward(Reward):
    """
    Implements a wrapper on the Reward base object. 
    
    Calculates various precision@k and recall@k metrics based on the RankingLoss criterion.
    
    Parameters
    -------
    state_estimator : StateEstimator object or Callable
        A state estimator model that maps from dialog history text -> embedding vector to represent state information.

    path_to_persona_facts : os.path or string, optional
        Path to list of persona facts. Used to generate candidates for ranking / scoring. 
        If none, a list of persona facts is constructed from the PersonaChat data set and saved locally.
        The default is none.
    """
    def __init__(self, state_estimator, path_to_persona_facts=None):
        super().__init__(state_estimator, None)
        self.criterion = RankingLoss(state_estimator)
        self.candidates = load_persona_facts(path_to_persona_facts)
        self.metrics = {'prec1s': [], 'prec5s': [],
                        'rec5s': [], 'rec10s': [], 'rewards': []}
    def calculate_reward(self, scb, mcb):
        """
        Obtains reward from given persona and state information and passes it to the state callback.
    
        Attributes
        -------
        scb.reward : float
            A scalar reward for current turn.
            
        scb.rewards: List of floats
            List of reward for each turn.
    
        """
        loss = self.criterion(states=scb.state.view(1,-1), 
                              positives=[scb.personas], 
                              candidates=self.candidates)
        scb.reward = - loss.item()
        scb.rewards.append(scb.reward)
        return scb, mcb
    
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
        prec1, prec5, rec5, rec10 = self.criterion.score(states=scb.state.view(1,-1), 
                                                        positives=[scb.personas], 
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
        
