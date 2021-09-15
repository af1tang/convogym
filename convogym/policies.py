#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:07:07 2021

@author: af1tang
"""
import os
import warnings
import random, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from convogym.load_data import default_action_space as action_space
from convogym._configs import opts
from convogym.utils._optim_helpers import _process_dqn_batch
from convogym.utils._device import to_device
from convogym.utils._swa_utils import AveragedModel
from convogym.utils._visualization import plot_losses


from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

default_policy = nn.Sequential(nn.Linear(1025, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, 512),
                        nn.Tanh(), nn.Dropout(.1),
                        nn.Linear(512, len(action_space)))

class Policy(nn.Module):
    """
    The Policy object is a base dialog policy that outputs a distribution over an action space (turn-level goals) given state information about a dialog trajectory.
    
    The base Policy object is not trainable. A wrapper object should be created with a new self.update_policy method to specify the training loop.
    
    Parameters
    ----------
    policy : torch.nn.Module, optional
        Pytorch network that maps from state space representation of dialog history to a distribution over action space. 
        The default is default_policy.
        
    lr : float, optional
        Learning rate for the policy network. The default is 1e-4.
        
    action_space : List of string, optional
        List of strings of action space. The default is action_space.
        
    from_pretrained : os.path or string, optional
        A checkpoint path to save or load the parameters of the policy.
        
    EPS_START : float, optional
        Float (0, 1.0) for the initial epsilon parameter in epsilon-greedy sampling. The default is .5.
        
    EPS_END : float, optional
        Float (0, 1.0) for the final epsilon parameter in epsilon-greedy sampling. The default is .5.
        
    EPS_DECAY : int, optional
        Number of steps between EPS_START and EPS_END. The default is 2048.

    Attributes
    -------
    huber_loss : nn.Module 
        Huber loss function used to calculate the Bellman Error during Q-learning. Not used unless Q-learning is the training method.
        
    action_space : List of strings
        List of possible actions that the policy can output. These correspond to turn-level goals that get incorpoated as prefix tokens in the decoder model for response generation.
    
    Examples
    -------
    >>> from convogym.policies import Policy # default policy
    >>> from convogym.utils._device import to_var
    >>> import torch, torch.nn as nn
    >>> action_space = [ 
        'talk about work.', 
        'ask about hobbies.',
        'talk about movies.'
        ]
    >>> policy_net = nn.Sequential(nn.Linear(100, len(action_space))
                                )
    >>> policy = Policy( policy = policy_net,
                        action_space = action_space) 
    >>> action = policy.act( inp=to_var(torch.rand((1,100))), turn=1 ) 
    >>> print("[%d] %s" %(action, action_space[action]))
    [1] ask about hobbies.        
    >>> inp = tokenizer.encode(''.join(['<|act|>'] + [action_space[action]] + ['<|sep|>'] + ['<|p1|>'] + ['<|sep|>'] + ['<|start|>']) )
    >>> print(inp)        
    [50262, 2093, 546, 45578, 13, 50257, 50260, 50257, 50259]        
    >>> tokenizer.decode(model.generate(to_var(inp).long().view(1,-1)).tolist()[0][len(inp):] )        
    'hello do you have any hobbies?<|endoftext|>'        

    """    
    def __init__(self, policy=default_policy,
                 from_pretrained=None,
                 lr=1e-4, action_space=action_space,
                EPS_START=.5, EPS_END=.05, EPS_DECAY=int(2e4)):
        super(Policy, self).__init__()
        self.policy = to_device(policy)
        # loss func
        self.huber_loss = nn.SmoothL1Loss()
        # epsilon sampling params
        self.EPS_START, self.EPS_END, self.EPS_DECAY = EPS_START, EPS_END, EPS_DECAY
        self.reset_stats()
        self.action_space = action_space
        # load params, warn if not trained
        self.checkpoint_path = from_pretrained
        if from_pretrained:
            self.load_params()
        else:
            warnings.warn("This policy has NOT been trained yet.")
        
    def reset_stats(self):
        """
        Reset global stats. 
        
        Attributes
        -------
        global_step : int
            Total number of actions (i.e., turns) taken in the environment.
        
        global_epoch : int 
            Total number of iterations of the training data.
            
        stats : dict
            Dictionary of losses at each batch update. Loss can be: 
                - reward in policy gradient
                - Belmman error in Q-learning
                - cross entropy in imitation learning.

        """
        # eps greedy params
        self.global_step = 0
        # logging
        self.stats, self.global_epoch = {'losses': {}}, 0
    
    def act(self, inp, turn):
        """
        Outputs a distribution over self.action_space and samples an action from the distribution.

        Parameters
        ----------
        inp : torch.tensor
            State representation of current dialog trajectory. The default representation is a (1, d) shaped torch.tensor feature vector in R^d.
            
        turn : int
            Current turn in conversation. In the initial turn, the action is sampled from the output distribution; otherwise the argmax action is chosen.

        Returns
        -------
        action : int
            Integer index of the action chosen from given action space.

        """
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(inp)
            p = F.softmax(logits,-1)
            # sampling w/ eps-greedy
            eps = random.random()
            threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.global_step / self.EPS_DECAY)
            self.global_step +=1
            if eps > threshold:
                if turn == 0:
                    action = torch.multinomial(p, num_samples=1).item()
                else:
                    action = p.max(1)[1].item()
            else:
                action = random.randrange(len(self.action_space))
        return action
    
    def update_policy(self, *args, **kwargs):
        """
        Not implemented
        """
        warnings.warn("update_policy() is called but not implemented.")

    def load_params(self):
        """
        Try to load the parameters of the identifier model using load_path. Raises an exception if fails.
        """
        try: 
            state_dict = torch.load(self.checkpoint_path)
            self.policy.load_state_dict(state_dict)
        except Exception as e:
            warnings.warn(str(e) + '''
                          
                          This policy model is NOT trained yet.
                          ''')
                
    def save(self):
        """
        Saves model parameters and config to save_path.
        """
        torch.save(self.policy.state_dict(), self.checkpoint_path)


class DQNPolicy(Policy):
    """
    A wrapper policy object that inherits from the base Policy object.
    
    DQNPolicy object specifies a training loop for the policy network using the self.update_policy() method. 
    
    New Attributes
    -------
    swa_policy : Callable or torch.nn.Module
        The target network used to calculate the Q-target.
        
    optimizer : torch.optim object
        Optimizer used to implement backpropagation on policy parameters.
        
    Scheduler: torch.optim.lr_scheduler object
        Scheduler used to change learning rate after each gradient step.
    
    """
    def __init__(self, policy=default_policy,
                 checkpoint_path=os.path.join(opts.example_path, 'policy.pt'),
                 lr=1e-4, action_space = action_space, gamma=.95,
                EPS_START=.5, EPS_END=.05, EPS_DECAY=int(2e4), t_total=int(2e4)):
        super().__init__(policy=policy, checkpoint_path=checkpoint_path,
                         lr=lr, action_space=action_space, 
                         EPS_START=EPS_START, EPS_END=EPS_END, EPS_DECAY=EPS_DECAY)
        # optim and sched
        self.optimizer = AdamW(self.policy.parameters(), lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, 
                                                num_training_steps=t_total)
        self.gamma = gamma
        # swa tools
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                0.99 * averaged_model_parameter + 0.01 * model_parameter
        self.swa_policy = AveragedModel(self.policy, avg_fn = ema_avg)
        self.swa_policy.eval()
    
    def update_policy(self, memory_buffer, num_train_epochs=1):
        self.policy.train()
        for epoch in range(num_train_epochs):
            print()
            dataloader = DataLoader(memory_buffer, batch_size=64, shuffle=True)
            epoch_loss = []
            for i, batch in enumerate(dataloader):#dataloader_old):
                # get batch on buffer trajectories 
                xx, y, q_targets = _process_dqn_batch(batch, self.policy, self.swa_policy, self.gamma)
                # forward
                yhat = self.policy(xx)   # logits
                loss = self.huber_loss(yhat[torch.arange(len(y)),y], q_targets)      
                # backward
                self.policy.zero_grad()
                loss.backward()
                self.optimizer.step()
                # take swa (polyak avg)
                self.swa_policy.update_parameters(self.policy)
                # tracking
                epoch_loss.append(loss.item())
            print("epoch: %d | loss: %.2f | lr: %s" %(epoch, np.mean(epoch_loss), str(self.scheduler.get_last_lr()[-1])))
            # on epoch end
            self.scheduler.step()
            self.stats['losses'][self.global_epoch] = {'dqn_loss': np.mean(epoch_loss)}
            plot_losses(self.stats['losses'], title='dqn_loss' )
            self.global_epoch +=1