#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 20:25:09 2021

@author: af1tang
"""
import torch, random
from _configs import opts
from _action_space import action_space
from _tokenizer import start_tok
from _personas import ( train_personas, test_personas,
                        get_custom_persona, get_random_persona, 
                        get_sequence_personas
                        )
from utils._visualization import display_dialog_history 
from utils._reshape import flatten 
from utils._turn_filter import to_tokens

from learners import Learner
from agents import Agent
from callbacks import StateCb, MessageCb


## gym environments ##
class Gym:
    """
    The base Gym environment object used to train conversational agents. 
    
    A generative decoder (self.model) is used to decode turn-level responses. 
    
        - "user" (self.user) refers to the person 1 (initiator of convesation).
        - "agent" (self.agent) referes to person 2 (conversational partner).
        
    The self.user object may correspond to user inputs if self.interactive=True. 
    Otherwise, self.user is another Agent object (self.model parameterized by another set of persona facts). 

    Conversational histories and personality parameters are recorded for both the agent and the user.
    
    Parameters
    ----------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT.
        
    tokenizer : Huggingface transformer tokenizer (default = GPT2Tokenizer).
        The corresponding tokenizer for the transformer model. 
        The default tokenizer is af1tang/personaGPT.
        
    user : Agent object, optional
        If interactive=True, user is set to a function that takes user input strings as responses. 
        Otherwise, a new Agent object is initialized with a persona sampled from self.reset_persona_func.
        (default = None)
    
    interactive : bool, optional
        Whether to use human interaction. 
        (default = True)
        
    reset_persona_func : partial function, optional
        A sampling function that generates a set of persona facts from list of persona profiles.
        (default = get_custom_persona)
        
    length : int, optional
        Length of a given conversation. 
        (default = 8)
        
    top_k : int, optional
        Parameter to control number of candidates during top-k sampling at each decoding step. 
        (default = 10)
        
    top_p : float32, optional
        Float32 between [0.0 - 1.0] used for nucleus sampling. The 
        (default = 0.92)
        
    max_length : int, optional
        Maximum number of tokens allowed in conversation. 
        (default = 1024)

    Attributes
    -------
    data : dictionary
        A dictionary of dialog histories and persona facts for each conversation conducted.
            - hx: list of strings
                List of strings corresponding to each turn response.
            - p1: list of strings
                List of strings corresponding to the person facts of the first person.
            - p2: list of strings
                List of strings corresponding to the person facts of the second person.

    """
    def __init__(self, model, tokenizer, user=None, interactive = True,  
                 reset_persona_func=get_custom_persona, 
                 length=8, top_k=10, top_p = .92, max_length=1024 ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.user = user
        self.interactive = interactive
        self.length, self.top_k, self.top_p, self.max_length = length, top_k, top_p, max_length
        self.reset_persona_func = reset_persona_func
        self.data = {'hx': [], 'p1': [], 'p2': []}
    
    def _reset_agents(self, persona_input, agent_personas, user_personas):
        """
        Reset self.agent and self.user Agent objects to handle conversations. 

        Parameters
        ----------
        persona_input : List of Strings
            A set of input personas to parameterize the agent (conversational partner). 
            If None, it is sampled from self.reset_persona_func.
            
        agent_personas : List of list of strings
            List of person profiles (each 3-5 strings of persona facts). 
            
            example: 
                - _personas.train_personas used during training self.agent
                - _personas.test_personas used during evaluation of self.agent
            
            (default = _personas.train_personas) 
            
        user_personas : List of list of strings
             List of person profiles (each 3-5 strings of persona facts).
            
            example:
                -  _personas.train_personas used during both training and testing of self.agent
            (default = _personas.train_personas) 
            
        Returns
        -------
        None.

        """
        if persona_input:
            self.agent = Agent(persona_input, top_k=self.top_k,
                               top_p=self.top_p, max_length=self.max_length)
        else:
            persona = self.reset_persona_func(agent_personas)
            self.agent = Agent(persona, reverse=False, top_k=self.top_k, 
                              top_p=self.top_p, max_length=self.max_length)
            
        if self.interactive:
            # human input
            self.user = self._interact
        else:
            # another persona model
            persona = self.reset_persona_func(user_personas)
            self.user = Agent(persona, reverse=True, top_k=self.top_k, 
                              top_p=self.top_p, max_length=self.max_length)
        
    def _interact(self, model, msg, act):
        """
        Get user input for response. Only used when self.interactive=True.

        Parameters
        ----------
        model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
            Currently supports Pytorch Huggingface (transformers) models only. 
            The default model is af1tang/personaGPT model card.
            
        msg : String
            Bot input message as a list of integers (tokenized).
            
        act : bool
            Whether to use action prefix instead of persona prefix.
            NOT used during base Gym. 

        Returns
        -------
        outp : List of integers
            Tokenized (string -> int) list of response tokens.

        """
        if msg:
            print("Bot: {}".format(self.tokenizer.decode(msg, skip_special_tokens=True)))
        outp = self.tokenizer.encode(input(">> User: ") + self.tokenizer.eos_token)
        return outp
    
    def _on_convo_begin(self, scb, mcb):
        """
        Callback updates for state (scb) and message (mcb) callbacks before conversation begins.

        Default behavior updates state persona tracker with the agent persona facts. 
        
        Parameters
        ----------
        scb : StateCb object
            State callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Message callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        Returns
        -------
        scb : StateCb object
            Updated callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Updated callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        """
        scb.personas = self.agent.p2
        return scb,mcb
        
    def _on_user_begin(self, scb, mcb):
        """
        Callback updates for state (scb) and message (mcb) callbacks before generating user response.

        Parameters
        ----------
        scb : StateCb object
            State callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Message callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        Returns
        -------
        scb : StateCb object
            Updated callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Updated callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        """
        return scb, mcb
    
    def _on_user_end(self, scb,mcb):
        """
        Callback updates for state (scb) and message (mcb) callbacks after generating user response.

        Updates dialog history (mcb.dialog_hx) with encoded user message. 
        
        Parameters
        ----------
        scb : StateCb object
            State callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Message callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        Returns
        -------
        scb : StateCb object
            Updated callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Updated callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        """
        mcb.dialog_hx.append(mcb.msg)
        return scb, mcb
    
    def _on_agent_end(self, scb, mcb):
        """
        Callback updates for state (scb) and message (mcb) callbacks after generating agent response.

        Default behavior:
            - updates dialog history with agent response
            - checks whether conversation ends (based on turn count and length)
        
        Parameters
        ----------
        scb : StateCb object
            State callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Message callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        Returns
        -------
        scb : StateCb object
            Updated callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Updated callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        """
        mcb.dialog_hx.append(mcb.msg)
        scb.turn +=1
        if scb.turn >= self.length:
            scb.done = True
        return scb, mcb
    
    def _on_convo_end(self, scb, mcb):
        """
        Callback updates for state (scb) and message (mcb) callbacks after conversation ends.

        Default behavior: 
            - prints dialog history 
            - updates self.data with dialog history, persona 1 and persona 2 facts.
            
        Parameters
        ----------
        scb : StateCb object
            State callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Message callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        Returns
        -------
        scb : StateCb object
            Updated callback object that tracks state information of conversation.
            See callbacks.StateCb documentation for details.
            
        mcb : MessageCb object
            Updated callback object that tracks current message information of conversation.
            See callbacks.MessageCb documentation for details.

        """
        if not self.interactive:
            print('p1: ')
            print()
            for p in self.user.p1: print(p)
        print('-'*10)
        print('p2: ')
        print()
        for p in scb.personas: print(p)
        print('-'*10)
        print()
        display_dialog_history(mcb.dialog_hx)
        self.data['hx'].append(to_tokens( mcb.dialog_hx ))
        if self.interactive:
            self.data['p1'].append([])
        else:
            self.data['p1'].append(self.user.p1)
        self.data['p2'].append(scb.personas)
        del scb, mcb

    def _sim_convo(self, persona_input = None, 
                   agent_personas=None, user_personas=None): 
        """
        Generate a dialog trajectory consisting of 
            - dialog history
            - prefix information (persona or action prefixes) for each agent
            - state information associated w/ each turn (scb)

        Parameters
        ----------
        persona_input : List of strings, optional
            Used when custom persona is provided during self.interactive=True. The default is None.
            
        agent_personas : List of strings, optional
            Used when agent personas are sampled from a predfined list of personas. The default is None.
            
        user_personas : List of strings, optional
            Used when user personas are sampled from a predfined list of personas. for . The default is None.

        Returns
        -------
        None.

        """
        self._reset_agents(persona_input, agent_personas, user_personas)
        scb, mcb = StateCb(personas=agent_personas), MessageCb()
        scb, mcb = self._on_convo_begin(scb,mcb)
        # run for turns, updating rewards at each turn
        while not scb.done:
            # person 1 (user) 
            scb,mcb = self._on_user_begin(scb,mcb)             
            mcb.msg = self.user(self.model, mcb.msg, act=scb.act)
            scb,mcb = self._on_user_end(scb,mcb)

            # person 2 (agent)
            mcb.msg = self.agent(self.model, mcb.msg, act=scb.act)
            scb, mcb = self._on_agent_end(scb,mcb)
        self._on_convo_end(scb, mcb) 
    
    def sim_convos(self, num_convos=1, agent_personas=None, user_personas=None):
        """
        Simulates conversational episodes between user and agent dialog models using a given set of agent_personas and user_personas. 

        Parameters
        ----------
        num_convos : int, optional
            Number of conversations to simulate. The default is 1.
            
        agent_personas : List of strings, optional
            Used when agent personas are sampled from a predfined list of personas. The default is None.
            
        user_personas : List of strings, optional
            Used when user personas are sampled from a predfined list of personas. for . The default is None.


        Returns
        -------
        None.

        """
        print("Conducting conversations ...")
        print()
        for i in range(num_convos):
            self._sim_convo(None, agent_personas, user_personas)
 
    def run_evaluation(self):
        """
        Not implemented in base Gym object. 
        
        Used in RLGym to evaluate trained agent policies.
        """
        raise NotImplementedError("evaluation loop not implemented for current environment.")
    
        
# active learning environments
class ActiveGym(Gym):
    """
    The active learning Gym environment is built from the base Gym environment. 
    
    In this setting, the user (human) chooses action prefixes rather than personas to control decoded responses.
        - Unlike persona prefixes ("<|p1|>", "<|p2|>") which delimits persona facts, 
            action prefix ("<|act|>") indicates the turn-level goals used to condition the language model for decoding.
        - Action space defines a set of "actions" (turn-level objective) that can be collected to train the model.
        - Active learning data can be saved for downstream training tasks. 
    
    Parameters
    ----------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT model card.
        
    tokenizer : Huggingface transformer tokenizer (default = GPT2Tokenizer).
        The corresponding tokenizer for the transformer model. 
        The default tokenizer is af1tang/personaGPT.
        
    training_data : torch.utils.data.Dataset or List of tuples , optional
        List of training batches to sample batches from. Used to re-fit model to prevent catastrophic forgetting.
                
    user : Agent object, optional
        In ActiveGym, the user is set to an Agent object without any persona facts. 
        At each turn, the user provides an action (turn-level goal) to train the model with.
        
    length : int, optional
        Length of a given conversation. 
        (default = 8)
        
    top_k : int, optional
        Parameter to control number of candidates during top-k sampling at each decoding step. 
        (default = 10)
        
    top_p : float32, optional
        Float32 between [0.0 - 1.0] used for nucleus sampling. The 
        (default = 0.92)
        
    max_length : int, optional
        Maximum number of tokens allowed in conversation. 
        (default = 1024)
        
    train_model : bool, optional
        Whether to fine-tune the model during active learning episodes. If False, the model is not fine-tuned between corrections.
        (default = True)

    Attributes
    -------
    data : dict
        Active learning data in dictionary format: 
            - X: List of list of int
                Input tokens (encoded by tokenizer) of action prefix + dialog history.
            - Y: List of list of int
                Human-provided response (ground truth).
            - dialog_hx: List of string
                List of responses at each turn.
            - actions: List of int
                List of actions (turn-level goals) represented by indices 
                
    learner : Learner object
        Training wrapper to update the decoder model (self.model) on active learning data.

    """
    def __init__(self, model, tokenizer, training_data=None,
                 user=None, length=8, top_k=10, top_p = .92,
                 max_length=1024, train_model = False):
        
        super().__init__(model=model, tokenizer=tokenizer, user=None, interactive=True, 
                         reset_persona_func=get_sequence_personas,
                         length=length, top_k=top_k, top_p=top_p, max_length=max_length)
        self.train_model = train_model
        if train_model:
            raise ValueError("training_set cannot be None while train_model = True.")
            self.learner = Learner(model = model, training_data = training_data, 
                                   lr = opts.lr, 
                                   use_param_groups = opts.use_param_groups,
                                   schedule_func=opts.schedule_func,
                                   gradient_accumulation_steps=opts.gradient_accumulation_steps,
                                   max_grad_norm=opts.max_grad_norm,
                                   optim_func=opts.optim_func,
                                   total_iters=opts.total_iters)
            
        self.data = {'X': [], 'y': [], 'dialog_hx': [], 'actions': []}
    
    def _reset_agents(self, persona_input, agent_personas=None, user_personas=None):
        """
        In ActiveGym, the user is set to an Agent object without any persona facts. 
            - At each turn, the user provides an action (turn-level goal) to train the model with.
            - At the end of each conversation, a new persona profile (3-5 persona facts) is sampled from a list of training personas.
            
        """
        self.user = Agent([], top_k=self.top_k, top_p=self.top_p, max_length=self.max_length)
        self.agent = Agent(persona_input, top_k=self.top_k,
                               top_p=self.top_p, max_length=self.max_length)
        
    def _on_convo_begin(self, scb, mcb):
        """
        At convo begin, set persona facts (ground truth) to agent persona facts. 

        Parameters
        ----------
        scb.personas : List of string
            Agent's persona facts set as ground truth. (User has not persona facts, uses action codes instead.)
            
        scb.record : bool
            Whether to record convo to self.data['dialog_hx'] and self.['actions'].
            (default = True)
            
        """
        scb.personas = self.agent.p2
        scb.record = True
        return scb,mcb
        
    def _on_user_begin(self, scb, mcb):
        """
        On user begin: 
            - User selects an action (turn-level goal) from action space. 
            - A candidate response is generated using the action code.
            - Updates message callback 'x' state (mcb.x) to the prefix tokens of the user agent.
                If first turn, '<|start|>' token is ommitted (used as first token in the response).
        """
        if mcb.msg is not None:
            mcb.dialog_hx.append(mcb.msg)
        action = None
        while action not in action_space:
            display_dialog_history(mcb.dialog_hx)
            print()
            print(" actions: ")
            for k,v in enumerate(action_space): print(k,v)
            try:
                int_act = int(input(" input [0-10]: " ))
                action = action_space[int_act]
            except:
                action = None            
        scb.action = action
        scb.actions.append(int_act)
        scb.act = True
        self.user.p1 = [scb.action]
        # cache current x
        x = self.tokenizer.encode(''.join(['<|act|> '] + self.user.p1 + ['<|sep|>'] + ['<|p1|>'] + [] + ['<|sep|>'] + ['<|start|>']))
        x += flatten(mcb.dialog_hx)
        x = torch.tensor([x])
        # set inp as input_ids for dataset
        if scb.turn == 0:
            x = x[:, :-1]
        mcb.x = x
        return scb, mcb
    
    def _on_user_end(self, scb,mcb):
        """
        On user end:
            - Check if decoded response is sensible and uses action (turn-level goals) correctly.
                - if satisfactory (y): continue conversation 
                - if unsatisfactory (n): user provides corrective response (ground-truth)
                
            - If user provides a corrected output, the conversation ends (scb.done=True) and the dialog history is not recorded (scb.record=False).
        
            - However, self.data is updated with the prefix tokens (mcb.x) and user response and added to the growing active learning dataset.
        
            - The last batch self.data['X'][-1] and self.data['y'][-1] is used as the active learning batch to update parameters.

        """
        mcb.dialog_hx.append(mcb.msg)
        # check if need revision
        print(); print('-'*50)
        display_dialog_history(self.user.dialog_history)
        print('-'*12, ' iter %d, turn %d '%(self.iter, scb.turn), '-'*12 )
        print("action: ", scb.action)
        decision = input(" continue? [y/n] ")
        # decision tree
        if decision == 'y':
            # augment even more turns to active data
            # self.data['X'].extend(mcb.x.tolist()); self.data['y'].append(mcb.msg)
            # continue conversation
            x = self.tokenizer.encode(''.join(['<|p2|>'] + scb.personas + ['<|sep|>'] + ['<|start|>']))
            x += flatten(mcb.dialog_hx)
            x = torch.tensor([x])
            mcb.x = x
        else:
            y = [[]]
            # get corrected user response
            while len(y[0]) < 2:
                y = self.tokenizer.encode(input("  >> user: ") + self.tokenizer.eos_token, return_tensors='pt')
            if scb.turn ==0:
                y = torch.cat( (torch.tensor([start_tok]).unsqueeze(0), y), -1)
            # extend active learning data and prepare active learning batch
            self.data['X'].extend( mcb.x.tolist() ); self.data['y'].extend( y.tolist())
            mcb.batch = (mcb.x,y)
            # retart convo
            scb.done, scb.record = True, False
        scb.act = False
        return scb, mcb
    
    def _on_agent_end(self, scb, mcb):
        """
        On agent end:
            - Check if decoded response is sensible and uses action (turn-level goals) correctly.
                - if satisfactory (y): continue conversation 
                - if unsatisfactory (n): user provides corrective response (ground-truth)
                
            - If user provides a corrected output, the conversation ends (scb.done=True) and the dialog history is not recorded (scb.record=False).
            
            - self.data is updated with the prefix tokens (mcb.x) and user response and added to the growing active learning dataset.
            
            - The last batch self.data['X'][-1] and self.data['y'][-1] is used as the active learning batch to update parameters.

        """
        if not scb.done:
            display_dialog_history(self.agent.dialog_history)
            print('-'* 12, ' iter %d, turn %d ' %(self.iter, scb.turn), '-' * 12)
            print(" personas: ")
            for p in scb.personas: print(p)
            decision = input( " continue? [y/n] " )
            if decision == 'y':
                scb.turn +=1
                if scb.turn >= self.length:
                    scb.done = True
            else:
                y = [[]]
                while len(y[0]) < 2:
                    y = self.tokenizer.encode(input("  >> user: ") + self.tokenizer.eos_token, return_tensors='pt')
                # extend active learning data and prepare active learning batch
                self.data['X'].extend( mcb.x.tolist() ); self.data['y'].extend( y.tolist())
                mcb.batch = (mcb.x,y)
                # retart convo
                scb.done, scb.record = True, False
        return scb, mcb
    
    def _on_convo_end(self, scb, mcb):
        """
        On conversation end: 
            - if scb.record (i.e., no user corrections), save dialog history and actions used
            - if user corrections, update model if training mode.
        """
        if scb.record:
            self.data['dialog_hx'].append(to_tokens(mcb.dialog_hx))
            self.data['actions'].append(scb.actions)
        elif self.train_model:
            self.learner.fit_on_active_batch(mcb.batch)
        del scb, mcb
        
    def sim_convos(self, num_convos=9999):
        """
        Generate dialog trajectories with 
            max number of episodes = min( size of persona set * epochs, num_convos). 
        """
        print("Conducting conversations ...")
        print()
        max_num_convos = min(len(train_personas) * opts.num_epochs, num_convos)
        for self.iter, persona in enumerate(self.reset_persona_func(train_personas)):
            self._sim_convo(persona, None, None)
            if self.iter > max_num_convos:
                break

# RL environments
class RLGym(Gym):
    """
    The reinforcement learning Gym environment is built from the base Gym environment. 
    
    In this setting, the user is a policy that learns to output a distribution over an action space (previously defined turn-level goals).
    The policy is trained to direct conversations toward dialog-level goals, which is represented as a reward object (self.reward_obj). 
    Like the active learning setting, the turn-level goals are incorporated as part of the input (i.e., as context) to the decoder model. 
    Unlike the action learning, the model is fixed, and the policy is trained to optimize over turn-level goals as actions rather than token-level outputs by the decoder model. 
    
    Parameters
    ----------
    model : A Huggingface transformer decoder model (default = GPT2LMHeadModel).
        Currently supports Pytorch Huggingface (transformers) models only. 
        The default model is af1tang/personaGPT model card.
        
    tokenizer : Huggingface transformer tokenizer (default = GPT2Tokenizer).
        The corresponding tokenizer for the transformer model. 
        The default tokenizer is af1tang/personaGPT.
        
    policy : Policy object
        The policy outputs a turn-level goal (e.g., "talk about work") to be incorporated into the user object (self.user). 
        See convogym.policies for details.
            
    env : Env object
        An environment object to estimate state information from dialog history.
        See convogym.environments for details.
        
    reward_obj : Reward object
        A reward object to track reward function outputs at each dialog turn.
        See convogym.rewards for details.
        
    max_buffer_size : int, optional
        Maximum replay buffer size to use for training the policy. The default is 1000.
        
    length : int, optional
        Length of a given conversation. 
        (default = 8)
        
    top_k : int, optional
        Parameter to control number of candidates during top-k sampling at each decoding step. 
        (default = 10)
    
    top_p : float32, optional
        Float32 between [0.0 - 1.0] used for nucleus sampling. The 
        (default = 0.92)
    
    max_length : int, optional
        Maximum number of tokens allowed in conversation. 
        (default = 1024)

    Attributes
    -------
    data : dictionary
        A dictionary of dialog histories and persona facts for each conversation conducted.
            - dialog_hx: list of strings
                List of strings corresponding to each turn response.
            - actions: list of int
                List of integers corresponding to index of the sampled action at each turn.
            - personas: list of strings
                List of strings corresponding to the person facts of the agent (second) person.
    
    memory_buffer : List of tuple
        Current dataset of batch tuples of the form 
            (states, contexts, actions, next_states, next_contexts, next_acts, rewards). 
            
        (state, context, action) -> (next_state, next_context), reward is observed at each transition as a part of the Markov Decision Process. 
            
    """
    def __init__(self, model, tokenizer, policy, env, reward_obj, 
                 max_buffer_size = 1000,
                 length=8, top_k=10, top_p = .92, max_length=1024):        
        super().__init__(model=model, tokenizer=tokenizer, user=None, interactive=False, 
                         reset_persona_func=get_sequence_personas,
                         length=length, top_k=top_k, top_p=top_p, max_length=max_length)
        # REQUIRE: policy object
        self.policy = policy
        self.R = reward_obj
        self.Env = env
        self.data = {'dialog_hx':[], 'actions': [], 'personas': []}
        self.memory_buffer, self.max_buffer_size = [], max_buffer_size
    
    def _reset_agents(self, persona_input, agent_personas=None, user_personas=None):
        """
        Resets Agent objects for user (no input personas, uses action prefixes) and agent (randomly sampled persona, uses persona prefixes).
        """
        self.user = Agent([], top_k=self.top_k, top_p=self.top_p, max_length=self.max_length)
        self.agent = Agent(persona_input, top_k=self.top_k,
                               top_p=self.top_p, max_length=self.max_length)
    
    def _on_convo_begin(self, scb, mcb):
        """
        On conversation begin:
            Initializes the state and context objects for tracking using a StateCb callback (scb). 
            The default state and context objects are torch.tensor feature vectors. 
        """
        scb = self.Env.reset(scb)
        scb.personas = self.agent.p2
        return scb, mcb
        
    def _on_user_begin(self, scb, mcb):
        """
        On user begin:
            - Obtains a state and context representation from current dialog history.
            - Policy maps state and context -> action (sampled from action space, no gradient).

        Parameters
        ----------
        int_act : int
            Sampled output (no grad) from policy output. Corresponds the index of an action in action space.
            
        action : string
            Sampled action.

        scb.act : bool
            Used in 
                - self.user._reset_inp(act=scb.act) to prepare input format to include the <|act|> token.
        """
        # sample action from policy
        inp = self.Env.get_policy_inp(scb)
        int_act = self.policy.act(inp, scb.turn)
        # update action
        scb.action = action_space[int_act]
        scb.actions.append(int_act)
        scb.act = True
        self.user.p1 = [scb.action]
        return scb, mcb
    
    def _on_user_end(self, scb,mcb):
        """
        Updates dialog history, switches scb.act = False.
        
        Parameters
        -------
        scb.act : bool
            Used in self.agent._reset_inp(act=scb.act) to prepare input format to use persona prefixes (e.g., <|p2|>).
        """
        mcb.dialog_hx.append(mcb.msg)
        scb.act = False
        return scb, mcb
    
    def _on_agent_end(self, scb, mcb):
        """
        On agent end:
            - Track dialog history and update turn counts.
            - Get (state, context) -> (next_state, next_context) transition from environment. 
            - Get reward for dialog trajectory up to current turn.
        """
        mcb.dialog_hx.append(mcb.msg)
        # update state based on dialog history
        scb, mcb = self.Env.get_curr_state(scb, mcb)
        # calculate reward 
        scb, mcb = self.R.calculate_reward(scb, mcb)
        # transition to next state
        scb, mcb = self.Env.get_next_state(scb, mcb)
        # next turn
        scb.turn +=1
        if scb.turn >= self.length:
            scb.done = True
        return scb, mcb
    
    def _update_memory(self,batch):
        """
        Update memory buffer with current episode of batch tuples. If max buffer size exceeded, delete some episodes.
        """
        self.memory_buffer.extend([list(tuples) for tuples in batch])
        # delete from front of memory batch if full
        if len(self.memory_buffer) > self.max_buffer_size:
            self.memory_buffer = self.memory_buffer[len(batch):]
    
    def _on_convo_end(self, scb, mcb):
        """
        On conversation end:
            - Display dialog history, actions and personas.
            - Update self.data with trajectory data. 
            - Update memory buffer.
        """
        # display
        print('actions: ')
        print()
        for p in scb.actions: print(action_space[p])
        print('-'*10)
        print('p2: ')
        print()
        for p in scb.personas: print(p)
        print('-'*10)
        print()
        display_dialog_history(mcb.dialog_hx)
        # evaluate dialog and log
        self.R.score_trajectory(scb, mcb)
        self.data['dialog_hx'].append(to_tokens(mcb.dialog_hx))
        self.data['actions'].append(scb.actions)
        self.data['personas'].append(self.agent.p2)
        # update memory buffer
        next_states = scb.states[1:]; states = scb.states[:-1]
        next_contexts = scb.contexts[1:]; contexts = scb.contexts[:-1]
        next_acts = scb.actions[1:] + [0]
        batch = list(zip(states, contexts, scb.actions, next_states, next_contexts, next_acts, scb.rewards))
        self._update_memory(batch)
        del scb, mcb
    
    def sim_convos(self, num_convos=9999, training=True):
        """
        Generate dialog trajectories. 
            - If training mode, update policy parameters at end of episode using gradient descent over memory buffer samples. 
            - If test mode, evaluate policy on test set personas, no gradient updates for policy.

        Parameters
        ----------
        num_convos : int, optional
            Max number of conversations to generate. The default is 9999.
            
        training : bool, optional
            Whether to update the policy after each generated trajectory. If True, train personas are used to parameterize agent. The default is True.
        """
        print("Conducting conversations ...")
        print()
        if training:
            max_num_convos = min(len(train_personas) * opts.num_epochs, num_convos)
            for epoch in range(opts.num_epochs):
                for self.iter, persona in enumerate(self.reset_persona_func(train_personas)):
                    print("="*20, "epoch %d, iter %d, training" %(epoch+1, self.iter), "="*20)
                    self._sim_convo(persona, None, None)
                    # policy update
                    print("Offline batch updates ... ")
                    self.policy.update_policy(self.memory_buffer)
                    print()
                    if self.iter > max_num_convos:
                        break
        else:
            max_num_convos = min(len(test_personas), num_convos)
            for self.iter, persona in enumerate(self.reset_persona_func(test_personas)):
                print("="*20, "iter %d, testing" %self.iter, "="*20)
                self._sim_convo(persona, None, None)
                if self.iter > max_num_convos:
                    break
        