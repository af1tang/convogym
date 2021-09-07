#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:00:13 2021

@author: af1tang
"""
import argparse
from convogym.gyms import Gym, ActiveGym, get_random_persona, get_custom_persona
from convogym._decoder import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('-M', '--mode', type=int, 
                        dest='mode', default=2,
                        help='''interaction mode (0, 1 or 2): 
                        (0) interact: you give prompts to the persona model,
                        (1) observe: you observe some bot-on-bot action.
                        (2) control: you picks things to talk about, 
                                        the model decodes a response for you.''')
    parser.add_argument('-convos', '--num_convos', type=int, 
                        dest='num_convos', default=1,
                        help='''number of conversations to try (default 1). 
                                pick -1 to go through the entire training set personas.''')                                        
    parser.add_argument('-turns', '--num_turns', type=int, 
                        dest='turns', default=8,
                        help='number of turns in conversation (default 8)')
    parser.add_argument('-maxlen', '--max_length', type=int, 
                        dest='max_length', default=1000,
                        help='max num of tokens in convo (default 1000)') 
    parser.add_argument('-k', '--top_k', type=int,
                        dest='top_k', default=10,
                        help='top_k sampling parameter (default 10)')
    parser.add_argument('-p', '--top_p', type=float, 
                        dest='top_p', default=.92,
                        help='nucleus sampling parameter (default 0.92)')    

    args = parser.parse_args()
    while args.mode not in (0,1,2):
            try: 
                args.mode = int(input('''> Please input a valid mode number (0,1,2): 
                                      
                (0) interact: you give prompts to the persona model,
                (1) observe: you observe some bot-on-bot action.
                (2) control: you picks things to talk about, the model decodes a response for you.
                                          
                >> MODE: '''))
            except:
                args.mode = -99
    if args.mode == 0:
        gym = Gym(model = model, interactive = True, 
                  reset_persona_func=get_custom_persona, length=args.turns)
    elif args.mode == 1:
        gym = Gym(model = model, interactive = False,
                  reset_persona_func=get_random_persona, length=args.turns)
    elif args.mode == 2:
        gym = ActiveGym(model = model, length = args.turns)
    gym.sim_convos(args.num_convos)
