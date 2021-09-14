# Convogym

<p align="center">
<img src="/figs/overview.png?raw=true" width=50% height=50% >
</p>

### A gym environment to train chatbots

convogym is a Python library for simulating conversations between chatbots and for creating new training data based on human-chatbot interactions. 


## Gym Types

There are currently 3 different gym environments: `Gym` (base class), `ActiveGym` (for active learning new turn-level goals), and `RLGym` (use reinforcement learning to guide turn-level goals toward dialog-level goals). 

* The basic `Gym` environment is designed to interact with human users (`interactive=True`) or to simulate dialog episodes with randomly generated personality facts (`interactive=False`). Users can manually enter personality facts (`reset_persona_func=get_custom_persona`) to observe different effects on personalized dialog generation. 

* In `ActiveGym`, human users teaches the decoder model to use new turn-level goals which are represented as contextual inputs to the decoder model. At each turn, the user chooses a turn-level goal for which the decoder tries to incorporate into its responses. If the response does not live up to user expectations, the user is prompted to enter a correct response for which to train the decoder model with. When `train_mode=True`, the decoder parameters are updated after each correction. 

* In `RLGym`, human users no longer choose the turn-level goals, nor do they provide corrections to the responses. Instead, a dialog policy is trained to output a distribution over turn-level goals for which turn-level goals can be sampled from and used to guide the conversation. The default dialog-level objective function is a ranking loss used to detect relevant personality traits (from the PersonaChat dataset). 

## Core Modules 
There are 6 core modules in convogym. 

* `decoders` A module to do controlled text generation using language models. Currently supports [Huggingface](https://huggingface.co/models) transformers such as GPT-2, DialogGPT, etc. The default model is currently the `af1tang/personaGPT` [model card](https://huggingface.co/af1tang/personaGPT), which can handle personalities from the PersonaChat dataset.  

* `agents` A module that handles response generation, formats tokens for natural language understanding (NLU) and controlled natural language generation (NLG). The base class is the `Agent` object, which handles formatting prefix tokens to do conditional text generation. 

* `states` A module to learn a representation of dialog histories. The base class is `StateEstimator` which takes dialog history (tokens) and map to an embedding space (a feature vector in `R^n`).  

* `rewards` A module to train the state estimation and dialog policy. The base class is the `Reward` object, which provides a dialog-level object to guide conversations toward specific end points (dialog-level goals). 

* `policies` A module which learns to output _turn-level goals_ -- goals that decoder model can use to generate customized responses (e.g., personalized for the user, preserves sensibility of responses). The base class is the `Policy` object which interfaces with the training environment. A default `action_space` is provided which defines a preliminary set of turn-level goals (actions) for which a policy learns to work with. This can be overwritten for specific uses.

* `gyms` A module that ties together the core components (decoder, agents, state estimators, reward functions and policy) into an interactive interface from which dialog episodes can be simulated (`interactive=False`) or from which users can specify ground-truth responses for active learning (`interactive=False`). 

## Dependencies

convogym requires the following dependencies: 

* numpy>=1.20.2
* scipy>=1.6.2
* pandas>=1.2.4
* transformers==4.10.0
* dotenv
* tqdm

## Installation

There are 2 ways to install the current package.

The easiest way is to use `pip` installation:

```
pip install -U convogym
```

_(09.12.21) Under construction_

To build from source, use Git to clone the latest repo:

```
git clone git://github.com/af1tang/convogym.git 
cd convogym
```

As of September 2021, convogym is NOT supported on Apple Silicon M1 hardware. 

## Getting Started

#### Interactive Dialog

The decoder model can be interpreted as a straight forward language model that can generate a variety of stylized text. We can load the default decoder model as follows:

```
>>> from convogym._decoder import model
>>> from convogym._tokenizer import tokenizer
```

We can initialize a gym environment to conduct a short conversation (3 rounds) with the decoder model as follows.

```
>>> from convogym.gyms import Gym
>>> from convogym._personas import get_custom_persona, get_random_persona
>>> gym = Gym(model=model, tokenizer=tokenizer, interactive=True, reset_persona_func=get_custom_persona, length=3)
>>> gym.sim_convos(num_convos=1)
```

In this case, `get_custom_persona` prompts us to give our partner, the decoder model, a set of personality facts to go off of. Alternatively, we can use `reset_persona_func=get_random_persona` to sample from a list of personas provided in the `convogym/data` folder. 

```
>>> gym = Gym(model=model, tokenizer=tokenizer, interactive=False, reset_persona_func=get_random_persona, length=3)
>>> gym.sim_convos(num_convos=3)
```

When we set `interactive=False`, conversations are simulated using self-play between 2 decoder models, parameterized by  different personalities which is displayed at the end of each episode. We can also access the dialog history and personalities directly through `gym.data`. 

---

#### Decoding w/ Turn-Level Goals

Suppose we want to teach the decoder model to generate responses related to specific topics (e.g., talk about hobbies) rather than personalities. We can create the following _prefix tokens_ to describe these turn-level goals.

```
>>> from utils._device import to_var # use GPU
>>> goal = "ask about hobbies."
>>> inp = tokenier.encode("<|act|>" + goal + "<|p2><|sep|><|start|>") 
>>> print(inp)        
[50262, 2093, 546, 45578, 13, 50257, 50260, 50257, 50259]        
>>> tokenizer.decode(model.generate(to_var(inp).long().view(1,-1)).tolist()[0][len(inp):] )  
'hello do you have any hobbies?<|endoftext|>'
```

We can find a default list of turn-level goals using 

```
>>> from convogym._action_space import action_space
>>> print(action_space)
```

So how do we train the decoder to utilize _new_ turn-level goals? The answer is through `ActiveGym` (Active Learning Gym). 

```
>>> from convogym.gyms import ActiveGym
>>> from convogym.training_data import train_decoder_data
>>> new_goals = ['talk about pokemon.', 'ask about favorite anime.']
>>> gym = ActiveGym(model=model, tokenizer=tokenizer, action_space=new_goals,
>>>								 training_data=train_decoder_data, train_model=True)
>>> gym.sim_convos(1)
```

In this setting, we are prompted to choose a goal from `new_goals` at each turn. The decoder `model` then tries output the correct response. When `train_model=True`, the decoder model is fine-tuned with gradient descent whenever we provide corrections. 

---

#### Dialog Policy

Now suppose we want to train a model to output turn-level goals. We can use `RLGym` (Reinforcement Learning Gym) to interact with a policy model. 

```
>>> from convogym.gyms import RLGym
>>> from convogym.environments import Env
>>> from convogym.states import StateEstimator
>>> from convogym.rewards import ManualReward
>>> from convogym.rewards import Policy
>>> state_estimator = StateEstimator(model=model, tokenizer=tokenizer, use_pretrained=True)
>>> gym = RLGym( model=model, tokenizer=tokenizer, 
				 policy=DQNPolicy(action_space=new_goals),
				 env=Env(state_estimator),
				 reward_obj=ManualReward(state_estimator),
		  )
>>> gym.sim_convos(training=True)
```

In `ManualReward`, the user provides a ground truth reward for each dialog trajectory. This assumes that the user already have a task-specific reward in mind. 

Alternatively, users can also design dialog-level objective functions to train the policy (`training=True`). For example, the base class `Reward` uses a _ranking loss_ designed for the PersonaChat to identify relevant personalities used to parameterize the decoder model. 

---

#### Examples

Example scripts of various ways to use convogym can be found at `convogym/examples`. 

## How to Contribute

Contributors at all levels of experience are welcomed. 

* If you see an issue, please report it on the issues page. When possible, please provide reproducible code snippet and details of the functions used and the shape of the data involved. 

* Please also open an issue page when making a feature request.

* To contribute, please clone the git repo and create a fork. 


A more detailed documentation page to come.


## How to Cite 

Coming soon.
