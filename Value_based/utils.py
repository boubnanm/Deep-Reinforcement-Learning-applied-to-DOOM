import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

import numpy as np
import random
from vizdoom import *

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore') 

"""
Environment tools
"""

def create_environment(scenario = 'basic', window = False):
    """
    Description
    ---------------
    Creates VizDoom game instance along with some predefined possible actions.
    
    Parameters
    ---------------
    scenario : String, either 'basic' or 'deadly_corridor', the Doom scenario to use (default='basic')
    window   : Boolea, whether to render the window of the game or not (default=False)
    
    Returns
    ---------------
    game             : VizDoom game instance.
    possible_actions : List, the one-hot encoded possible actions.
    """
    
    game = DoomGame()
    if window:
        game.set_window_visible(True)
        
    else:
        game.set_window_visible(False)
    
    # Load the correct configuration
    if scenario == 'basic':
        game.load_config("scenarios/basic.cfg")
        game.set_doom_scenario_path("scenarios/basic.wad")
        game.init()
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]
        
    elif scenario == 'deadly_corridor':
        game.load_config("deadly_corridor.cfg")
        game.set_doom_scenario_path("deadly_corridor.wad")
        game.init()
        possible_actions = np.identity(6,dtype=int).tolist()
        possible_actions.extend([[0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1]])
    
    return game, possible_actions
       

def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action, frame_skip)
            print ("\treward:", reward)
            time.sleep(0.02)
            
        print ("Result:", game.get_total_reward())
        time.sleep(2)
        
    game.close()

"""
Preproessing tools
"""
def get_state(game):
    """
    Description
    --------------
    Get the current state from the game.
    
    Parameters
    --------------
    game : VizDoom game instance.
    
    Returns
    --------------
    state : 4-D Tensor, we add the temporal dimension.
    """
    
    state = game.get_state().screen_buffer
    return state[:, :, None] 

# class Crop(object):
#     """Crops the given PIL.Image using given pixel coordinates.

#     Args:
#         i – int, Upper pixel coordinate.
#         j – int, Left pixel coordinate.
#         h – int, Height of the cropped image.
#         w – int, Width of the cropped image.
#     """

#     def __init__(self, i, j, h, w):
#         self.i = i
#         self.j = j
#         self.h = h
#         self.w = w

#     def __call__(self, img):
#         """
#         Args:
#             img (PIL.Image): Image to be cropped.

#         Returns:
#             PIL.Image: Cropped image.
#         """
#         return img.crop((self.i, self.h, self.j, self.w))
    
# def transforms(crop = False, coords = (30, 300, 60, 180), resize = (120, 160)):
#     """
#     Description
#     -------------
#     Preprocess image screen before feeding it to a neural network.
    
#     Parameters
#     -------------
#     crop   : boolean, whether to crop or not (default=False)
#     coords : tuple, when crop is True, the coordinates to apply cropping (default=(30, 300, 60, 180)).
#              Be careful to the value of this parameter with respect to the screen resolution.
#     resize : tuple, shape of the resized frame (default=(120,160))
    
#     Returns
#     -------------
#     torchvision.transforms.transforms.Compose object, the composed transformations.
#     """
    
#     if crop:
#         return T.Compose([T.ToPILImage(),
#                     Crop(coords[0], coords[1], coords[2], coords[3]),
#                     T.Resize(resize),
#                     T.ToTensor()])
        
#     else:
#         return T.Compose([T.ToPILImage(),
#                     T.Resize(resize),
#                     T.ToTensor()])

def transforms(resize = (120, 160)):
    """
    Description
    -------------
    Preprocess image screen before feeding it to a neural network.
    
    Parameters
    -------------
    resize : tuple, shape of the resized frame (default=(120,160))
    
    Returns
    -------------
    torchvision.transforms.transforms.Compose object, the composed transformations.
    """
    
    return T.Compose([T.ToPILImage(),
                T.Resize(resize),
                T.ToTensor()])
    
preprocess_frame = transforms()

def stack_frames(stacked_frames, state, is_new_episode, maxlen = 4):
    """
    Description
    --------------
    Stack multiple frames to create a notion of motion in the state.
    
    Parameters
    --------------
    stacked_frames : collections.deque object of maximum length maxlen.
    state          : the return of get_state() function.
    is_new_episode : boolean, if it's a new episode, we stack the same initial state maxlen times.
    maxlen         : Int, maximum length of stacked_frames (default=4)
    
    Returns
    --------------
    stacked_state  : 4-D Tensor, same information as stacked_frames but in tensor. This represents a state.
    stacked_frames : the updated stacked_frames deque.
    """
    
    # Preprocess frame
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([frame[None] for i in range(maxlen)], maxlen=maxlen) # We add a dimension for the batch
        # Stack the frames
        stacked_state = torch.cat(tuple(stacked_frames), dim = 1)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame[None]) # We add a dimension for the batch
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.cat(tuple(stacked_frames), dim = 1)
    
    return stacked_state, stacked_frames


"""
epsilon-greedy
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, model, possible_actions):
    """
    Description
    -------------
    Epsilon-greedy policy
    
    Parameters
    -------------
    explore_start    : Float, the initial exploration probability.
    explore_stop     : Float, the last exploration probability.
    decay_rate       : Float, the rate at which the exploration probability decays.
    state            : 4D-tensor (batch, motion, image)
    model            : models.DQNetwork or models.DDDQNetwork object, the architecture used.
    possible_actions : List, the one-hot encoded possible actions.
    
    Returns
    -------------
    action              : np.array of shape (number_actions,), the action chosen by the greedy policy.
    explore_probability : Float, the exploration probability.
    """
    
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*decay_step)
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
        
    else:
        Qs = model.forward(state.cuda())
        action = possible_actions[int(torch.max(Qs, 1)[1][0])]

    return action, explore_probability

"""
Double Q-learning tools
"""
def update_target(current_model, target_model):
    """
    Description
    -------------
    Update the parameters of target_model with those of current_model
    
    Parameters
    -------------
    current_model, target_model : torch models
    """
    target_model.load_state_dict(current_model.state_dict())








