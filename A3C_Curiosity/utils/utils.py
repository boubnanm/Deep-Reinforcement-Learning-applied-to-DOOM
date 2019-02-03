import numpy as np
import tensorflow as tf
import scipy
import scipy.signal
import random
import scipy.misc
import csv
import tensorflow.contrib.slim as slim
import os
import moviepy.editor as mpy

from vizdoom import *
from utils.network_params import *


def create_environment(scenario = 'basic', no_window = False, actions_type="all", player_mode=False):
    """
    Description
    ---------------
    Creates VizDoom game instance with provided settings.
    
    Parameters
    ---------------
    scenario : String, either 'basic' or 'deadly_corridor', the Doom scenario to use (default='basic')
    window   : Boolea, whether to render the window of the game or not (default=False)
    
    Returns
    ---------------
    game             : VizDoom game instance.
    possible_actions : np.array, the one-hot encoded possible actions.
    """
    
    game = DoomGame()
    if no_window:
        game.set_window_visible(False)        
    else:
        game.set_window_visible(True)
    
    # Load the correct configuration
    game.load_config(os.path.join("scenarios",params.scenario+".cfg"))
    game.set_doom_scenario_path(os.path.join("scenarios",params.scenario+".wad"))
    
    # Switch to RGB in player mode
    if player_mode:
        game.set_screen_format(ScreenFormat.RGB24)
    
    # Initiliaze game
    game.init()
    
    # Possible predefined actions for the scenario
    possible_actions = button_combinations(scenario)
    
    return game, possible_actions


def button_combinations(scenario='basic'):
    """
    Description
    ---------------
    Returns a list of possible action for a scenario.
    
    Parameters
    ---------------
    scenario : String, Doom scenario to use (default='basic')
    
    Returns
    ---------------
    actions : list, the one-hot encoded possible actions.
    """
    actions = []

    m_left_right = [[True, False], [False, True], [False, False]]  # move left and move right
    attack = [[True], [False]]
    m_forward_backward = [[True, False], [False, True], [False, False]]  # move forward and backward
    t_left_right = [[True, False], [False, True], [False, False]]  # turn left and turn right

    if scenario=='deadly_corridor':
        actions = np.identity(6,dtype=int).tolist()
        actions.extend([[0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0, 1], 
                        [1, 0, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0]])

    if scenario=='basic':
        for i in m_left_right:
            for j in attack:
                actions.append(i+j)

    if scenario=='my_way_home':
        actions = np.identity(3,dtype=int).tolist()
        actions.extend([[1, 0, 1],
                        [0, 1, 1]])

    if scenario=='defend_the_center':
        for i in t_left_right:
            for j in attack:
                actions.append(i+j)

    if scenario=='defend_the_line':
        for i in t_left_right:
            for j in attack:
                actions.append(i+j)

    return actions


def update_target_graph(from_scope,to_scope):
    """
    Description
    ---------------
    Copies set of variables from one network to the other.
    
    Parameters
    ---------------
    from_scope : String, scope of the origin network
    to_scope   : String, scope of the target network
    
    Returns
    ---------------
    op_holder  : list, variables copied.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def process_frame(frame, crop, resize):
    """
    Description
    ---------------
    Crop and resize Doom screen frame.
    
    Parameters
    ---------------
    frame  : np.array, screen image
    crop   : tuple, top, bottom, left and right crops
    resize : tuple, new width and height
    
    Returns
    ---------------
    s      : np.array, screen image cropped and resized.
    """
    y2, y1, x1, x2 = crop
    s = frame[y2:y1,x1:x2]
    s = scipy.misc.imresize(s,list(resize))
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s


def discount(x, gamma):
    """
    Description
    ---------------
    Returns gamma-discounted cumulated values of x
    [x0 + gamma*x1 + gamma^2*x2 + ..., 
     x1 + gamma*x2 + gamma^2*x3 + ...,
     x2 + gamma*x3 + gamma^2*x4 + ...,
     ...,
     xN]
    
    Parameters
    ---------------
    x      : list, list of values
    gamma  : float, top, bottom, left and right crops
    
    Returns
    ---------------
    np.array, gamma-discounted cumulated values of x
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.0):
    """
    Description
    ---------------
    Tensorflow zero-mean, std weights initializer.
    
    Parameters
    ---------------
    std  : float, std for the normal distribution
    
    Returns
    ---------------
    _initializer : Tensorflow initializer
    """
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

    
def make_gif(images, fname, fps=50):
    """
    Description
    ---------------
    Makes gifs from list of images
    
    Parameters
    ---------------
    images  : list, contains all images used to creates a gif
    fname   : str, name used to save the gif
    

    """
    def make_frame(t):
        try: x = images[int(fps*t)]
        except: x = images[-1]
        return x.astype(np.uint8)
    clip = mpy.VideoClip(make_frame, duration=len(images)/fps)
    clip.fps = fps
    clip.write_gif(fname, program='ffmpeg', fuzz=50, verbose=False)