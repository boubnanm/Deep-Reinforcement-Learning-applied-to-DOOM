import numpy as np
import tensorflow as tf
import scipy
import scipy.signal
import random
import scipy.misc
import csv
import tensorflow.contrib.slim as slim
import os

from vizdoom import *
from configs import *

# Doom Environment settings
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

def button_combinations(scenario):
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


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame, crop, resize):
    #s = frame[30:-35,20:-20]
    #s = scipy.misc.imresize(s,[80,128])
    y2, y1, x1, x2 = crop
    s = frame[y2:y1,x1:x2]
    s = scipy.misc.imresize(s,list(resize))
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

    
#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, fps=30):
    import moviepy.editor as mpy
    
    def make_frame(t):
        try:
            x = images[int(fps*t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=len(images)/fps)

    clip.write_gif(fname, fps=fps, verbose=False)