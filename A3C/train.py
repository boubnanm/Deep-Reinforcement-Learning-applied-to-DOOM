import os
import shutil

from agent import *
from networks import *

import threading
import multiprocessing

#import matplotlib.pyplot as plt
import tensorflow as tf
#from vizdoom import *

#from random import choice
from time import sleep
from time import time


def train_agents():
    tf.reset_default_graph()
    
    #Delete saves directory if not loading a model
    if not params.load_model:
        shutil.rmtree(params.model_path, ignore_errors=True)
        shutil.rmtree(params.frames_path, ignore_errors=True)
        shutil.rmtree(params.summary_path, ignore_errors=True)


    #Create a directory to save models to
    if not os.path.exists(params.model_path):
        os.makedirs(params.model_path)

    #Create a directory to save episode playback gifs to
    if not os.path.exists(params.frames_path):
        os.makedirs(params.frames_path)

    with tf.device("/cpu:0"): 
        
        # Generate global networks : Actor-Critic and ICM
        master_network = AC_Network(state_size, action_size, 'global') # Generate global AC network
        if params.use_curiosity:
            master_network_P = StateActionPredictor(state_size, action_size, 'global_P') # Generate global AC network
        
        # Set number of workers
        if params.num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = params.num_workers
        
        # Create worker classes
        workers = []
        for i in range(num_workers):
            trainer = tf.train.AdamOptimizer(learning_rate=params.lr)
            workers.append(Worker(i, state_size, action_size, trainer, params.model_path))
        saver = tf.train.Saver(max_to_keep=5)

        
    with tf.Session() as sess:
        
        # Loading pretrained model
        if params.load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # Starting initialized workers, each in a separate thread.
        coord = tf.train.Coordinator()
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(params.max_episodes,params.gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)