import numpy as np
import tensorflow as tf
import scipy
import scipy.signal
import random
import scipy.misc
import csv
import tensorflow.contrib.slim as slim

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

#The Below code is related to setting up the Doom environment
        game.load_config("scenarios/"+params.scenario+".cfg")
        game.set_doom_scenario_path("scenarios/"+params.scenario+".wad")
        if params.no_render:
            game.set_window_visible(False)
        if as_player:
            game.set_screen_format(ScreenFormat.RGB24)

        game.init()
        if params.actions=='all' :
            self.actions = self.button_combinations()
        else :
            self.actions = np.identity(action_size,dtype=bool).tolist()
        #End Doom set-up
        self.env = game


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
    game.load_config("scenarios/"+params.scenario+".cfg")
    game.set_doom_scenario_path("scenarios/"+params.scenario+".wad")
    
    # Switch to RGB in player mode
    if player_mode:
        game.set_screen_format(ScreenFormat.RGB24)
    
    # Initiliaze game
    game.init()
    
    # Possible predefined actions for the scenario
    possible_action = button_combinations(scenario)
    
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
    
#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS)/duration*t)]
        except:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps = len(images) / duration,verbose=False)
        #clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps = len(images) / duration,verbose=False)

        
#Record performance metrics and episode logs for the Control Center.
#def saveToCenter(i,rList,jList,bufferArray,summaryLength,h_size,sess,mainQN,time_per_step):
#    with open('./Center/log.csv', 'a') as myfile:
#        state_display = (np.zeros([1,h_size]),np.zeros([1,h_size]))
#        imagesS = []
#        for idx,z in enumerate(np.vstack(bufferArray[:,0])):
#            img,state_display = sess.run([mainQN.salience,mainQN.rnn_state],\
#                feed_dict={mainQN.scalarInput:np.reshape(bufferArray[idx,0],[1,21168])/255.0,\
#                mainQN.trainLength:1,mainQN.state_in:state_display,mainQN.batch_size:1})
#            imagesS.append(img)
#        imagesS = (imagesS - np.min(imagesS))/(np.max(imagesS) - np.min(imagesS))
#        imagesS = np.vstack(imagesS)
#        imagesS = np.resize(imagesS,[len(imagesS),84,84,3])
#        luminance = np.max(imagesS,3)
#        imagesS = np.multiply(np.ones([len(imagesS),84,84,3]),np.reshape(luminance,[len(imagesS),84,84,1]))
#        make_gif(np.ones([len(imagesS),84,84,3]),'./Center/frames/sal'+str(i)+'.gif',duration=len(imagesS)*time_per_step,true_image=False,salience=True,salIMGS=luminance)
#
#        images = zip(bufferArray[:,0])
#        images.append(bufferArray[-1,3])
#        images = np.vstack(images)
#        images = np.resize(images,[len(images),84,84,3])
#        make_gif(images,'./Center/frames/image'+str(i)+'.gif',duration=len(images)*time_per_step,true_image=True,salience=False)
#
#        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#        wr.writerow([i,np.mean(jList[-100:]),np.mean(rList[-summaryLength:]),'./frames/image'+str(i)+'.gif','./frames/log'+str(i)+'.csv','./frames/sal'+str(i)+'.gif'])
#        myfile.close()
#    with open('./Center/frames/log'+str(i)+'.csv','w') as myfile:
#        state_train = (np.zeros([1,h_size]),np.zeros([1,h_size]))
#        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#        wr.writerow(["ACTION","REWARD","A0","A1",'A2','A3','V'])
#        a, v = sess.run([mainQN.Advantage,mainQN.Value],\
#            feed_dict={mainQN.scalarInput:np.vstack(bufferArray[:,0])/255.0,mainQN.trainLength:len(bufferArray),mainQN.state_in:state_train,mainQN.batch_size:1})
#        wr.writerows(zip(bufferArray[:,1],bufferArray[:,2],a[:,0],a[:,1],a[:,2],a[:,3],v[:,0])