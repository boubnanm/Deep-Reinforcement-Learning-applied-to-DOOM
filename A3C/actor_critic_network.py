import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from constants import constants

from configs import *
from utils import *

resize0, resize1 = resize

class AC_Network():
    
    def __init__(self, s_size, a_size, scope, trainer, as_player=False):
        """
        Description
        --------------
        Actor-Critic network.

        Parameters
        --------------
        s_size      : Int, dimension of the state space (width*height*channels).
        a_size      : Int, dimension of the action space.
        scope       : Int, the stride used in the conv layer (default=2)
        trainer     : tf.train, Tensorflow optimizer used for the module.
        as_player   : Bool, module used for training or playing.
        """
        
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,resize0,resize1,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn, num_outputs=16, kernel_size=[8,8], stride=[4,4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=32, kernel_size=[4,4], stride=[2,2], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,time_major=False)
            lstm_c, lstm_h = lstm_state          
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])    
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(inputs=rnn_out, 
                                               num_outputs=a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            
            self.value = slim.fully_connected(inputs=rnn_out,
                                              num_outputs=1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if (scope != 'global') and (not as_player):
                
                #Variables for loss functions
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
    
class StateActionPredictor(object):
    
    def __init__(self, ob_space, ac_space, scope, trainer, as_player=False):
        """
        Description
        --------------
        ICM Module.

        Parameters
        --------------
        ob_space    : Int, dimension of the state space (width*height*channels).
        ac_space    : Int, dimension of the action space.
        scope       : Int, the stride used in the conv layer (default=2)
        trainer     : tf.train, Tensorflow optimizer used for the module.
        as_player   : Bool, module used for training or playing.
        """
        
        with tf.variable_scope(scope):
            input_shape = [None,ob_space]
            self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
            self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
            self.aindex = aindex = tf.placeholder(shape=[None],dtype=tf.int32)
            self.asample = asample = tf.one_hot(self.aindex,ac_space,dtype=tf.float32)

            # Feature encoding: Encode states
            size = 256
            phi1 = self.state_encoder(phi1)
            phi2 = self.state_encoder(phi2)

            # Inverse model: Predict action from current and next states
            phi = tf.concat([phi1, phi2],1)
            phi = tf.nn.relu(linear(phi, size, "inv1", normalized_columns_initializer(0.01)))
            logits = linear(phi, ac_space, "invlast", normalized_columns_initializer(0.01))
            self.ainvprobs = tf.nn.softmax(logits, dim=-1)  
            self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
            

            # Forward model: Predict next state from current state and current action
            f = tf.concat([phi1, asample], 1)
            f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
            f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
            self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss') * phi1.get_shape()[1].value

            beta = constants['FORWARD_LOSS_WT']
            lr_pred = constants['PREDICTION_LR_SCALE']
            self.loss = lr_pred*(beta*self.forwardloss + (1-beta)*self.invloss)
            
            # Only the worker network need ops for loss functions and gradient updating.
            if (scope != 'global_P') and (not as_player):
                
                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_P')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

    def state_encoder(self, x):
        """
        Description
        --------------
        Encoder Module : encodes the state.

        Parameters
        --------------
        x    : Tensor, state.
        """
        imageIn = tf.reshape(x,shape=[-1,resize0,resize1,1])
        conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=imageIn,num_outputs=16, kernel_size=[8,8], stride=[4,4], padding='VALID')
        conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=conv1,num_outputs=32, kernel_size=[4,4],stride=[2,2], padding='VALID')
        encoding = slim.fully_connected(slim.flatten(conv2),256,activation_fn=tf.nn.elu)

        return encoding

    def pred_act(self, s1, s2):
        """
        Description
        --------------
        Returns action probabilities using the inverse model.

        Parameters
        --------------
        s1    : Tensor, current state.
        s2    : Tensor, next state.
        """
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        """
        Description
        --------------
        Returns the intrinsic reward using the forward model.

        Parameters
        --------------
        s1      : Tensor, current state.
        s2      : Tensor, next state.
        asample : Tensor, one hot encoding for the sampled action.
        """
        sess = tf.get_default_session()
        error = sess.run(self.forwardloss, {self.s1: [s1], self.s2: [s2], self.asample: [asample]}) * constants['PREDICTION_BETA']
        return error
    
    
    
# def doomHead(x):
#     ''' Learning by Prediction ICLR 2017 paper
#         (their final output was 64 changed to 256 here)
#         input: [None, 120, 160, 1]; output: [None, 1280] -> [None, 256];
#     '''
#     print('Using doom head design')
#     x = tf.nn.elu(conv2d(x, 8, "l1", [5, 5], [4, 4]))
#     x = tf.nn.elu(conv2d(x, 16, "l2", [3, 3], [2, 2]))
#     x = tf.nn.elu(conv2d(x, 32, "l3", [3, 3], [2, 2]))
#     x = tf.nn.elu(conv2d(x, 64, "l4", [3, 3], [2, 2]))
#     x = flatten(x)
#     x = tf.nn.elu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
#     return x


# def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
#     with tf.variable_scope(name):
#         stride_shape = [1, stride[0], stride[1], 1]
#         filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

#         # there are "num input feature maps * filter height * filter width"
#         # inputs to each hidden unit
#         fan_in = np.prod(filter_shape[:3])
#         # each unit in the lower layer receives a gradient from:
#         # "num output feature maps * filter height * filter width" /
#         #   pooling size
#         fan_out = np.prod(filter_shape[:2]) * num_filters
#         # initialize weights with random weights
#         w_bound = np.sqrt(6. / (fan_in + fan_out))

#         w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
#                             collections=collections)
#         b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
#                             collections=collections)
#         return tf.nn.conv2d(x, w, stride_shape, pad) + b


# def linear(x, size, name, initializer=None, bias_init=0):
#     w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
#     b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
#     return tf.matmul(x, w) + b