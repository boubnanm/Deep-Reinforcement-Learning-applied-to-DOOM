import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from constants import constants

from configs import *
from utils import *

resize0, resize1 = resize

class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer, as_player=False):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,resize0,resize1,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn,num_outputs=16, kernel_size=[8,8], stride=[4,4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1,num_outputs=32, kernel_size=[4,4],stride=[2,2], padding='VALID')
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
            self.policy = slim.fully_connected(rnn_out,
                                               a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,
                                              1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if (scope != 'global') and (not as_player):
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

def doomHead(x):
    ''' Learning by Prediction ICLR 2017 paper
        (their final output was 64 changed to 256 here)
        input: [None, 120, 160, 1]; output: [None, 1280] -> [None, 256];
    '''
    print('Using doom head design')
    x = tf.nn.elu(conv2d(x, 8, "l1", [5, 5], [4, 4]))
    x = tf.nn.elu(conv2d(x, 16, "l2", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 32, "l3", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "l4", [3, 3], [2, 2]))
    x = flatten(x)
    x = tf.nn.elu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def doomHead2(x):#,scope):
#     with tf.variable_scope(scope):
#         inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
    imageIn = tf.reshape(x,shape=[-1,resize0,resize1,1])
    conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=imageIn,num_outputs=16, kernel_size=[8,8], stride=[4,4], padding='VALID')
    conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=conv1,num_outputs=32, kernel_size=[4,4],stride=[2,2], padding='VALID')
    hidden = slim.fully_connected(slim.flatten(conv2),256,activation_fn=tf.nn.elu)
        
    return hidden

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b
    
class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space, scope, trainer, as_player=False):
        with tf.variable_scope(scope):
            # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
            # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
            input_shape = [None,ob_space]# + list(ob_space)
            self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
            self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
#             self.asample = asample = tf.placeholder(tf.float32, [None, ac_space]) 
            self.aindex = aindex = tf.placeholder(shape=[None],dtype=tf.int32)
#             self.actions_onehot = tf.one_hot(self.aindex,ac_space,dtype=tf.float32)
            self.asample = asample = tf.one_hot(self.aindex,ac_space,dtype=tf.float32)

    
            # feature encoding: phi1, phi2: [None, LEN]
            size = 256
            phi1 = doomHead2(phi1)#doomHead(phi1)
#             with tf.variable_scope(scope, reuse=True):
            phi2 = doomHead2(phi2)#doomHead(phi2)

            # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
            g = tf.concat([phi1, phi2],1)
            g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
            logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
            self.ainvprobs = tf.nn.softmax(logits, dim=-1)
            
#             aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
            
            self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=aindex), name="invloss")
            

            # forward model: f(phi1,asample) -> phi2
            # Note: no backprop to asample of policy: it is treated as fixed for predictor training
            f = tf.concat([phi1, asample], 1)
            f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
            f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
            self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
            # self.forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
            # self.forwardloss = cosineLoss(f, phi2, name='forwardloss')
            self.forwardloss = self.forwardloss * phi1.get_shape()[1].value  # lenFeatures=288. Factored out to make hyperparams not depend on it.
            
            beta = constants['FORWARD_LOSS_WT']
            lr_pred = constants['PREDICTION_LR_SCALE']
            self.loss = lr_pred*(beta*self.forwardloss + (1-beta)*self.invloss)

            # variable list
            if (scope != 'global_P') and (not as_player):
                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_P')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


    def pred_act(self, s1, s2):
        '''
        returns action probability distribution predicted by inverse model
            input: s1,s2: [h, w, ch]
            output: ainvprobs: [ac_space]
        '''
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        # error = sess.run([self.forwardloss, self.invloss],
        #     {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error[0], ' ErrorI:', error[1])
        error = sess.run(self.forwardloss,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        error = error * constants['PREDICTION_BETA']
        return error
    
    
class StateActionPredictor_origin(object):
    def __init__(self, ob_space, ac_space, scope, designHead='doom'):
        with tf.variable_scope(scope):
            # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
            # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
            input_shape = [None] + list(ob_space)
            self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
            self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
            self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

            # feature encoding: phi1, phi2: [None, LEN]
            size = 256
            if designHead == 'doom':
                phi1 = doomHead(phi1)
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2 = doomHead(phi2)

            # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
            g = tf.concat(1,[phi1, phi2])
            g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
            aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
            logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
            self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits, aindex), name="invloss")
            self.ainvprobs = tf.nn.softmax(logits, dim=-1)

            # forward model: f(phi1,asample) -> phi2
            # Note: no backprop to asample of policy: it is treated as fixed for predictor training
            f = tf.concat(1, [phi1, asample])
            f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
            f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
            self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
            # self.forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
            # self.forwardloss = cosineLoss(f, phi2, name='forwardloss')
            self.forwardloss = self.forwardloss * 288.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.

            # variable list
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_act(self, s1, s2):
        '''
        returns action probability distribution predicted by inverse model
            input: s1,s2: [h, w, ch]
            output: ainvprobs: [ac_space]
        '''
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        # error = sess.run([self.forwardloss, self.invloss],
        #     {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error[0], ' ErrorI:', error[1])
        error = sess.run(self.forwardloss,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        error = error * constants['PREDICTION_BETA']
        return error
    
    
class StatePredictor(object):
    '''
    Loss is normalized across spatial dimension (42x42), but not across batches.
    It is unlike ICM where no normalization is there across 288 spatial dimension
    and neither across batches.
    '''

    def __init__(self, ob_space, ac_space, designHead='universe', unsupType='state'):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])
        self.stateAenc = unsupType == 'stateAenc'

        # feature encoding: phi1: [None, LEN]
        if designHead == 'universe':
            phi1 = universeHead(phi1)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        elif 'tile' in designHead:  # for mario tiles
            phi1 = universeHead(phi1, nConvs=2)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        else:
            print('Only universe designHead implemented for state prediction baseline.')
            exit(1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, phi1.get_shape()[1].value, "f1", normalized_columns_initializer(0.01)))
        if 'tile' in designHead:
            f = inverseUniverseHead(f, input_shape, nConvs=2)
        else:
            f = inverseUniverseHead(f, input_shape)
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        if self.stateAenc:
            self.aencBonus = 0.5 * tf.reduce_mean(tf.square(tf.subtract(phi1, phi2_aenc)), name='aencBonus')
        self.predstate = phi1

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_state(self, s1, asample):
        '''
        returns state predicted by forward model
            input: s1: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: s2: [h, w, ch]
        '''
        sess = tf.get_default_session()
        return sess.run(self.predstate, {self.s1: [s1],
                                            self.asample: [asample]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        bonus = self.aencBonus if self.stateAenc else self.forwardloss
        error = sess.run(bonus,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error)
        error = error * constants['PREDICTION_BETA']
        return error