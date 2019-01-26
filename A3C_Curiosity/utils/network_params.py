from utils.args import parse_arguments

params = parse_arguments()

import numpy as np

if params.scenario == 'deadly_corridor':
    resize = (100,181)
    crop = (30,-35,1,-1)
    
    if params.actions=='all':
        action_size = 10
    elif params.actions=='single':
        action_size = 6
    
    state_size = np.prod(resize)
    
elif params.scenario == 'basic':
    resize = (84,84)
    crop = (10,-10,30,-30)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)

        
elif params.scenario == 'defend_the_center':
    resize = (84,159)
    crop = (40,-32,1,-1)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    
elif params.scenario == 'defend_the_line':
    resize = (84,159)
    crop = (40,-32,1,-1)
    
    if params.actions=='all':
        action_size = 6
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    
elif params.scenario == 'my_way_home':
    resize = (84,112)
    crop = (1,-1,1,-1)
    
    if params.actions=='all':
        action_size = 5
    elif params.actions=='single':
        action_size = 3
    
    state_size = np.prod(resize)
    

# ICM Module parameters

beta = 0.2
lr_pred = 10.0
pred_bonus_coef = 0.01

#constants = {
#            'GAMMA': 0.99,  # discount factor for rewards
#            'LAMBDA': 1.0,  # lambda of Generalized Advantage Estimation: https://arxiv.org/abs/1506.02438
#            'ENTROPY_BETA': 0.01,  # entropy regurarlization constant.
#            'ROLLOUT_MAXLEN': 20, # 20 represents the number of 'local steps': the number of timesteps
#                                # we run the policy before we update the parameters.
#                                # The larger local steps is, the lower is the variance in our policy gradients estimate
#                                # on the one hand;  but on the other hand, we get less frequent parameter updates, which
#                                # slows down learning.  In this code, we found that making local steps be much
#                                # smaller than 20 makes the algorithm more difficult to tune and to get to work.
#            'GRAD_NORM_CLIP': 40.0,   # gradient norm clipping
#            'REWARD_CLIP': 1.0,       # reward value clipping in [-x,x]
#            'MAX_GLOBAL_STEPS': 100000000,  # total steps taken across all workers
#            'LEARNING_RATE': 1e-4,  # learning rate for adam
#
#            'PREDICTION_BETA': 0.01,  # weight of prediction bonus
#                                      # set 0.5 for unsup=state
#            'PREDICTION_LR_SCALE': 10.0,  # scale lr of predictor wrt to policy network
#                                          # set 30-50 for unsup=state
#            'FORWARD_LOSS_WT': 0.2,  # should be between [0,1]
#                                      # predloss = ( (1-FORWARD_LOSS_WT) * inv_loss + FORWARD_LOSS_WT * forward_loss) * PREDICTION_LR_SCALE
#            'POLICY_NO_BACKPROP_STEPS': 0,  # number of global steps after which we start backpropagating to policy
#            }

