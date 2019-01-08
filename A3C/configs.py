from args import parse_arguments

params = parse_arguments()

import numpy as np

if params.scenario == 'deadly_corridor':
    resize = (100,181)
    crop = (30,-35,1,-1)
    if params.actions=='all':
        action_size = 10
    else:
        action_size = 7
    s_size = np.prod(resize)
    
elif params.scenario == 'basic':
    resize = (84,84)
    crop = (10,-10,30,-30)
    if params.actions=='all':
        action_size = 6
    else:
        action_size = 3
    s_size = np.prod(resize)

        
elif params.scenario == 'defend_the_center':
    resize = (84,159)
    crop = (40,-32,1,-1)
    if params.actions=='all':
        action_size = 6
    else:
        action_size = 3
    s_size = np.prod(resize)