import argparse
from Agent import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train options')

    parser.add_argument('--scenario', type=str, default='basic', metavar='S', help="scenario to use, either basic or deadly_corridor")
    parser.add_argument('--window', type=int, default=0, metavar='WIN', help="0: don't render screen | 1: render screen")
#     parser.add_argument('--crop', type=int, default=0, metavar='CROP', help="Whether to crop screen or not")
#     parser.add_argument('--coords', type=tuple, default=(30, 300, 60, 180), metavar='COOR', help="Cropping coordinates")
    parser.add_argument('--resize', type=tuple, default=(120, 160), metavar='RES', help="Size of the resized frame")
    parser.add_argument('--stack_size', type=int, default=4, metavar='SS', help="Number of frames to stack to create motion")
    parser.add_argument('--explore_start', type=float, default=1., metavar='EI', help="Initial exploration probability")
    parser.add_argument('--explore_stop', type=float, default=0.01, metavar='EL', help="Final exploration probability")
    parser.add_argument('--decay_rate', type=float, default=1e-3, metavar='DR', help="Decay rate of exploration probability")
    parser.add_argument('--memory_size', type=int, default=1000, metavar='MS', help="Size of the experience replay buffer")
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS', help="Batch size")
    parser.add_argument('--gamma', type=float, default=.99, metavar='GAMMA', help="Discounting rate")
    parser.add_argument('--memory_type', type=str, default='uniform', metavar='MT', help="Uniform or prioritized replay buffer")
    parser.add_argument('--total_episodes', type=int, default=500, metavar='EPOCHS', help="Number of training episodes")
    parser.add_argument('--pretrain', type=int, default=100, metavar='PRE', help="number of initial experiences to put in the replay buffer")
    parser.add_argument('--frame_skip', type=int, default=4, metavar='FS', help="the number of frames to repeat the action on")
    parser.add_argument('--enhance', type=str, default='none', metavar='ENH', help="values : 'none', 'dueling'")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help="The learning rate")
    parser.add_argument('--max_tau', type=int, default=100, metavar='LR', help="Number of steps to performe double q-learning update")
    parser.add_argument('--freq', type=int, default=50, metavar='FQ', help="Number of episodes to save model weights")
    parser.add_argument('--init_zeros', type=int, default=0, metavar='FQ', help="1: Initialize weigts to 0")
    
    args = parser.parse_args()
    game, possible_actions = create_environment(scenario = args.scenario, window = args.window)
    agent = Agent(possible_actions, args.scenario, memory = args.memory_type, max_size = args.memory_size, stack_size = args.stack_size, 
                 batch_size = args.batch_size, resize = args.resize)
    agent.train(game, total_episodes = args.total_episodes, pretrain = args.pretrain, frame_skip = args.frame_skip, enhance = args.enhance, lr = args.lr, max_tau = args.max_tau, explore_start = args.explore_start, explore_stop = args.explore_stop, decay_rate = args.decay_rate, gamma = args.gamma, freq = args.freq, init_zeros = args.init_zeros)






























