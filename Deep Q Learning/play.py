import argparse
from utils import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Playing options')
    
    parser.add_argument('--scenario', type=str, default='basic', metavar='S', help="scenario to use, either basic or deadly_corridor")
    parser.add_argument('--window', type=int, default=0, metavar='WIN', help="0: don't render screen | 1: render screen")
    parser.add_argument('--weights', type=str, metavar='S', help="Path to the weights we want to load")
    parser.add_argument('--total_episodes', type=int, default=500, metavar='EPOCHS', help="Number of training episodes")
    parser.add_argument('--enhance', type=str, default='none', metavar='ENH', help="values : 'none', 'dueling'")
    parser.add_argument('--frame_skip', type=int, default=4, metavar='FS', help="the number of frames to repeat the action on")
    
    args = parser.parse_args()
    
    test_environment(weights = args.weights, scenario = args.scenario, window = args.window, total_episodes = args.total_episodes, enhance = args.enhance, frame_skip = args.frame_skip)
    
    
































