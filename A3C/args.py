import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Doom A3C parameters')

    parser.add_argument("--scenario", type=str, choices=('basic', 'deadly_corridor', 'defend_the_center', 'defend_the_line', 'my_way_home'), 
                        default="deadly_corridor", help="Doom scenario")
    parser.add_argument("--actions", type=str, choices=('all','single'), 
                        default="single", help="Possible actions : 'all' for combinated actions and 'single' for single actions")
    parser.add_argument("--num_processes", type=int, default=-1, help="Number of processes for parallel algorithms  '-1' to use all available cpus")
    parser.add_argument("--model_path", type=str, default="./saves/model", help="Path to save models")
    parser.add_argument("--frames_path", type=str, default="./saves/frames", help="Path to save gifs")
    parser.add_argument("--summary_path", type=str, default="./saves/summary", help="Path to save training summary")
    parser.add_argument("--load_model", type=int, default=0, help="Either to load model or not")
    parser.add_argument("--max_episodes", type=int, default=1600, help="Maximum episodes per worker")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=30, help="Maximum steps per worker to update global network")
    parser.add_argument("--play",type=int, default=0, help="Launch agent to play")
    parser.add_argument("--play_episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--freq_model_save", type=int, default=200, help="Frequence of episodes to save model")
    parser.add_argument("--freq_gif_save", type=int, default=25, help="Frequence of episodes to save gifs")
    parser.add_argument("--freq_summary", type=int, default=5, help="Frequence of episodes to save gifs")

    game_args, _ = parser.parse_known_args()
    
    game_args.model_path += "/"+game_args.scenario
    game_args.frames_path += "/"+game_args.scenario
    game_args.summary_path += "/"+game_args.scenario
    
    return game_args
