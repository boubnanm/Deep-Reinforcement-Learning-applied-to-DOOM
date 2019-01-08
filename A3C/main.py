import tensorflow as tf
from configs import *

from play import play_with_agent
from train import train_agents


if __name__ == '__main__':

    if params.play:
        play_with_agent(params)
    else:
        train_agents()