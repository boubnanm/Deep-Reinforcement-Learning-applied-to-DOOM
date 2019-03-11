# Deep Reinforcement learning Applied to DOOM.

This is the final project for the Reinforcement Learning Course of the 2018/2019 MVA Master class.

This project is carried by [Mehdi Boubnan](https://github.com/Swirler) & [Ayman Chaouki](https://github.com/Chaoukia). It consists of training an agent to play in different scenarios of the game DOOM with deep reinforcement learning methods from Deep Q learning and its enhancements like double Q learning, deep recurrent network (with LSTM), deep dueling architecture and prioritized replay to Asynchronous Advantage Actor-Critic (A3C) and Curiosity-Driven learning. 

You can take a look at our paper **[Deep reinforcement learning applied to Doom](https://github.com/Swirler/Deep-Reinforcement-Learning-applied-to-DOOM/blob/master/Deep%20reinforcement%20learning%20applied%20to%20Doom.pdf)** for more details about the algorithms and some empirical results.
>>>>>>> e98fdc4744c016fbd7008c87560d66773cb70d83

Here are two examples of agents trained with A3C.
<p align="center">
<img align="center" src="A3C_Curiosity/gifs/deadly_corridor.gif"/>
<img align="center" src="A3C_Curiosity/gifs/defend_the_center.gif"/>
</p>

## Getting Started

### Prerequisites

- Operating system enabling the installation of VizDoom (there are some building problems with Ubuntu 16.04 for example), we use Ubuntu 18.04.
- NVIDIA GPU + CUDA and CuDNN (for optimal performance for deep Q learning methods).
- Python 3.6 (in order to install tensorflow).

### Installation

- Install VizDoom package according to your operating system, see https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md
- Install pytorch.
```
conda install pytorch torchvision -c pytorch
```
- Install tensorflow with GPU support, see https://www.tensorflow.org/install/pip
- Install tensorboard and tensorboardX.
```
pip install tensorboard
pip install tensorboardX
```
- Install moviepy.
```
pip install moviepy
```
- Clone this repo
```
git clone https://github.com/Swirler/Deep-Reinforcement-Learning-applied-to-DOOM
cd Deep-Reinforcement-Learning-applied-to-DOOM
```

## Deep Q Learning

```
cd "Deep Q Learning"
```

### Repositories

- scenarios : Configurations and .wad files of the following scenarios (basic, deadly corridor and defend the center).
- weights   : The weights of training each scenario will be saved here.

### Training

- You can view training rewards, game variables and loss plots by running ```tensorboard --logdir runs``` and clicking the URL http://localhost:6006
- Train a model with train.py , for example:

```
python train.py --scenario basic --window 1 --batch_size 32 --total_episodes 100 --lr 0.0001 --freq 20
```

### Testing

- The previous command saves training weights in weights/basic/ each 20 episodes. You can use the following command to view your agent playing:

```
python play.py --scenario basic --window 1 --weights weights/none_19.pth --total_episodes 20 --frame_skip 2
```

## A3C & Curiosity

```
cd "A3C_Curiosity"
```

### Repositories

- scenarios : Configurations and .wad files of the following scenarios (basic, deadly corridor, defend the center, defend the line and my way home).
- saves   : Models, tensorboad summaries and workers gifs during training will be saved here.

### Training

- You can view training rewards, game variables and loss plots by running ```python utils/launch_tensorboard.py```
- Train a model with main.py , for example:
    - Deadly corridor with default parameters : 
    ```
    python main.py --scenario deadly_corridor --actions all --num_workers 12 --max_episodes 1600
    ```
    - Basic with default parameters : 
    ```
    python main.py --scenario basic --actions single --num_workers 12 --max_episodes 1200
    ```
  
    - Deadly corridor with default parameters with curiosity: 
    ```
    python main.py --use_curiosity --scenario deadly_corridor --actions all --num_workers 12 --max_episodes 1600
    ```

See utils/args.py for more parameters.

### Testing

- You can use the following command to view your agent playing using the last trained model:

```
python main.py --play --scenario deadly_corridor --actions all --play_episodes 10
```


