from utils import *
from memory import *
from models import *
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))

class Agent:
    
    def __init__(self, possible_actions, scenario, memory = 'uniform', max_size = 1000, stack_size = 4, batch_size = 64):
        """
        Description
        --------------
        Constructor of Agent class.
        
        Attributes
        --------------
        possible_actions : List, contains the one-hot encoded possible actions to take for the agent.
        scenario         : String, either 'basic' or 'deadly_corridor'
        memory           : String with values in ['uniform', 'prioritized'] depending on the type of replay memory to be used     (defaul='uniform')
        max_size         : Int, maximum size of the replay buffer (default=1000)
        stack_size       : Int, the number of frames to stack to create motion (default=4)
        batch_size       : Int, the batch size used for backpropagation (default=64)
        """
        
        if memory == 'uniform':
            self.memory = MemoryUniform(max_size)
            
        elif memory == 'prioritized':
            self.memory = MemoryPrioritized(max_size)
            
        self.memory_type = memory
        self.stack_size = stack_size
        self.possible_actions = possible_actions
        self.scenario = scenario
        self.batch_size = batch_size
            
    def train(self, game, total_episodes = 100, pretrain = 100, frame_skip = 4, enhance = 'none', lr = 1e-4, max_tau = 100, 
                     explore_start = 1.0, explore_stop = 0.01, decay_rate = 0.0001, gamma = 0.99, freq = 50, init_zeros = False):
        """
        Description
        --------------
        Unroll trajectories to gather experiences and train the model.
        
        Parameters
        --------------
        game               : VizDoom game instance.
        total_episodes     : Int, the number of training episodes (default=100)
        pretrain           : Int, the number of initial experiences to put in the replay buffer (default=100)
        frame_skip         : Int, the number of frames to repeat the action on (default=4)
        enhance            : String in ['none', 'dueling'] (default='none')
        lr                 : Float, the learning rate (default=1e-4)
        max_tau            : Int, number of steps to performe double q-learning parameters update (default=100)
        explore_start      : Float, the initial exploration probaboility (default=1.0)
        explore_stop       : Float, the final exploration probability (default=0.01)
        decay_rate         : Float, the decay rate of the exploration probability (default=1e-3)
        gamma              : Float, the reward discoundting coefficient, should be between 0 and 1 (default=0.99)
        freq               : Int, number of episodes to save model weights (default=50)
        init_zeros         : Boolean, whether to initialize the weights to zero or not.
        """
        
        # Pretraining phase
        game.new_episode()
        state = get_state(game)
        stacked_frames = deque([torch.zeros((120, 160), dtype=torch.int) for i in range(self.stack_size)], maxlen = self.stack_size)
        state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size)
        for i in range(pretrain):
            action = random.choice(self.possible_actions)
            reward = game.make_action(action, frame_skip)
            reward = torch.tensor([reward], dtype = torch.float)
            action = torch.tensor([action], dtype = torch.float)
            done = game.is_episode_finished()
            if done:
                # Set next state to zeros
                next_state = np.zeros((240, 320), dtype='uint8')[:, :, None] # (240, 320) is the screen resolution, see cfg files /scenarios
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size)
                # Add experience to replay buffer
                self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                # Start a new episode
                game.new_episode()
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size)

            else:
                # Get next state
                next_state = get_state(game)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size)
                # Add experience to memory
                self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                # update state variable
                state = next_state
        
        # Exploration-Exploitation phase
        decay_step = 0
        if enhance == 'none':
            dqn_model = DQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            target_dqn_model = DQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            if torch.cuda.is_available():
                print("End of trainig phase: The screen might be frozen now, don't worry, models take some time to be loaded on GPU")
                dqn_model.cuda()
                target_dqn_model.cuda()

        elif enhance == 'dueling':
            dqn_model = DDDQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            target_dqn_model = DDDQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            if torch.cuda.is_available():
                print("End of trainig phase: The screen might be frozen now, don't worry, models take some time to be loaded on GPU")
                dqn_model.cuda()
                target_dqn_model.cuda()

        optimizer = optim.Adam(dqn_model.parameters(), lr=lr)
        for episode in range(total_episodes):
            # When tau > max_tau perform double q-learning update.
            tau = 0
            episode_rewards = []
            game.new_episode()
            done = game.is_episode_finished()
            state = get_state(game)
            stacked_frames = deque([torch.zeros((120, 160), dtype=torch.int) for i in range(4)], maxlen = 4)
            state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size)
            while (not done):
                tau += 1
                decay_step += 1
                # Predict the action to take
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, dqn_model, self.possible_actions)
                # Perform the chosen action on frame_skip frames
                reward = game.make_action(action, frame_skip)
                # Check if the episode is done
                done = game.is_episode_finished()
                # Add the reward to total reward
                episode_rewards.append(reward)
                reward = torch.tensor([reward], dtype = torch.float)
                action = torch.tensor([action], dtype = torch.float)
                if done:
                    next_state = np.zeros((240, 320), dtype='uint8')[:, :, None]
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size)
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))
                    # Add experience to the replay buffer
                    self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))

                else:
                    # Get the next state
                    next_state = get_state(game)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size)
                    # Add experience to memory
                    self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                    # Update state variable
                    state = next_state

                # Learning phase
                if self.memory_type == 'uniform':
                    transitions = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))
                    states_mb = torch.cat(batch.state)
                    actions_mb = torch.cat(batch.action)
                    rewards_mb = torch.cat(batch.reward)
                    next_states_mb = torch.cat(batch.next_state)
                    dones_mb = torch.cat(batch.dones)
                    if torch.cuda.is_available(): # Then use GPU device
                        next_states_mb = next_states_mb.cuda()
                        states_mb = states_mb.cuda()
                        q_next_state = dqn_model(next_states_mb).cpu()
                        q_target_next_state = target_dqn_model(next_states_mb).cpu()
                        q_state = dqn_model(states_mb).cpu()

                    else: # Then use CPU device
                        q_next_state = dqn_model.forward(next_states_mb)
                        q_target_next_state = target_dqn_model.forward(next_states_mb)
                        q_state = dqn_model.forward(states_mb)

                    targets_mb = rewards_mb + (gamma*dones_mb*torch.max(q_target_next_state, 1)[0])
                    output = (q_state * actions_mb).sum(1)
                    loss = F.mse_loss(output, targets_mb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if tau > max_tau:
                    # Update the parameters of our target_dqn_model with DQN_weights
                    update_target(dqn_model, target_dqn_model)
                    tau = 0
                    
            if ((episode + 1) % (freq + 1)) == 0: # +1 just to avoid the conditon episode != 0
                model_file = 'weights/' + self.scenario + '/' + enhance + '_' + str(episode) + '.pth'
                torch.save(dqn_model.state_dict(), model_file)
                print('\nSaved model to ' + model_file)
                    
                    
                    
                    
                    