from utils import *
from memory import *
from models import *
from collections import namedtuple
from tensorboardX import SummaryWriter

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))

class Agent:
    
    def __init__(self, possible_actions, scenario, memory = 'uniform', max_size = 1000, stack_size = 4, batch_size = 64, resize = (120, 160)):
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
        resize           : tuple, shape of the resized frame (default=(120,160))
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
        self.resize = resize
        
    def get_reward(self, variables_cur, variables_prev):
        """
        Description
        --------------
        Reward reshaping
        
        Parameters
        --------------
        variables_cur  : dict, dictionnary containing current variables (kills, health and ammo.
        variables_prev : dict, dictionnary containing previous variables (kills, health and ammo.
        
        Returns
        --------------
        Float, a reshaped reward.
        """
        
        r = 0
        if self.scenario == 'defend_the_center':
            r += variables_cur['kills'] - variables_prev['kills']
            if variables_cur['ammo'] < variables_prev['ammo']:
                r -= 0.1

            if variables_cur['health'] < variables_prev['health']:
                r -= 0.1
                
        elif self.scenario == 'deadly_corridor':
            r += (variables_cur['kills'] - variables_prev['kills'])*5
            if variables_cur['ammo'] < variables_prev['ammo']:
                r -= 0.1

            if variables_cur['health'] < variables_prev['health']:
                r -= 1
                
        return r
        
            
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
        
        # Setting tensorboadX and variables of interest
        writer = SummaryWriter()
        kill_count = np.zeros(10) # This list will contain kill counts of each 10 episodes in order to compute moving average
        ammo = np.zeros(10) # This list will contain ammo of each 10 episodes in order to compute moving average
        rewards = np.zeros(10) # This list will contain rewards of each 10 episodes in order to compute moving average
        losses = np.zeros(10) # This list will contain losses of each 10 episodes in order to compute moving average
        # Pretraining phase
        game.new_episode()
        # Initialize current and previous game variables dictionnaries
        variables_cur = {'kills' : game.get_game_variable(KILLCOUNT), 'health' : game.get_game_variable(HEALTH), 
                        'ammo' : game.get_game_variable(AMMO2)}
        variables_prev = variables_cur.copy()
        # Get 1st state
        state = get_state(game)
        stacked_frames = deque([torch.zeros(self.resize, dtype=torch.int) for i in range(self.stack_size)], maxlen = self.stack_size)
        state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size, self.resize)
        for i in range(pretrain):
            # Get action and reward
            action = random.choice(self.possible_actions)
            reward = game.make_action(action, frame_skip)
            # Update the game vaiables dictionnaries and get the reshaped reward
            variables_cur['kills'] = game.get_game_variable(KILLCOUNT)
            variables_cur['health'] = game.get_game_variable(HEALTH)
            variables_cur['ammo'] = game.get_game_variable(AMMO2)
            reward += self.get_reward(variables_cur, variables_prev)
            variables_prev = variables_cur.copy()
            # Put reward and action in tensor form
            reward = torch.tensor([reward], dtype = torch.float)
            action = torch.tensor([action], dtype = torch.float)
            done = game.is_episode_finished()
            if done:
                # Set next state to zeros
                next_state = np.zeros((240, 320), dtype='uint8')[:, :, None] # (240, 320) is the screen resolution, see cfg files /scenarios
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                # Add experience to replay buffer
                self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                # Start a new episode
                game.new_episode()
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size, self.resize)

            else:
                # Get next state
                next_state = get_state(game)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                # Add experience to memory
                self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                # update state variable
                state = next_state
        
        # Exploration-Exploitation phase
        decay_step = 0
        if enhance == 'none':
            dqn_model = DQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            target_dqn_model = DQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            if use_cuda:
                print("End of trainig phase: The screen might be frozen now, don't worry, models take some time to be loaded on GPU")
                dqn_model.cuda()
                target_dqn_model.cuda()

        elif enhance == 'dueling':
            dqn_model = DDDQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            target_dqn_model = DDDQNetwork(init_zeros = init_zeros, out = len(self.possible_actions))
            if use_cuda:
                print("End of trainig phase: The screen might be frozen now, don't worry, models take some time to be loaded on GPU")
                dqn_model.cuda()
                target_dqn_model.cuda()

        optimizer = optim.Adam(dqn_model.parameters(), lr=lr)
        for episode in range(total_episodes):
            # When tau > max_tau perform double q-learning update.
            tau = 0
            episode_rewards = []
            game.new_episode()
            variables_cur = {'kills' : game.get_game_variable(KILLCOUNT), 'health' : game.get_game_variable(HEALTH), 
                            'ammo' : game.get_game_variable(AMMO2)}
            variables_prev = variables_cur.copy()
            # Get 1st state
            done = game.is_episode_finished()
            state = get_state(game)
            stacked_frames = deque([torch.zeros(self.resize, dtype=torch.int) for i in range(self.stack_size)], maxlen = self.stack_size)
            state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size, self.resize)
            while (not done):
                tau += 1
                decay_step += 1
                # Predict the action to take
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, dqn_model, self.possible_actions)
                # Perform the chosen action on frame_skip frames
                reward = game.make_action(action, frame_skip)
                # Update the game vaiables dictionnaries and get the reshaped reward
                variables_cur['kills'] = game.get_game_variable(KILLCOUNT)
                variables_cur['health'] = game.get_game_variable(HEALTH)
                variables_cur['ammo'] = game.get_game_variable(AMMO2)
                reward += self.get_reward(variables_cur, variables_prev)
                variables_prev = variables_cur.copy()
                # Check if the episode is done
                done = game.is_episode_finished()
                # Add the reward to total reward
                episode_rewards.append(reward)
                reward = torch.tensor([reward], dtype = torch.float)
                action = torch.tensor([action], dtype = torch.float)
                if done:
                    next_state = np.zeros((240, 320), dtype='uint8')[:, :, None]
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode),
                              'Total reward: {:.2f}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))
                    # Add experience to the replay buffer
                    self.memory.add((state, action, reward, next_state, torch.tensor([not done], dtype = torch.float)))
                    # Add number of kills and ammo variables
                    kill_count[episode%10] = game.get_game_variable(KILLCOUNT)
                    ammo[episode%10] = game.get_game_variable(AMMO2)
                    rewards[episode%10] = total_reward
                    losses[episode%10] = loss
                    # Update writer
                    if (episode > 0) and (episode%10 == 0):
                        writer.add_scalar('kills', kill_count.mean(), episode)
                        writer.add_scalar('ammo', ammo.mean(), episode)
                        writer.add_scalar('reward', rewards.mean(), episode)
                        writer.add_scalar('loss', losses.mean(), episode)

                else:
                    # Get the next state
                    next_state = get_state(game)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size, self.resize)
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
                    print('model updated')
                    tau = 0
                    
            if (episode % freq) == 0: # +1 just to avoid the conditon episode != 0
                model_file = 'weights/' + self.scenario + '/' + enhance + '_' + str(episode) + '.pth'
                torch.save(dqn_model.state_dict(), model_file)
                print('\nSaved model to ' + model_file)
        
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
                    
                    
                    