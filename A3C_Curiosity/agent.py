import tensorflow as tf
import numpy as np
import time
import os
from PIL import Image

from utils.utils import *
from utils.network_params import *
from utils.networks import *

class Worker():
    
    def __init__(self, name, s_size, a_size, trainer=None, model_path=None, player_mode=False):
        """
        Description
        --------------
        Agent for A3C and curiosity models.

        Parameters
        --------------
        game            : Env, environment used for the training.
        name            : str, name of the worker.
        s_size          : Int, dimension of state space (width*height*channels).
        a_size          : Int, dimension of action space.
        trainer         : tf.train, Tensorflow optimizer used for the module.
        model_path      : str, path for the model to load if any.
        global_episodes : tf.Variable, episode counter for all workers.
        player_mode     : Bool, worker used in playing mode.
        """
        
        self.name = "worker_" + str(name)
        self.number = name        
        
        if not player_mode:
            self.model_path = model_path
            self.trainer = trainer
            self.initialize_containers()
            self.summary_writer = tf.summary.FileWriter(params.summary_path+"/train_"+str(self.number))
        
        #The Below code is related to setting up the Doom environment  
        self.env, self.actions = create_environment(scenario=params.scenario, no_window=params.no_render, 
                                                    actions_type=params.actions, player_mode=player_mode)

        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer, player_mode) 
        self.update_local_ops = update_target_graph('global', self.name)
        
        if params.use_curiosity:
            self.local_Pred = StateActionPredictor(s_size, a_size, self.name+"_P", trainer, player_mode)
            self.update_local_ops_P = update_target_graph('global_P',self.name+"_P")  

    
    def initialiaze_game_vars(self):
        """
        Description
        --------------
        Initialize game variables used for reward reshaping.
        
        """
        self.last_total_health = 100.0
        self.last_total_ammo2 = 52  
        self.last_total_kills = 0
    
    def initialize_containers(self):
        """
        Description
        --------------
        Initialize episode containers used for tensorboard summary.
        
        """
        self.episode_rewards = []
        self.episode_curiosities = []
        self.episode_lengths = []
        self.episode_mean_values = []
        
        if params.scenario=='deadly_corridor':
            self.episode_kills = []
            self.episode_health = []
            self.episode_ammo = []
        
        if params.scenario=='basic':
            self.episode_ammo = []
        
        if params.scenario=='defend_the_center':
            self.episode_ammo = []
            self.episode_kills = []
    
    def update_containers(self):
        """
        Description
        --------------
        Update episode containers used for tensorboard summary.
        
        """
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_step_count)
        self.episode_mean_values.append(np.mean(self.episode_values))
        
        if params.use_curiosity:
            self.episode_curiosities.append(self.episode_curiosity)
        
        if params.scenario=='deadly_corridor':
            self.episode_kills.append(self.last_total_kills)
            self.episode_health.append(np.maximum(0,self.last_total_health))    
            self.episode_ammo.append(self.last_total_ammo2)
        
        if params.scenario=='basic':
            self.episode_ammo.append(self.last_total_ammo2)
        
        if params.scenario=='defend_the_center':
            self.episode_ammo.append(self.last_total_ammo2)
            self.episode_kills.append(self.last_total_kills)
            
    def update_summary(self):
        """
        Description
        --------------
        Update tensorboard summary using episode containers.
        
        """
        mean_reward = np.mean(self.episode_rewards[-params.freq_summary:])
        mean_length = np.mean(self.episode_lengths[-params.freq_summary:])
        mean_value = np.mean(self.episode_mean_values[-params.freq_summary:])
        
        if params.use_curiosity:
            mean_curiosity = np.mean(self.episode_curiosities[-params.freq_summary:])
                    
        summary = tf.Summary()
        summary.value.add(tag='Perf/Episode_Length', simple_value=float(mean_length))
        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
        if params.use_curiosity:
            summary.value.add(tag='Perf/Curiosity', simple_value=float(mean_curiosity))
        
        if params.scenario=='deadly_corridor':
            mean_kills = np.mean(self.episode_kills[-params.freq_summary:])
            mean_health = np.mean(self.episode_health[-params.freq_summary:])
            mean_ammo = np.mean(self.episode_ammo[-params.freq_summary:])
            summary.value.add(tag='Perf/Kills', simple_value=float(mean_kills))
            summary.value.add(tag='Perf/Health', simple_value=float(mean_health))
            summary.value.add(tag='Perf/Ammo', simple_value=float(mean_ammo))
        
        if params.scenario=='basic':
            mean_ammo = np.mean(self.episode_ammo[-params.freq_summary:])
            summary.value.add(tag='Perf/Ammo', simple_value=float(mean_ammo))
        
        if params.scenario=='defend_the_center':
            mean_ammo = np.mean(self.episode_ammo[-params.freq_summary:])
            mean_kills = np.mean(self.episode_kills[-params.freq_summary:])
            summary.value.add(tag='Perf/Ammo', simple_value=float(mean_ammo))
            summary.value.add(tag='Perf/Kills', simple_value=float(mean_kills))
        
        summary.value.add(tag='Losses/Value Loss', simple_value=float(self.v_l))
        summary.value.add(tag='Losses/Policy Loss', simple_value=float(self.p_l))
        summary.value.add(tag='Losses/Entropy', simple_value=float(self.e_l))
        summary.value.add(tag='Losses/Grad Norm', simple_value=float(self.g_n))
        summary.value.add(tag='Losses/Var Norm', simple_value=float(self.v_n))
        if params.use_curiosity:
            summary.value.add(tag='Losses/Inverse Loss', simple_value=float(self.Inv_l))
            summary.value.add(tag='Losses/Forward Loss', simple_value=float(self.Forward_l))
            
        
        self.summary_writer.add_summary(summary, self.episode_count)
        self.summary_writer.flush()
        
        
    def print_end_episode_perfs(self):
        """
        Description
        --------------
        Print episode statistics depending on the scenario.
        
        """
        if params.scenario=='deadly_corridor':
            print('{}, health: {}, kills:{}, episode #{}, ep_reward: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, np.maximum(0,self.last_total_health), self.last_total_kills, self.episode_count,
                            self.episode_reward, self.episode_step_count, self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
            
        if params.scenario=='basic':
            print('{}, episode #{}, ep_reward: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, self.episode_count, self.episode_reward, self.episode_step_count, 
                            self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
        
        if params.scenario=='defend_the_center':
            print('{}, kills:{}, episode #{}, ep_reward: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, self.last_total_kills, self.episode_count, self.episode_reward, self.episode_step_count, 
                            self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
                          
        if params.scenario=='my_way_home':
            print('{}, episode #{}, ep_reward: {}, ep_curiosity: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, self.episode_count, self.episode_reward, self.episode_curiosity, self.episode_step_count, 
                            self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
             
    
    def get_health_reward(self):
        """
        Description
        --------------
        Health reward.
        
        """
        d_health = self.env.get_game_variable(GameVariable.HEALTH) - self.last_total_health
        self.last_total_health = self.env.get_game_variable(GameVariable.HEALTH)
        if d_health == 0:
            return 0
        elif d_health < 0:
            return -5
    
    def get_ammo_reward(self):
        """
        Description
        --------------
        Ammo reward.
        
        """
        d_ammo = self.env.get_game_variable(GameVariable.AMMO2) - self.last_total_ammo2
        self.last_total_ammo2 = self.env.get_game_variable(GameVariable.AMMO2)
        if d_ammo == 0:
            return 0
        elif d_ammo > 0:
            return d_ammo * 0.5
        else:
            return -d_ammo * 0.5
        
    def get_kill_reward(self):
        """
        Description
        --------------
        Kill reward.
        
        """
        d_kill = self.env.get_game_variable(GameVariable.KILLCOUNT) - self.last_total_kills
        self.last_total_kills = self.env.get_game_variable(GameVariable.KILLCOUNT)
        if d_kill > 0 :
            return d_kill*100
        return 0
    
    def get_custom_reward(self,game_reward):
        """
        Description
        --------------
        Final reward reshaped.
        
        Parameters
        --------------
        game_reward : float, reward provided by the environment
        """
        if params.scenario=='basic':
            return game_reward/100.0
        
        if params.scenario=='defend_the_center':
            self.last_total_kills = self.env.get_game_variable(GameVariable.KILLCOUNT)
            return game_reward + self.get_ammo_reward()/10
        
        if params.scenario=='deadly_corridor':
            return (game_reward/5 + self.get_health_reward() + self.get_kill_reward() + self.get_ammo_reward())/100.
        
        if params.scenario=='my_way_home':
            return game_reward
        
        else:
            return game_reward
    
    def choose_action_index(self, policy, deterministic=False):
        """
        Description
        --------------
        Choose action from stochastic policy.
        
        Parameters
        --------------
        policy        : np.array, actions probabilities
        deterministic : boolean, whether to 
        """
        if deterministic:
            return np.argmax(policy[0])
        else:
            return np.argmax(policy == np.random.choice(policy[0],p=policy[0]))
    
    
    def train(self,rollout,sess,gamma,bootstrap_value):
        """
        Description
        --------------
        Unroll trajectories to train the model.
        
        Parameters
        --------------
        rollout            : list, buffer containing experiences.
        sess               : Tensorflow session
        gamma              : Float, discount factor
        bootstrap_value    : Float, bootstraped value function if episode is not finished
        """
        rollout = np.array(rollout)
        observations, actions, rewards, next_observations, _, values = rollout.T
        
        # Process the rollout by constructing variables for the loss functions
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = discounted_rewards - self.value_plus[:-1]

        # Update the local Actor-Critic network using gradients from loss
        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     self.local_AC.inputs:np.vstack(observations),
                     self.local_AC.actions:actions,
                     self.local_AC.advantages:advantages,
                     self.local_AC.state_in[0]:self.batch_rnn_state[0],
                     self.local_AC.state_in[1]:self.batch_rnn_state[1]}
        
        self.v_l,self.p_l,self.e_l,self.g_n,self.v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
                                                                                         self.local_AC.policy_loss,
                                                                                         self.local_AC.entropy,
                                                                                         self.local_AC.grad_norms,
                                                                                         self.local_AC.var_norms,
                                                                                         self.local_AC.state_out,
                                                                                         self.local_AC.apply_grads],
                                                                                        feed_dict=feed_dict)
        Losses = [self.v_l,self.p_l,self.e_l]
        Grad_vars = [self.g_n,self.v_n]
        
        # Update the local ICM network using gradients from loss
        if params.use_curiosity:
            feed_dict_P = {self.local_Pred.s1:np.vstack(observations),
                           self.local_Pred.s2:np.vstack(next_observations),
                           self.local_Pred.aindex:actions}        
            
            self.Inv_l, self.Forward_l, _ = sess.run([self.local_Pred.invloss,
                                                      self.local_Pred.forwardloss,
                                                      self.local_Pred.apply_grads],
                                                     feed_dict=feed_dict_P)
            
            Losses += [self.Inv_l,self.Forward_l]
            
        return list(np.array(Losses)/len(rollout))+Grad_vars
        
    def work(self,max_episodes,gamma,sess,coord,saver):
        """
        Description
        --------------
        Unroll trajectories to train the model.
        
        Parameters
        --------------
        max_episodes    : Int, maximum episodes for worker.
        gamma           : Float, discount factor
        sess            : Tensorflow session
        coord           : Tensorflow coordinator for threading
        saver           : Tensorflow saver
        """
    
        print ("Starting worker " + str(self.number))
        self.episode_count = 0
        total_steps = 0
        
        with sess.as_default(), sess.graph.as_default():                 
            while (not coord.should_stop()) and (self.episode_count<max_episodes):
                # Copy the global networks weights to local network weights
                sess.run(self.update_local_ops)
                if params.use_curiosity: sess.run(self.update_local_ops_P)
                
                # Initialize buffer for training
                episode_buffer = []
                
                # Initialize frames buffer to save gifs
                episode_frames = []
                
                # Initialize variables to record performance for tensorflow summary
                self.episode_values = []        
                self.episode_reward = 0
                self.episode_curiosity = 0
                self.episode_step_count = 0
                
                # Initialize game vars (health, kills ...)
                self.initialiaze_game_vars()
                
                # Begin new episode
                d = False
                self.env.new_episode()
                self.episode_st = time.time()
                
                # Initialize LSTM gates
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                
                # Get first state and process it
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s, crop, resize)
                
                while self.env.is_episode_finished() == False:
                    
                    # Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                                                  feed_dict={self.local_AC.inputs:[s],
                                                             self.local_AC.state_in[0]:rnn_state[0],
                                                             self.local_AC.state_in[1]:rnn_state[1]})
                     
                    action_index = self.choose_action_index(a_dist, deterministic=False)
                    
                    # Get extrinsic reward
                    if params.no_reward: reward = 0
                    else: reward = self.get_custom_reward(self.env.make_action(self.actions[action_index], 2))
                        
                    # Check if episode is finished to process next state
                    done = self.env.is_episode_finished()
                    if done == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1, crop, resize)
                    else:
                        s1 = s
                    
                    # Get intrinsic reward
                    if params.use_curiosity:
                        curiosity = np.clip(self.local_Pred.pred_bonus(s,s1,a_dist[0]),-1,1)/5
                        self.episode_curiosity += curiosity
                    else:
                        curiosity = 0
                    
                    # Total reward
                    r = curiosity + reward
                    
                    # Append step to buffer
                    episode_buffer.append([s,action_index,r,s1,d,v[0,0]])

                    # Update variables
                    self.episode_values.append(v[0,0])
                    self.episode_reward += r
                    s = s1                    
                    total_steps += 1
                    self.episode_step_count += 1
                    
                    # If the episode hasn't ended, but maximum steps is reached, we update the global network using the current rollout.
                    if len(episode_buffer) == params.n_steps and done != True and self.episode_step_count != max_episodes - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs:[s],
                                                 self.local_AC.state_in[0]:rnn_state[0],
                                                 self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        
                        Losses_grads = self.train(episode_buffer, sess, gamma, v1)
                        
                        if params.use_curiosity:
                            self.v_l,self.p_l,self.e_l,self.Inv_l,self.Forward_l,self.g_n,self.v_n = Losses_grads  
                        else :
                            self.v_l,self.p_l,self.e_l,self.g_n,self.v_n = Losses_grads
                            
                        # Empty buffer
                        episode_buffer = []
                        
                        # Copy the global network weights to the local network
                        sess.run(self.update_local_ops)
                        if params.use_curiosity: sess.run(self.update_local_ops_P)
                    
                    if done == True:
                        # Print perfs of episode
                        self.print_end_episode_perfs()
                        break
                
                # Update containers for tensorboard summary
                self.update_containers()
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    if params.use_curiosity:
                        self.v_l,self.p_l,self.e_l,self.Inv_l,self.Forward_l,self.g_n,self.v_n = self.train(episode_buffer,sess,gamma,0.0)
                    else :
                        self.v_l,self.p_l,self.e_l,self.g_n,self.v_n = self.train(episode_buffer,sess,gamma,0.0)            
                                             
                # Periodically save gifs of episodes              
                if self.name == 'worker_0' and self.episode_count % params.freq_gif_save == 0 and self.episode_count != 0:
                    time_per_step = 0.05
                    images = np.array(episode_frames)
                    gif_path = os.path.join(params.frames_path,'image'+str(self.episode_count)+'.gif')
                    make_gif(images, gif_path)
                
                # Periodically save model parameters
                if self.episode_count % params.freq_model_save == 0 and self.name == 'worker_0' and self.episode_count != 0:
                    saver.save(sess,self.model_path+'/model-'+str(self.episode_count)+'.cptk')
                    print ("Saved Model")
                
                # Periodically save summary statistics
                if self.episode_count % params.freq_summary == 0 and self.episode_count != 0:
                    self.update_summary()
                
                self.episode_count += 1
                
            print("{} finished {} episodes with {} steps. Going to sleep ..".format(self.name, self.episode_count, total_steps))


    def play_game(self, sess, episode_num):
        if not isinstance(sess, tf.Session):
            raise TypeError('Saver should be tf.train.Saver')
        
        print("Playing",episode_num,"episodes..")
        
        for i in range(episode_num):
            print("Launching episode",i+1)
            time.sleep(3)
            
            # Initialize frames buffer to save gifs
            episode_frames = []
            
            # Begin episode
            self.env.new_episode()
            
            # Initialize variables and LSTM gates
            rnn_state = self.local_AC.state_init
            episode_rewards = 0
            last_total_shaping_reward = 0
            step = 0
            s_t = time.time()

            while not self.env.is_episode_finished():
                # Get state
                state = self.env.get_state().screen_buffer
                episode_frames.append(state)
                s = process_frame(np.array(Image.fromarray(state).convert("L")), crop, resize)
                
                # Get action from policy module
                a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value,self.local_AC.state_out],
                                                feed_dict={self.local_AC.inputs: [s],
                                                           self.local_AC.state_in[0]:rnn_state[0],
                                                           self.local_AC.state_in[1]:rnn_state[1]})
                
                a_index = self.choose_action_index(a_dist, deterministic=True)
                
                # Make action and get reward
                reward = self.env.make_action(self.actions[a_index])
                
                step += 1
                episode_rewards += reward
                
                print('Current step: #{}, Current health: {}, Current kills: {},\
                       Current reward: {}'.format(step, self.env.get_game_variable(GameVariable.HEALTH), \
                                                  self.env.get_game_variable(GameVariable.KILLCOUNT), reward))

            print('End episode: {}, Total Reward: {}, {}'.format(i, episode_rewards, last_total_shaping_reward))
            print('Time cost: {}'.format(time.time() - s_t))
            
            # Periodically save gif
            if (i+1)%5==0 or i==0:
                print("Saving episode GIF..")
                images = np.array(episode_frames)
                gif_file = os.path.join(params.gif_path,params.scenario+"_"+str(i+1)+".gif")
                make_gif(images, gif_file)
                print("Done")
            