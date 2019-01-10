from vizdoom import *
import tensorflow as tf
import numpy as np
from random import choice
import time

from utils import *
from configs import *

from actor_critic_network import *

class Worker():
    def __init__(self, game, name, s_size, action_size, trainer=None, model_path=None, global_episodes=None, as_player=False):
        self.name = "worker_" + str(name)
        self.number = name        
        
        if not as_player:
            self.model_path = model_path
            self.trainer = trainer
            self.global_episodes = global_episodes
            self.increment = self.global_episodes.assign_add(1)

            self.initialize_containers()
        
            self.summary_writer = tf.summary.FileWriter(params.summary_path+"/train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size,action_size,self.name,trainer,as_player)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        #The Below code is related to setting up the Doom environment
        game.load_config("scenarios/"+params.scenario+".cfg")
        game.set_doom_scenario_path("scenarios/"+params.scenario+".wad") #This corresponds to the simple task we will pose our agent

        game.init()
        if params.actions=='all' :
            self.actions = self.button_combinations()
        else :
            self.actions = np.identity(action_size,dtype=bool).tolist()
        #End Doom set-up
        self.env = game
    
    def button_combinations(self):
        actions = []
        
        m_left_right = [[True, False], [False, True], [False, False]]  # move left and move right
        attack = [[True], [False]]
        m_forward_backward = [[True, False], [False, True], [False, False]]  # move forward and backward
        t_left_right = [[True, False], [False, True], [False, False]]  # turn left and turn right
        
        if params.scenario=='deadly_corridor':
            actions = np.identity(6,dtype=int).tolist()
            actions.extend([[0, 0, 1, 0, 1, 0],
                            [0, 0, 1, 0, 0, 1], 
                            [1, 0, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0]])
                            
        if params.scenario=='basic':
            for i in m_left_right:
                for j in attack:
                    actions.append(i+j)
         
        if params.scenario=='defend_the_center':
            for i in t_left_right:
                for j in attack:
                    actions.append(i+j)
                    
        if params.scenario=='defend_the_line':
            for i in t_left_right:
                for j in attack:
                    actions.append(i+j)
            
        return actions
    
    def initialize_containers(self):
        self.episode_rewards = []
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
        
        if params.scenario=='defend_the_line':
            self.episode_kills = []
    
    def update_containers(self):
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_step_count)
        self.episode_mean_values.append(np.mean(self.episode_values))
        
        if params.scenario=='deadly_corridor':
            self.episode_kills.append(self.last_total_kills)
            self.episode_health.append(np.maximum(0,self.last_total_health))    
            self.episode_ammo.append(self.last_total_ammo2)
        
        if params.scenario=='basic':
            self.episode_ammo.append(self.last_total_ammo2)
        
        if params.scenario=='defend_the_center':
            self.episode_ammo.append(self.last_total_ammo2)
            self.episode_kills.append(self.last_total_kills)
        
        if params.scenario=='defend_the_line':
            self.episode_kills.append(self.last_total_kills)
            
    def update_summary(self):
        mean_reward = np.mean(self.episode_rewards[-params.freq_summary:])
        mean_length = np.mean(self.episode_lengths[-params.freq_summary:])
        mean_value = np.mean(self.episode_mean_values[-params.freq_summary:])
                    
        summary = tf.Summary()
        summary.value.add(tag='Perf/Episode_Length', simple_value=float(mean_length))
        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
        
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
            
        if params.scenario=='defend_the_line':
            mean_kills = np.mean(self.episode_kills[-params.freq_summary:])
            summary.value.add(tag='Perf/Kills', simple_value=float(mean_kills))
        
        summary.value.add(tag='Losses/Value Loss', simple_value=float(self.v_l))
        summary.value.add(tag='Losses/Policy Loss', simple_value=float(self.p_l))
        summary.value.add(tag='Losses/Entropy', simple_value=float(self.e_l))
        summary.value.add(tag='Losses/Grad Norm', simple_value=float(self.g_n))
        summary.value.add(tag='Losses/Var Norm', simple_value=float(self.v_n))
        
        self.summary_writer.add_summary(summary, self.episode_count)
        self.summary_writer.flush()
             
    
    def get_health_reward(self):
        d_health = self.env.get_game_variable(GameVariable.HEALTH) - self.last_total_health
        self.last_total_health = self.env.get_game_variable(GameVariable.HEALTH)
        if d_health == 0:
            return 0
        elif d_health < 0:
            return -5#0.1*d_health
    
    def get_ammo_reward(self):
        d_ammo = self.env.get_game_variable(GameVariable.AMMO2) - self.last_total_ammo2
        self.last_total_ammo2 = self.env.get_game_variable(GameVariable.AMMO2)
        if d_ammo == 0:
            return 0
        elif d_ammo > 0:
            return d_ammo * 0.5
        else:
            return -d_ammo * 0.5
        
    def get_kill_reward(self):
        d_kill = self.env.get_game_variable(GameVariable.KILLCOUNT) - self.last_total_kills
        self.last_total_kills = self.env.get_game_variable(GameVariable.KILLCOUNT)
        if d_kill > 0 :
            return d_kill*100
        return 0
    
    def get_bonus_reward(self):
        kills=self.env.get_game_variable(GameVariable.KILLCOUNT)
        if (kills%2 == 0) and (kills != 0):
            return 2
        else:
            return 1
        
    def initialiaze_game_vars(self):
        self.last_total_health = self.env.get_game_variable(GameVariable.HEALTH)#100.0
        self.last_total_ammo2 =self.env.get_game_variable(GameVariable.AMMO2)# 52 
        self.last_total_kills = self.env.get_game_variable(GameVariable.KILLCOUNT)#0
    
    def get_custom_reward(self,game_reward):
        if params.scenario=='basic':
            return game_reward/100.0
        
        if params.scenario=='defend_the_center':
            return game_reward + self.get_ammo_reward()/10 + 0*self.get_kill_reward()
        
        if params.scenario=='defend_the_line':
            return game_reward + 0*self.get_kill_reward()
        
        if params.scenario=='deadly_corridor':
            return (game_reward/5 + self.get_health_reward() + self.get_kill_reward() + self.get_ammo_reward())/100.
    
    def choose_action_index(self, policy, deterministic=False):
        if deterministic:
            return np.argmax(policy[0])
        else:
            return np.argmax(policy == np.random.choice(policy[0],p=policy[0]))
    
    def print_end_episode_perfs(self):
        if params.scenario=='deadly_corridor':
            print('{}, health: {}, kills:{}, episode #{}, ep_reward: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, np.maximum(0,self.last_total_health), self.last_total_kills, self.episode_count,
                            self.episode_reward, self.episode_step_count, self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
            
        if params.scenario=='basic':
            print('{}, episode #{}, ep_reward: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, self.episode_count, self.episode_reward, self.episode_step_count, 
                            self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
        
        if params.scenario=='defend_the_center':
            print('{}, kills:{}, ammo:{}, episode #{}, ep_reward: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, self.last_total_kills, self.last_total_ammo2, self.episode_count, self.episode_reward, self.episode_step_count, 
                            self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
        
        if params.scenario=='defend_the_line':
            print('{}, kills:{}, episode #{}, ep_reward: {}, steps:{}, av_reward:{}, time costs:{}'.format(
                            self.name, self.last_total_kills, self.episode_count, self.episode_reward, self.episode_step_count, 
                            self.episode_reward/self.episode_step_count, time.time()-self.episode_st))
            

    
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        
        observations, actions, rewards, next_observations, _, values = rollout.T
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the local network using gradients from loss
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
        
        return self.v_l / len(rollout),self.p_l / len(rollout),self.e_l / len(rollout), self.g_n,self.v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        self.episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():        
            start_time = time.time()
            while (not coord.should_stop()) and (self.episode_count<=params.max_episodes):
                sess.run(self.update_local_ops)
                episode_buffer = []
                self.episode_values = []
                episode_frames = []
                self.episode_reward = 0
                self.episode_step_count = 0
                d = False
                
                self.env.new_episode()
                self.episode_st = time.time()
                
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s, crop, resize)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                
                # Initialize game vars
                self.initialiaze_game_vars()
                
                
                while self.env.is_episode_finished() == False:
                    
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                                                  feed_dict={self.local_AC.inputs:[s],
                                                             self.local_AC.state_in[0]:rnn_state[0],
                                                             self.local_AC.state_in[1]:rnn_state[1]})
                    
                    action_index = self.choose_action_index(a_dist, deterministic=False)

                    r = self.get_custom_reward(self.env.make_action(self.actions[action_index], 2))

                    d = self.env.is_episode_finished()
                    
                    if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1, crop, resize)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,action_index,r,s1,d,v[0,0]])
                    self.episode_values.append(v[0,0])

                    self.episode_reward += r
                    s = s1                    
                    total_steps += 1
                    self.episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full (maximum steps), then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == params.n_steps and d != True :
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs:[s],
                                                 self.local_AC.state_in[0]:rnn_state[0],
                                                 self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        self.v_l,self.p_l,self.e_l,self.g_n,self.v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        # Update the general network from the local network
                        sess.run(self.update_local_ops)
                    if d == True:
                        # Print perfs of episode
                        self.print_end_episode_perfs()
                        break
                
                # Summaries for tensorboard
                self.update_containers()
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    self.v_l,self.p_l,self.e_l,self.g_n,self.v_n = self.train(episode_buffer,sess,gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.                    
                if self.name == 'worker_0' and self.episode_count % params.freq_gif_save == 0 and self.episode_count != 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, params.frames_path+'/image'+str(self.episode_count)+'.gif',
                                 duration=len(images)*time_per_step,true_image=True,salience=False)
                    
                if self.episode_count % params.freq_model_save == 0 and self.name == 'worker_0' and self.episode_count != 0:
                    saver.save(sess,self.model_path+'/model-'+str(self.episode_count)+'.cptk')
                    print ("Saved Model")
                
                if self.episode_count % params.freq_summary == 0 and self.episode_count != 0:
                    self.update_summary()
                    
                if self.name == 'worker_0':
                    sess.run(self.increment)
                
                self.episode_count += 1
            print("{} episodes done for {}. Time elapsed : {} s".format(self.episode_count, self.name, time.time() - start_time))
                
    def play_game(self, sess, episode_num):
        if not isinstance(sess, tf.Session):
            raise TypeError('saver should be tf.train.Saver')

        for i in range(episode_num):

            self.env.new_episode()
            state = self.env.get_state()
            s = process_frame(state.screen_buffer, crop, resize)
            rnn_state = self.local_AC.state_init
            episode_rewards = 0
            last_total_shaping_reward = 0
            step = 0
            s_t = time.time()

            while not self.env.is_episode_finished():
                state = self.env.get_state()
                s = process_frame(state.screen_buffer, crop, resize)
                a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value,self.local_AC.state_out],
                                     feed_dict={self.local_AC.inputs: [s],
                                                self.local_AC.state_in[0]:rnn_state[0],
                                                self.local_AC.state_in[1]:rnn_state[1]})
                # get a action_index from a_dist in self.local_AC.policy
                a_index = self.choose_action_index(a_dist, deterministic=True)
                # make an action
                reward = self.env.make_action(self.actions[a_index])

                step += 1

                shaping_reward = doom_fixed_to_double(self.env.get_game_variable(GameVariable.USER1)) / 100.
                r = (shaping_reward - last_total_shaping_reward)
                last_total_shaping_reward += r

                episode_rewards += reward
                
                print('Current step: #{}, Current health: {}, Current kills: {},\
                       Current reward: {}'.format(step, self.env.get_game_variable(GameVariable.HEALTH), \
                                                  self.env.get_game_variable(GameVariable.KILLCOUNT), reward))

            print('End episode: {}, Total Reward: {}, {}'.format(i+1, episode_rewards, last_total_shaping_reward))
            print('time costs: {}'.format(time.time() - s_t))
            time.sleep(5)
            
