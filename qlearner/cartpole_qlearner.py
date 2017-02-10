'''
Very basic Q-Learner that is tailored for the cart-pole environment.

This implementation leaves a lot of space for improvements. 
'''

import gym
from gym import wrappers

import math
import random
import numpy as np

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './tmp/cartpole-experiment-1')

no_episodes = 400
observation_accuracy = 1

epsilon = 1.0
alpha = 0.4
Q = dict()
available_actions = [0, 1]

        
def get_maxQ_action(state):

    maxQ = get_maxQ(state)
    maxQ_actions = []

    # add all actions with maxQ to actions-list
    for action in Q[state].keys():
        if Q[state][action] == maxQ:
            maxQ_actions.append(action)

    # randomly choose one of the maxQ actions
    return maxQ_actions[random.randint(0, len(maxQ_actions) - 1)]
    
    
def get_maxQ(state):

    maxQ = -10000.0

    for action in Q[state].keys():
        if Q[state][action] > maxQ:
            maxQ = Q[state][action]

    return maxQ
    
    
def test():
    state = env.reset()
    diff = 0.0
    epsilon = math.exp(-1.0 * alpha * 400)
    print "Epsilon : " , epsilon
    for t in range(200):
        state = map(lambda x: round(x,observation_accuracy), state)
        state_str = ' '.join([str(x) for x in state])

        if not(state_str in Q):
            Q[state_str] = dict()
            for action in available_actions:
                Q[state_str][action] = 0.0

        if random.random() <= epsilon:
            print "Taking Random Action"
            action = available_actions[random.randint(0, len(available_actions) - 1)]
        else:
            action = get_maxQ_action(state_str)

        obs, rew, done, info = env.step(action)

        state = obs

        if done:
            print ( " made it {} steps ".format(t+1) )
            break



def train():
    for i_episode in range(no_episodes):
        observation = env.reset()
    
        diff = 0.0
        
        epsilon = math.exp(-1.0 * alpha * i_episode)
    
        is_done = False
    
        for t in range(200):
            #env.render()
        
            # 1. Build State
            state = map(lambda x: round(x,observation_accuracy), observation)
            state_str = ' '.join([str(x) for x in state])
        
            # 2. Create Q entry for state
            if not(state_str in Q):
                Q[state_str] = dict()
                for action in available_actions:
                    Q[state_str][action] = 0.0
                
            # 3. Choose action
            if random.random() <= epsilon:
                action = available_actions[random.randint(0, len(available_actions) - 1)]
            else:
                action = get_maxQ_action(state_str)
            
            # 4. do step & receive reward
            observation, reward, done, info = env.step(action)
        
            # reward is always 1.0, so I'll just create my own
            curr_diff = abs(np.sum(observation))
            reward = diff - curr_diff
            diff = curr_diff

            # 5. learn
            Q[state_str][action] += (alpha * (reward - Q[state_str][action]))
            
            # 6. Check if done with this episode
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode + 1, t+1))
                is_done = True
                break
        
        if is_done == False:
            print("Episode {} COMPLETED".format(i_episode + 1))
        
        

