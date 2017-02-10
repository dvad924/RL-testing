'''
implementation of DQN framework
'''
import numpy as np
import gym

from collections import deque
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

def buildmodel(obs_shape,act_shape):
    model = Sequential()
    # Input layer for the state space shape

    model.add(Flatten(input_shape=(1,) + obs_shape))
    
    # First fully connected layer
    model.add(Dense(40))
    model.add(Activation('relu'))

    # Second fully connected layer
    model.add(Dense(40))
    model.add(Activation('relu'))


    # Output Layer
    model.add(Dense(act_shape))
    model.add(Activation('linear'))

    print( model.summary())

    adam = Adam(lr=1e-2)
    model.compile(loss='mse', optimizer=adam)
    return model




def e_greedy_choice(eps, noptions,mx):
    # Initilize prob vector of equal chance eps
    pvec = np.ones((noptions,)) * (eps/noptions)
    # give the greedy choice a boost by 1-e
    pvec[mx] += 1 - eps
    # return a random choice based on those probabilities
    return np.random.choice(len(pvec), p=pvec)


class RandMem(object):
    def __init__(self,size):
        self.dq = deque()
        self.count = 0
        self.size = size

    def add(self,obs):
        if self.count == self.size:
            self.dq.popleft()
            self.dq.append(obs)
        else:
            self.dq.append(obs)
            self.count += 1

    def size(self):
        return self.size

    def clear(self):
        self.dq.clear()
        self.count = 0
    
    def sample(self,batch):

        b = []
        if batch >= self.size:
            b = random.sample(self.dq,self.count)
        else:
            b = random.sample(self.dq, batch)
            
        state_t = np.array([_[0] for _ in b])
        action_t = np.array([_[1] for _ in b])
        reward_t = np.array([_[2] for _ in b])
        state_t2 = np.array([_[3] for _ in b])
        term     = np.array([_[4] for _ in b])

        return state_t, action_t, reward_t, state_t2, term

def init_mem(env,N,eps,Q,mem):
    done = False
    mem.clear()
    s = env.reset()
    for _ in xrange(N):
        st = s.reshape(1,1,4)
        a = e_greedy_choice(eps, env.action_space.n, np.argmax(Q.predict(st)))
        ns,r,done,_ = env.step(a)
        ns = ns.reshape(1,4)
        mem.add((s.reshape(1,4),a,r,ns,done))
        s = ns
        if done:
            s = env.reset()
        
BATCH = 100
HIST = 100
NEPISODES = 2000
Epsilon = 0.2
Gamma = 0.9
epsilon_decay = 0.995

env = gym.make('CartPole-v0')
mem = RandMem(HIST)

print "constructing model"
Q = buildmodel(env.observation_space.shape,env.action_space.n)

init_mem(env,HIST,Epsilon,Q,mem)
print "sampling for memory"
maxlen = 0
for i_ep in xrange(NEPISODES):
    print "training"
    done = False
    state = env.reset() # initialize state
    n_steps= 1
    loss = 0
    while not done:
        state = state.reshape(1,1,4)
        # get next action
        a = e_greedy_choice(Epsilon,env.action_space.n, np.argmax(Q.predict(state)))
        # get reward and next state
        nstate,r,done,_ = env.step(a)
        nstate = nstate.reshape(1,4)
        mem.add((state.reshape(1,4),a,r,nstate,done))

        state = nstate
        
        n_steps += 1
        #now sample minibatch
        minibatch = mem.sample(BATCH)

        inputs = np.zeros((BATCH,state.shape[0],state.shape[1]))
        targets = np.zeros((BATCH,env.action_space.n))
        
        # for i in xrange(BATCH):
        #     s_t = minibatch[0][i]
        #     a_t = minibatch[1][i]
        #     r_t = minibatch[2][i]
        #     s1_t = minibatch[3][i]
        #     term = minibatch[4][i]

        #     inputs[i:i+1] = s_t
            
        #     targets[i] = Q.predict(s_t)

        #     Q_sa = targets[i]
            
        #     if term:
        #         targets[i,a_t] = r_t
        #     else:
        #         targets[i,a_t] = r_t + Gamma * np.amax(Q_sa)

        # Vectorized ?
        inputs = minibatch[0]
        term = minibatch[4]
        targets = Q.predict(inputs)

        Q_sa = targets
        a_ts = minibatch[1]

        tval = term == True
        nval = term == False
        Q_sa[tval, a_ts[tval]] = minibatch[2][tval]
        Q_sa[nval, a_ts[nval]] = minibatch[2][nval] + Gamma * np.amax(Q_sa[nval],axis=1)

        loss += Q.train_on_batch(inputs,targets)
    print "Loss : {}".format(float(loss)/n_steps)
    print "Episode length: {}".format(n_steps)
    print "Episode {} complete".format(i_ep)
    Epsilon *= epsilon_decay
    print Epsilon
    maxlen = max(maxlen,n_steps)
print "Maximum Length : ", maxlen
