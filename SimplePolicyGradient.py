# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 02:11:23 2018

@author: Guntae
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt


gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make('CartPole-v0')

s_size = env.observation_space.shape[0]
a_size = env.action_space.n
h_size = 8

tf.reset_default_graph()

# inference network
state_holder = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
hidden = slim.fully_connected(state_holder, h_size, activation_fn=tf.nn.relu)
output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax)

reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
action_one_hot = slim.one_hot_encoding(action_holder, a_size)


responsible_weight = tf.reduce_sum(tf.multiply(output, action_one_hot), 1)

loss = -(tf.log(responsible_weight)*reward_holder)

#loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=action_one_hot)
#weighted_loss = tf.multiply(loss, reward_holder)
optimizer = tf.train.AdamOptimizer()
update = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_episodes = 2000

rList = []
i = 0
e = 0.1
while i < total_episodes:
    i += 1
    s0 = env.reset()
    t_r = 0
    
    ep_history = []
    
    while True:
        # env.render()
        
        #if np.random.randn(1) > e:
        policy = sess.run(output, feed_dict={state_holder: [s0]})
        a0 = np.random.choice(range(0, a_size), p=policy[0])
        #else:
        #    a0 = np.random.randint(env.action_space.n)
        
        s1, r0, done, _ = env.step(a0)
        
        ep_history.append([s0, a0, r0, s1, done])
             
        t_r += r0
        
        if done:
            ep_history_np = np.array(ep_history)
            ep_history_np[:, 2] = discount_rewards(ep_history_np[:, 2])
            
            # update theta
            sess.run(update, feed_dict={state_holder: np.vstack(ep_history_np[:,0]),
                                    reward_holder: ep_history_np[:, 2],
                                    action_holder: ep_history_np[:, 1]}) 
            break
    
        s0 = s1
        
    rList.append(t_r)
    
    if i % 100 == 0:
        print("reward : " + str(t_r))
    
env.close()
sess.close()