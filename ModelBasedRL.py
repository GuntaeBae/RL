# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 02:11:23 2018

@author: Guntae
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()

tf.reset_default_graph()

o_size = 4
w1_size = 8

observations = tf.placeholder(dtype=tf.float32, shape=[None, o_size], name='input_x')
W1 = tf.get_variable('W1', shape=[o_size, w1_size], tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable('W2', shape=[w1_size, ])