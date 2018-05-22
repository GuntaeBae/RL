# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 02:11:23 2018

@author: Guntae
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
from gym import spaces
import cv2
from collections import deque

gamma = 0.99

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
    
    

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC'):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [1, nf, 1, 1], initializer=tf.constant_initializer(0.0))
        if data_format == 'NHWC': b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)

def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.leaky_relu
    h = pool(activ(conv(scaled_images, 'c1', nf=32, rf=5, stride=1, init_scale=np.sqrt(2), pad='SAME')))
    h2 = pool(activ(conv(h, 'c2', nf=32, rf=5, stride=1, init_scale=np.sqrt(2), pad='SAME')))
    h3 = pool(activ(conv(h2, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), pad='SAME')))
    h4 = pool(activ(conv(h3, 'c4', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad='SAME')))
    h5 = conv_to_fc(h4)
    return activ(fc(h5, 'fc1', nh=512, init_scale=np.sqrt(2)))

env_id = 'CartPole-v0'
env_id = 'PongDeterministic-v0'
env = gym.make(env_id)
env.seed(0)
env = WarpFrame(env)
#env = ScaledFloatFrame(env)
env = ClipRewardEnv(env)
env = FrameStack(env, 4)

action_index = [0,2,3]

s_size = env.observation_space.shape
a_size = len(action_index)#env.action_space.n
h_size = 8

tf.reset_default_graph()

# inference network
state_holder = tf.placeholder(shape=[None]+list(s_size), dtype=tf.float32)
with tf.variable_scope("model"):
    hidden = nature_cnn(state_holder)
    logits = slim.fully_connected(hidden, a_size, activation_fn=None)
    output_p = tf.nn.softmax(logits)
    output_v = slim.fully_connected(hidden, 1, activation_fn=None)

target_v_holder = tf.placeholder(shape=[None,], dtype=tf.float32)
advantage_holder = tf.placeholder(shape=[None,], dtype=tf.float32)
action_holder = tf.placeholder(shape=[None,], dtype=tf.int32)
#action_one_hot = slim.one_hot_encoding(action_holder, a_size)

eps = 1e-6
# Calculate losses
# Entropy
entory_loss = tf.reduce_mean(-tf.reduce_sum(output_p * tf.log(output_p + eps), axis=1))

# Policy Graident loss
#logits_response = tf.reduce_sum(output_p_logits * action_one_hot, axis=1)
#actor_loss = -tf.reduce_sum(logits_response * advantage_holder)
neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_holder)
#pg_loss = tf.reduce_mean(ADV * neglogpac)
actor_loss = tf.reduce_mean(advantage_holder * neglogpac)

 # Value/Q function loss, and explained variance
critic_loss = 0.5 * tf.reduce_mean(tf.square(tf.squeeze(output_v) - target_v_holder)/2.0)


loss = actor_loss + 0.5 * critic_loss - 0.01 * entory_loss

'''
#loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=action_one_hot)
#weighted_loss = tf.multiply(loss, reward_holder)
optimizer1 = tf.train.AdamOptimizer(learning_rate=1e-4)
update_actor = optimizer1.minimize(actor_loss)

optimizer2 = tf.train.AdamOptimizer(learning_rate=1e-4)
update_critic = optimizer2.minimize(value_loss)
'''

#optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
#update = optimizer.minimize(loss)
def discount_advantage(r, v):
    advantage = np.zeros_like(r)
    
    for t in range(0, r.size):
        advantage[t] = r[t] + gamma * v[t+1] - v[t]
        
    return advantage

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        
    #baseline = np.mean(discounted_r)
    #discounted_r = discounted_r - baseline
    return discounted_r         

def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()

params = find_trainable_variables("model")
grads = tf.gradients(loss, params)
grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
grads = list(zip(grads, params))
trainer = tf.train.AdamOptimizer(learning_rate=2e-4)
update = trainer.apply_gradients(grads)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_episodes = 600

f = open("c:/result.txt", 'a')

rList = []
i = 0
e = 0.1
while i < total_episodes:
    i += 1
    s0 = env.reset()
    t_r = 0
    step = 0
    ep_history = []
    
    while True:
        #env.render()
        
        #if np.random.randn(1) > e:
        
        policy, value = sess.run([output_p, output_v], feed_dict={state_holder: [s0]})
        a0 = np.random.choice(range(0, a_size), p=policy[0])
        #else:
        #    a0 = np.random.randint(env.action_space.n)

        s1, r0, done, _ = env.step(action_index[a0])
        
        ep_history.append([s0, a0, r0, s1, done, value[0,0]])
             
        t_r += r0
        step += 1
        
        if len(ep_history) == 30 or done:
            if done:
                value = 0.0
            else:
                value = sess.run(output_v, feed_dict={state_holder: [s1]})[0,0]
            
            ep_history_np = np.array(ep_history)
            rewards = ep_history_np[:, 2]
            values = ep_history_np[:, 5]
            rewards_plus = np.array(rewards.tolist() + [value])
            values_plus = np.array(values.tolist() + [value])
            
            dicounted_rewards = discount_rewards(rewards_plus)[:-1]
            
            #print(rewards, dicounted_rewards, values)
            advantage = discount_advantage(rewards, values_plus)
            #print(rewards, values_plus, advantage)
            _, e_l, a_l, c_l = sess.run(
                    [update, entory_loss, actor_loss, critic_loss],
                    feed_dict={
                    state_holder: np.stack(ep_history_np[:,0]),
                    target_v_holder : dicounted_rewards,
                    advantage_holder: advantage,
                    action_holder: ep_history_np[:, 1]
                    })
            #print(p_i_output)
            #print(o_v, dicounted_rewards, s_e)
            print(e_l, a_l, c_l)
            f.write(str(e_l) + " " + str(a_l) + " " + str(c_l))
            ep_history.clear()            
     
            if done:
                break
            
        s0 = s1
        
    rList.append(t_r)
    
    #if i % 100 == 0:
    print("game : " + str(i) + ", reward : " + str(t_r) + ", step : " + str(step))
    f.write("game : " + str(i) + ", reward : " + str(t_r) + ", step : " + str(step))
    
env.close()
sess.close()
f.close()

#%%
