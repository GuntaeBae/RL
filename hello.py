# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import gym
env = gym.make("Pong-v0")
env.reset()
for _ in range(10):
    # env.render()
    s, a, r, d = env.step(env.action_space.sample())
    print(s, a, r, d)
    if d:
        env.reset()