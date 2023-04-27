import tensorflow as tf
import numpy as np
import Pendulum
import matplotlib.pyplot as plt
import pickle


def test(actor, env, seed=123, render=True):
    o, _ = env.reset(seed=seed)
    high = env.action_space.high
    low = env.action_space.low
    os = []
    for _ in range(200):
        o, r, d, t, i, = env.step(actor(o)*(high - low)/2.0 + (high + low)/2.0)
        os.append(o)
        if render:
            env.render()
    return np.array(os)

saved = tf.saved_model.load("pendulum/actor")
# saved = tf.saved_model.load("pretty_please")
actor = lambda x: saved(np.array([x]))[0]
env = Pendulum.PendulumEnv(g=10., render_mode="human")

# os = test(actor, env)

for _ in range(5):
    test(actor, env)

# runs = np.array(list(map(lambda i: test(actor, env, seed=17+i, render=True)[:,1], range(5))))
