import fuzzy_rl.rl_algs.ddpg.ddpg as rl_alg
import Pendulum
import numpy as np
import time
import pickle
import tensorflow as tf

def on_save(actor, q_network, epoch, replay_buffer):
    actor.save("pendulum/actor")
    q_network.save("pendulum/critic")

def existing_actor_critic(*args, **kwargs):
    return tf.keras.models.load_model("pendulum/actor"), tf.keras.models.load_model("pendulum/critic")

rl_alg.ddpg(lambda: Pendulum.PendulumEnv(g=10.0)
    , hp = rl_alg.HyperParams(
        seed=int(time.time()* 1e5) % int(1e6),
        steps_per_epoch=1000,
        ac_kwargs={
            "actor_hidden_sizes":(32,32),
            "critic_hidden_sizes":(256,256),
            "obs_normalizer": np.array([1.0, 1.0, 8.0])
        },
        pi_bar_variance=[0.0,0.0,0.0],
        start_steps=1000,
        replay_size=int(1e5),
        gamma=0.9,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=200,
        act_noise=0.1,
        max_ep_len=200,
        epochs=10,
        train_every=50,
        train_steps=30,
    )
    , on_save=on_save
)