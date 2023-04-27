import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import gymnasium as gym

def mlp_functional(inputs, hidden_sizes=(32,), activation='relu', use_bias=True, output_activation="sigmoid", output_reg=1e-4):
    layer = inputs
    for hidden_size in hidden_sizes[:-1]:
       # glorot_limit = np.sqrt(6 / hidden_size*10.0 + layer.shape[1])*0.02
        layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation=activation,
            # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            # bias_regularizer=regularizers.l2(1e-3),
            # activity_regularizer=regularizers.l2(5e-3)
            #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-glorot_limit, maxval=glorot_limit)
    #tf.keras.initializers.RandomNormal(stddev=0.001)
        )(layer)
   # glorot_limit = np.sqrt(6 / hidden_sizes[-1] + layer.shape[1])*1e-3
    outputs = tf.keras.layers.Dense(
        units=hidden_sizes[-1],
        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        # bias_regularizer=regularizers.l2(1e-3),
        activity_regularizer=regularizers.l1_l2(l1=output_reg, l2=output_reg),
        activation=output_activation,
        #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-glorot_limit, maxval=glorot_limit),
        use_bias=use_bias,
    )(layer)

    return outputs

def scale_by_space(scale_me, space): #scale_me: [0,1.0]
    return scale_me*(space.high - space.low) + space.low

def unscale_by_space(unscale_me, space): #outputs [-0.5, 0.5]
    return (unscale_me - space.low)/(space.high - space.low) -0.5

"""
Actor-Critics
"""
def actor(obs_space, act_space, hidden_sizes, obs_normalizer):
    inputs = tf.keras.Input((obs_space.shape[0],))/obs_normalizer
    # unscaled = unscale_by_space(inputs, obs_space)
    linear_output = mlp_functional(inputs, hidden_sizes +(act_space.shape[0],),
        use_bias=True, output_activation=None, output_reg=1, activation="relu")
    tanhed = tf.keras.layers.Activation("tanh")(linear_output)
    # clipped = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(
    #     t, -1.0, 1.0))(normed)
    # scaled = scale_by_space(normed, act_space)
    model = tf.keras.Model(inputs,tanhed)
    model.compile()
    return model

def critic(obs_space, act_space, hidden_sizes, obs_normalizer):
    concated_normalizer = np.concatenate([obs_normalizer, np.ones(act_space.shape[0])])
    inputs = tf.keras.Input((obs_space.shape[0]+act_space.shape[0],))/concated_normalizer
    outputs = mlp_functional(inputs, hidden_sizes + (1,), output_activation=None)
    biased_normed = tf.keras.layers.Activation("sigmoid")(outputs*0.1 -0.5)
    model =tf.keras.Model(inputs, biased_normed) 
    model.compile()
    return model

def mlp_actor_critic(obs_space, act_space, obs_normalizer=None, actor_hidden_sizes=(64,64), critic_hidden_sizes=(256,256)):
    if obs_normalizer is None:
        obs_normalizer = obs_space.high*0.0 + 1.0
    obs_normalizer = np.array(obs_normalizer)
    return actor(obs_space, act_space, actor_hidden_sizes, obs_normalizer), critic(obs_space, act_space, critic_hidden_sizes, obs_normalizer)