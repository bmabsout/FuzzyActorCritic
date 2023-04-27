from dataclasses import asdict, dataclass
from typing import NamedTuple
import numpy as np
import tensorflow as tf
import pickle
import gymnasium as gym
import time
from . import core
from fuzzy_rl.utils.logx import EpochLogger

@tf.function
def geo(l,axis=0):
    return tf.exp(tf.reduce_mean(tf.math.log(l),axis=axis))

@tf.function
def p_mean(l, p, slack=0.0, axis=1):
    slacked = l + slack
    if(len(slacked.shape) == 1): #enforce having batches
        slacked = tf.expand_dims(slacked, axis=0)
    batch_size = slacked.shape[0]
    zeros = tf.zeros(batch_size, l.dtype)
    ones = tf.ones(batch_size, l.dtype)
    handle_zeros = tf.reduce_all(slacked > 1e-20, axis=axis) if p <=1e-20 else tf.fill((batch_size,), True)
    escape_from_nan = tf.where(tf.expand_dims(handle_zeros, axis=axis), slacked, slacked*0.0 + 1.0)
    handled = (
            geo(escape_from_nan, axis=axis)
        if p == 0 else
            tf.reduce_mean(escape_from_nan**p, axis=axis)**(1.0/p)
        ) - slack
    res = tf.where(handle_zeros, handled, zeros)
    return res

@tf.function
def p_to_min(l, p=0, q=0):
    deformator = p_mean(1.0-l, q)
    return p_mean(l, p)*deformator + (1.0-deformator)*tf.reduce_min(l)

# @tf.function
# def with_mixer(actions): #batch dimension is 0
#     return actions-tf.reduce_min(actions,axis=1)

# def mixer_diff_dfl(a1,a2):
#     return tf.abs(with_mixer(a1)-with_mixer(a2))/2.0

@tf.function
def weaken(weaken_me, weaken_by):
    return (weaken_me + weaken_by)/(1.0 + weaken_by)

@tf.custom_gradient
def scale_gradient(x, scale):
  grad = lambda dy: (dy * scale, None)
  return x, grad


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class HyperParams:
    def __init__( self,
            ac_kwargs={"actor_hidden_sizes":(32,32), "critic_hidden_sizes":(512,512)},
            seed=int(time.time()* 1e5) % int(1e6),
            steps_per_epoch=5000,
            pi_bar_variance=[
                1.0,1.0,1.0,
                1.0,1.0,1.0,
                0.1,0.1,0.1,
                0.0,0.0,0.0,0.0],
            epochs=100,
            replay_size=int(1e5),
            gamma=0.9,
            polyak=0.995,
            pi_lr=1e-4,
            q_lr=1e-4,
            batch_size=100,
            start_steps=10000,
            act_noise=0.1,
            max_ep_len=1000,
            train_every=50,
            train_steps=30,
        ):
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.pi_bar_variance = pi_bar_variance
        self.epochs = epochs
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.act_noise = act_noise
        self.max_ep_len = max_ep_len
        self.train_every = train_every
        self.train_steps = train_steps


"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(env_fn, hp: HyperParams=HyperParams(),actor_critic=core.mlp_actor_critic, logger_kwargs=dict(), save_freq=1, on_save=lambda *_:(), anchor_q=None):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    logger = EpochLogger(**logger_kwargs)
    logger.save_config(hp.__dict__)

    tf.random.set_seed(hp.seed)
    np.random.seed(hp.seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_q_val = 1.0/(1.0-hp.gamma)

    # Main outputs from computation graph
    with tf.name_scope('main'):
        pi_network, q_network = actor_critic(env.observation_space, env.action_space, **hp.ac_kwargs)
        if anchor_q:
            anchor_targ_network = tf.keras.models.clone_model(anchor_q)
    
    # Target networks
    with tf.name_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ_network, q_targ_network = actor_critic(env.observation_space, env.action_space, **hp.ac_kwargs)

    # make sure network and target network is using the same weights
    pi_targ_network.set_weights(pi_network.get_weights())
    q_targ_network.set_weights(q_targ_network.get_weights())
    if anchor_q:
        anchor_targ_network.set_weights(anchor_q.get_weights())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=hp.replay_size)
    # Separate train ops for pi, q
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.pi_lr)
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.q_lr)
    if anchor_q:
        anchor_q_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.q_lr)

    # Polyak averaging for target variables
    @tf.function
    def target_update():
        for v_main, v_targ in zip(pi_network.trainable_variables, pi_targ_network.trainable_variables):
            v_targ.assign(hp.polyak*v_targ + (1-hp.polyak)*v_main)
        for v_main, v_targ in zip(q_network.trainable_variables, q_targ_network.trainable_variables):
            v_targ.assign(hp.polyak*v_targ + (1-hp.polyak)*v_main)

    @tf.function
    def q_update(obs1, obs2, acts, rews, dones):
        with tf.GradientTape() as tape:
            q = tf.squeeze(q_network(tf.concat([obs1, acts], axis=-1)), axis=1)
            pi_targ = pi_targ_network(obs2)
            q_pi_targ = tf.squeeze(q_targ_network(tf.concat([obs2, pi_targ], axis=-1)), axis=1)
            backup = tf.stop_gradient(rews/max_q_val + (1 - d)*hp.gamma * q_pi_targ)
            q_loss = tf.reduce_mean((q-backup)**2) #+ sum(q_network.losses)*0.1
        grads = tape.gradient(q_loss, q_network.trainable_variables)
        grads_and_vars = zip(grads, q_network.trainable_variables)
        q_optimizer.apply_gradients(grads_and_vars)
        return q_loss, q

    @tf.function
    def anchor_q_update(obs1, obs2, acts, rews, dones):
        with tf.GradientTape() as tape:
            q = tf.squeeze(anchor_q(tf.concat([obs1, acts], axis=-1)), axis=1)
            pi_targ = pi_targ_network(obs2)
            q_pi_targ = tf.squeeze(anchor_targ_network(tf.concat([obs2, pi_targ], axis=-1)), axis=1)
            backup = tf.stop_gradient(rews/max_q_val + (1 - d)*hp.gamma * q_pi_targ)
            q_loss = tf.reduce_mean((q-backup)**2) #+ sum(q_network.losses)*0.1
        grads = tape.gradient(q_loss, anchor_q.trainable_variables)
        grads_and_vars = zip(grads, anchor_q.trainable_variables)
        anchor_q_optimizer.apply_gradients(grads_and_vars)
        return q_loss, q


    @tf.function
    def pi_update(obs1, obs2, debug=False):
        with tf.GradientTape() as tape:
            pi = pi_network(obs1)
            pi2 = pi_network(obs2)
            q_c = tf.stack([1.0])
            # if loss_q < 1e-4:
            q_c = tf.stack([tf.reduce_mean(q_targ_network(tf.concat([obs1, pi], axis=-1))**0.5)**2.0])
            noise = tf.random.normal(
                [len(hp.pi_bar_variance)], mean=0.0, stddev=np.array(hp.pi_bar_variance),
            )
            pi_bar = pi_network(obs1+noise)
            var_sum = sum(map(lambda v: tf.reduce_mean(tf.abs(v)**2.0),pi_network.trainable_variables))
            pi_weight_c = tf.stack([50.0/(50.0+var_sum)])**2.0
            before_tanh_c = tf.stack([(20.0/(20.0+sum(pi_network.losses)))**2.0])
            # objective for regularizing the output of the nn as well as the weights
            # tf.print(pi_network.trainable_variables)
            # tf.print(pi_network.losses)

            temporal_c = p_mean(p_mean((0.1/(0.1+tf.abs(pi-pi2)))**2.0, 1.0), 1.0)
            # objective for minimizing subsequent action differences

            spatial_c = p_mean(p_mean(0.1/(0.1+tf.abs(pi-pi_bar)), 0.), 0.)
            # objective representing similar inputs should map to similar outputs

            center_c = p_mean(p_mean(1.0 - (tf.abs(pi+0.76)/1.761),0.),0.)
            # objective for maintaining an output of -0.8

            reg_c = tf.squeeze(p_mean(tf.stack([spatial_c, temporal_c, before_tanh_c],axis=1), 0.0))
            # all_c = p_to_min(tf.stack([q_c, tf.stack([reg_c])]), q=0.0)
            all_c = p_mean(tf.stack([scale_gradient(q_c, 3e2), before_tanh_c], axis=1), 0.0)
            # all_c = q_c + 0.008*pi_diffs_c + 0.005*pi_bar_c + 0.025*center_c
            # if debug:
            #     tf.print("temporal_c", temporal_c)
            #     tf.print("spatial_c", spatial_c)
            #     tf.print("center_c", center_c)
            #     tf.print("pi_weight_c", pi_weight_c)
            #     tf.print("before_tanh_c", before_tanh_c)
            #     tf.print("reg_c", reg_c)
            #     tf.print("aq_c", aq_c)
            #     tf.print("q_c", q_c)
            pi_loss = 1.0 - all_c
        grads = tape.gradient(pi_loss, pi_network.trainable_variables)
        grads_and_vars = zip(grads, pi_network.trainable_variables)
        pi_optimizer.apply_gradients(grads_and_vars)
        return all_c, q_c, reg_c, temporal_c, spatial_c, before_tanh_c, pi_weight_c

    def get_action(o, noise_scale):
        minus_1_to_1 = pi_network(tf.constant(o.reshape(1,-1))).numpy()[0]
        noise = noise_scale * np.random.randn(act_dim)
        a = (minus_1_to_1 + noise) * (env.action_space.high - env.action_space.low)/2.0 + (env.action_space.high + env.action_space.low)/2.0
        return np.clip(a, env.action_space.low, env.action_space.high)

    def test_agent(n=1):
        print("testing agents")
        sum_step_return = 0
        for j in range(n):
            o, _ = env.reset()
            r, d, ep_ret, ep_len =0, False, 0, 0
            while not(d or (ep_len == hp.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _, _ = env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
                env.render()
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            sum_step_return += ep_ret/ep_len
        return sum_step_return/n

    start_time = time.time()
    o, _ = env.reset()
    r, d, ep_ret, ep_len = 0, False, 0, 0
    total_steps = hp.steps_per_epoch * hp.epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > hp.start_steps:
            a = get_action(o, hp.act_noise*t/total_steps)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==hp.max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if t > hp.start_steps and t % hp.train_every == 0:
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for train_step in range(hp.train_steps):
                batch = replay_buffer.sample_batch(hp.batch_size)

                obs1 = tf.constant(batch['obs1'])
                obs2 = tf.constant(batch['obs2'])
                acts = tf.constant(batch['acts'])
                rews = tf.constant(batch['rews'])
                dones = tf.constant(batch['done'])


                # Q-learning update
                loss_q, q_vals = q_update(obs1, obs2, acts, rews, dones)
                # if anchor_q:
                #     loss_anchor_q, anchor_q_vals = anchor_q_update(anchor_obs1, anchor_obs2, anchor_acts, anchor_rews, anchor_dones)
                logger.store(LossQ=loss_q)
                # logger.store(LossAQ=loss_anchor_q)

                # Policy update
                all_c, q_c, reg_c, temporal_c, spatial_c, center_c, pi_weight_c = pi_update(obs1, obs2, (train_step+1)%20 == 0)

                logger.store(
                    All=all_c,
                    Q=q_c,
                    # AQ=anchor_q_vals,
                    Reg=reg_c,
                    Temporal=temporal_c,
                    Spatial=spatial_c,
                    Center=center_c,
                    Pi_weight=pi_weight_c
                )

                # target update
                target_update()

        if d or (ep_len == hp.max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            r, d, ep_ret, ep_len = 0, False, 0, 0

        # End of epoch wrap-up
        if t > hp.start_steps and t % hp.steps_per_epoch == 0:
            print(hp.steps_per_epoch)
            epoch = t // hp.steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == hp.epochs-1):
                on_save(pi_network, q_network, epoch//save_freq, replay_buffer)
            #     logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            # test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', average_only=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Temporal', average_only=True)
            logger.log_tabular('Spatial', average_only=True)
            logger.log_tabular('Center', average_only=True)
            logger.log_tabular('Pi_weight', average_only=True)
            logger.log_tabular('Reg', average_only=True)
            logger.log_tabular('Q', average_only=True)
            # logger.log_tabular('AQ', average_only=True)
            logger.log_tabular('All', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('LossAQ', average_only=True)

            logger.dump_tabular()
    return pi_network

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)