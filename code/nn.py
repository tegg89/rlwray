import numpy as np
import scipy.signal
import tensorflow as tf

from util import gpu_sess, suppress_tf_warning
# from episci.environment_wrappers.frame_stack_wrapper import FrameStack
# from episci.environment_wrappers.tactical_action_adt_env_continuous import CustomADTEnvContinuous
# from episci.agents.utils.constants import Agents, RewardType


def get_env(batch=True):    
    red_distribution = {
        Agents.SPOT_4G: 0.15,
        Agents.SPOT_5G: 0.30,
        Agents.SPOT_RANDOM: 0.45,
        Agents.EXPERT_SYSTEM_TRIAL_2: 0.6,
        Agents.EXPERT_SYSTEM_TRIAL_3_SCRIMMAGE_4: 0.75,
        Agents.EXPERT_SYSTEM: 1.0
    }
    env_config = {
        "red_distribution": red_distribution,
        "reward_type": RewardType.SHAPED
    }
    if batch:
        return FrameStack(CustomADTEnvContinuous(env_config), k=25, frame_skip=10)  # keep past 'k' obs's and update every `frame_skip` steps
    else:
        return CustomADTEnvContinuous(env_config)


def get_eval_env(batch=True):
    red_distribution = {
        Agents.SPOT_4G: 0.15,
        Agents.SPOT_5G: 0.30,
        Agents.SPOT_RANDOM: 0.45,
        Agents.EXPERT_SYSTEM_TRIAL_2: 0.6,
        Agents.EXPERT_SYSTEM_TRIAL_3_SCRIMMAGE_4: 0.75,
        Agents.EXPERT_SYSTEM: 1.0
    }
    env_config = {
        "red_distribution": red_distribution,
        "reward_type": RewardType.SHAPED
    }
    if batch:
        return FrameStack(CustomADTEnvContinuous(env_config), k=25, frame_skip=10)  # keep past 'k' obs's and update every `frame_skip` steps
    else:
        return CustomADTEnvContinuous(env_config)


def mlp(x, hdims=[256, 256], actv=tf.nn.relu, out_actv=tf.nn.relu):
    ki = tf.truncated_normal_initializer(stddev=0.1)
    for hdim in hdims[:-1]:
        x = tf.layers.dense(x, units=hdim, activation=actv, kernel_initializer=ki)
    return tf.layers.dense(x, units=hdims[-1], activation=out_actv, kernel_initializer=ki)


def gaussian_loglik(x, mu, log_std):
    EPS = 1e-8
    pre_sum = -0.5 * (
        ( (x - mu) / (tf.exp(log_std) + EPS) ) ** 2 +
        2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def mlp_gaussian_policy(o, adim=2, hdims=[256,256], actv=tf.nn.relu):
    net = mlp(x=o, hdims=hdims, actv=actv, out_actv=actv) # feature 
    mu = tf.layers.dense(net, adim, activation=None) # mu
    log_std = tf.layers.dense(net, adim, activation=None) # log_std

    LOG_STD_MIN, LOG_STD_MAX = -10.0, +2.0
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX) 

    std = tf.exp(log_std) # std 
    pi = mu + tf.random_normal(tf.shape(mu)) * std  # sampled
    logp_pi = gaussian_loglik(x=pi, mu=mu, log_std=log_std) # log lik

    return mu, pi, logp_pi


def squash_action(mu, pi, logp_pi):
    # Squash those unbounded actions
    logp_pi -= tf.reduce_sum(2 * (
        np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
    mu, pi = tf.tanh(mu), tf.tanh(pi)
    return mu, pi, logp_pi


def mlp_ppo_actor_critic(o, a, hdims=[256,256], actv=tf.nn.relu,
                         out_actv=None, policy=mlp_gaussian_policy):
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy
        
    adim = a.shape.as_list()[-1]
    
    with tf.variable_scope('pi'):
        pi, logp, logp_pi, mu = policy(o=o, adim=adim, hdims=hdims, actv=actv)
        # mu, pi, logp_pi = squash_action(mu=mu, pi=pi, logp_pi=logp_pi)
        
    with tf.variable_scope('v'): 
        v = tf.squeeze(mlp(x=o, hdims=hdims+[1], actv=actv, out_actv=None), axis=1)
        
    return mu, logp, logp_pi, v


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    """
    Usage: a_ph,b_ph,c_ph = placeholders(adim,bdim,None)
    """
    return [placeholder(dim) for dim in args]

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
