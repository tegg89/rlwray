import numpy as np
import scipy.signal
import tensorflow as tf
from gym.spaces import Box, Discrete

# from util import gpu_sess, suppress_tf_warning
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


def mlp(x, hdims=[64,64], actv=tf.nn.relu, output_actv=None):
    for h in hdims[:-1]:
        x = tf.layers.dense(x, units=h, activation=actv)
    return tf.layers.dense(x, units=hdims[-1], activation=output_actv)


def gaussian_loglik(x, mu, log_std):
    EPS = 1e-8
    pre_sum = -0.5 * (
        ( (x - mu) / (tf.exp(log_std) + EPS) ) ** 2 +
        2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def mlp_gaussian_policy(o, a, hdims=[64,64], actv=tf.nn.relu, output_actv=None, action_space=None):
    adim = a.shape.as_list()[-1]
    mu = mlp(x=o, hdims=hdims+[adim], actv=actv, output_actv=output_actv)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(adim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_loglik(a, mu, log_std)
    logp_pi = gaussian_loglik(pi, mu, log_std)
    return pi, logp, logp_pi, mu # <= mu is added for the deterministic policy


def mlp_categorical_policy(o, a, hdims=[64,64], actv=tf.nn.relu, output_actv=None, action_space=None):
    adim = action_space.n
    logits = mlp(x=o, hdims=hdims+[adim], actv=actv, output_actv=None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=adim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=adim) * logp_all, axis=1)
    return pi, logp, logp_pi, pi


def squash_action(mu, pi, logp_pi):
    # Squash those unbounded actions
    logp_pi -= tf.reduce_sum(2 * (
        np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
    mu, pi = tf.tanh(mu), tf.tanh(pi)
    return mu, pi, logp_pi


def mlp_ppo_actor_critic(o, a, hdims=[64,64], actv=tf.nn.relu, 
                         output_actv=None, policy=None, action_space=None):
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi, mu = policy(
            o=o, a=a, hdims=hdims, actv=actv, output_actv=output_actv, action_space=action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x=o, hdims=hdims+[1], actv=actv, output_actv=None), axis=1)
    return pi, logp, logp_pi, v, mu
    

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
