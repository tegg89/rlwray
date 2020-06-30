import os

import gym
import ray
import numpy as np
import tensorflow as tf

from nn import *
from util import count_vars, PPOBuffer


def create_ppo_model(env=None, hdims=[256,256]):
    """
    Proximal Policy Optimization Model (compatible with Ray)
    """
    import tensorflow as tf # make it compatible with Ray actors
    
    # Have own session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Placeholders
    odim = env.observation_space.shape[0]
    adim = env.action_space.shape[0]
    o_ph, a_ph = placeholders(odim, adim)
    adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)
    
    # Actor-critic model 
    ac_kwargs = dict()
    ac_kwargs['action_space'] = env.action_space
    actor_critic = mlp_ppo_actor_critic
    pi, logp, logp_pi, v, mu = actor_critic(o_ph, a_ph, **ac_kwargs)
    
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [o_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]
    
    pi_vars,v_vars = get_vars('pi'),get_vars('v')

    model = {'o_ph':o_ph, 'a_ph':a_ph, 'adv_ph':adv_ph, 'ret_ph':ret_ph, 'logp_old_ph':logp_old_ph,
             'pi':pi, 'logp':logp, 'logp_pi':logp_pi, 'v':v, 'mu':mu,
             'all_phs':all_phs, 'get_action_ops':get_action_ops, 'pi_vars':pi_vars, 'v_vars':v_vars}
        
    return model, sess


def create_ppo_graph(model, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3):
    """
    PPO Computational Graph
    """    

    # PPO objectives
    ratio = tf.exp(model['logp'] - model['logp_old_ph'])  # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(model['adv_ph']>0, (1+clip_ratio)*model['adv_ph'], (1-clip_ratio)*model['adv_ph'])
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * model['adv_ph'], min_adv))
    v_loss = tf.reduce_mean((model['ret_ph'] - model['v'])**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(model['logp_old_ph'] - model['logp'])  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-model['logp'])  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))
  
    # Policy train op
    train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
    
    # Accumulate graph
    graph = {'pi_loss':pi_loss,'v_loss':v_loss,'approx_kl':approx_kl,'approx_ent':approx_ent,
             'clipfrac':clipfrac,'train_pi':train_pi,'train_v':train_v}
    
    return graph
    

def save_ppo_model(npz_path, R, VERBOSE=True):
    """
    Save PPO model weights
    """
    
    # PPO model
    tf_vars = R.model['main_vars'] + R.model['target_vars']
    data2save, var_names, var_vals = dict(), [], []

    for v_idx,tf_var in enumerate(tf_vars):
        var_name, var_val = tf_var.name, R.sess.run(tf_var)
        var_names.append(var_name)
        var_vals.append(var_val)
        data2save[var_name] = var_val
        if VERBOSE:
            print ("[%02d]  var_name:[%s]  var_shape:%s"%
                (v_idx, var_name, var_val.shape,)) 
            
    # Create folder if not exist
    dir_name = os.path.dirname(npz_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print ("[%s] created."%(dir_name))
        
    # Save npz
    np.savez(npz_path,**data2save)
    print ("[%s] saved."%(npz_path))
            
            
def restore_ppo_model(npz_path, R, VERBOSE=True):
    """
    Restore PPO model weights
    """
    
    # Load npz
    l = np.load(npz_path)
    print ("[%s] loaded."%(npz_path))
    
    # Get values of PPO model  
    tf_vars = R.model['main_vars'] + R.model['target_vars']
    var_vals = []
    for tf_var in tf_vars:
        var_vals.append(l[tf_var.name])   
        
    # Assign weights of PPO model
    R.set_weights(var_vals)
    