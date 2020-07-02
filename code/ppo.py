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
    adv_ph, ret_ph, rew_ph, logp_old_ph, v_old_ph = placeholders(None, None, None, None, None) 
    
    # Actor-critic model 
    ac_kwargs = dict()
    ac_kwargs['action_space'] = env.action_space
    actor_critic = mlp_ppo_actor_critic
    pi, logp, logp_pi, v, mu = actor_critic(o_ph, a_ph, **ac_kwargs)
    
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [o_ph, a_ph, adv_ph, ret_ph, rew_ph, logp_old_ph, v_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]
    
    pi_vars,v_vars = get_vars('pi'),get_vars('v')

    model = {'o_ph':o_ph, 'a_ph':a_ph, 'adv_ph':adv_ph, 'ret_ph':ret_ph, 'rew_ph':rew_ph, 'logp_old_ph':logp_old_ph, 'v_old_ph':v_old_ph,
             'pi':pi, 'logp':logp, 'logp_pi':logp_pi, 'v':v, 'mu':mu,
             'all_phs':all_phs, 'get_action_ops':get_action_ops, 'pi_vars':pi_vars, 'v_vars':v_vars}
        
    return model, sess


def create_ppo_graph(model, clip_ratio=0.2, lr=3e-4, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
    """
    PPO Computational Graph
    """    

    # PPO objectives
    
    # CALCULATE THE LOSS
    vpred = model['v']  # get the predicted value
    vpredclipped = model['v_old_ph'] + tf.clip_by_value(model['v'] - model['v_old_ph'], -clip_ratio, clip_ratio)
    v_losses1 = tf.square(vpred - model['rew_ph'])  # unclipped value
    v_losses2 = tf.square(vpredclipped - model['rew_ph'])  # clipped value
    v_loss = .5 * tf.reduce_mean(tf.maximum(v_losses1, v_losses2))
    # calculate policy loss
    ratio = tf.exp(model['logp'] - model['logp_old_ph'])  # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(model['adv_ph']>0, (1+clip_ratio)*model['adv_ph'], (1-clip_ratio)*model['adv_ph'])
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * model['adv_ph'], min_adv))
    approx_kl = tf.reduce_mean(model['logp_old_ph'] - model['logp'])  # a sample estimate for KL-divergence, easy to compute
    # final pg loss
    approx_ent = tf.reduce_mean(-model['logp'])  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))
    # total loss
    loss = pi_loss - approx_ent * ent_coef + v_loss * vf_coef
    
    # UPDATE THE PARAMETERS USING LOSS
    # 1. get the model parameters
    params = model['pi_vars'] + model['v_vars']
    # 2. build trainer
    trainer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
    print('[I] Created trainer')
    # 3. calculage gradients
    grads_and_var = trainer.compute_gradients(loss, params)
    grads, var = zip(*grads_and_var)
    if max_grad_norm is not None:
        # clip the gradients (normalize)
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads_and_var = list(zip(grads, var))
    train_op = trainer.apply_gradients(grads_and_var)
#     loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
#     stats_list = [pi_loss, v_loss, approx_ent, approx_kl, clip_frac]
    
    # Accumulate graph
    graph = {'pi_loss':pi_loss,'v_loss':v_loss,'approx_kl':approx_kl,'approx_ent':approx_ent,
             'clipfrac':clipfrac,'train_op':train_op}
    
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
    