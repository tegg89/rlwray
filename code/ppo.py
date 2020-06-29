import os

import gym
import ray
import numpy as np
import tensorflow as tf

from nn import *
from util import count_vars, PPOBuffer


def create_ppo_model(odim=10, adim=2, hdims=[256, 256], actv=tf.nn.relu):
    """
    Proximal Policy Optimization Model (compatible with Ray)
    """
    import tensorflow as tf # make it compatible with Ray actors
    
    # Have own session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Placeholders
    o_ph, a_ph = placeholders(odim, adim)
    adv_ph, ret_ph, logp_old_ph = placeholders(None, None, None)
    
    # Actor-critic model 
    ac_kwargs = dict()
    ac_kwargs['action_space'] = env.action_space
    actor_critic = mlp_actor_critic
    pi, logp, logp_pi, v, mu = mlp_ppo_actor_critic(o_ph, a_ph, **ac_kwargs)
    
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [o_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    model = {'o_ph':o_ph, 'a_ph':a_ph, 'adv_ph':adv_ph, 'ret_ph':ret_ph, 'logp_old_ph':logp_old_ph,
             'pi':pi, 'logp':logp, 'logp_pi':logp_pi, 'v':v, 'mu':mu,
             'all_phs':all_phs, 'get_action_ops':get_action_ops}
        
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
    train_p = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
    
    # Accumulate graph
    graph = {'pi_loss':pi_loss,'v_loss':v_loss,'approx_kl':approx_kl,'approx_ent':approx_ent,
             'clipfrac':clipfrac,'train_pi':train_pi,'train_v':train_v}
    
    return graph
    

class RolloutWorkerClass(object):
    """
    Worker without RAY (for update purposes)
    """
    def __init__(self, hdims=[256,256], actv=tf.nn.relu,
                 pi_lr=3e-4, vf_lr=1e-3, clip_ratio=0.2, 
                 seed=1, batch=True):
        import tensorflow as tf

        self.seed = seed
        # Each worker should maintain its own environment
        suppress_tf_warning() # suppress TF warnings
        gym.logger.set_level(40) # gym logger
        self.env = get_eval_env(batch)

        if batch:
            self.odim = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        else:
            self.odim = self.env.observation_space.shape[0]
        self.adim = self.env.action_space.shape[0]
        
        # Create PPO model and computational graph
        self.model, self.sess = create_ppo_model(odim=self.odim, adim=self.adim, hdims=hdims, actv=actv)
        self.graph = create_ppo_graph(self.model, clip_ratio=clip_ratio, pi_lr=pi_lr, vf_lr=vf_lr)

        # Initialize model
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def get_action(self, o, deterministic=False):
        return sess.run(model['get_action_ops'],feed_dict={model['o_ph']:o.reshape(1,-1)})
        # act_op = self.model['mu'] if deterministic else self.model['pi']
        # v_op = self.model['v'], 
        # p_op = self.model['pi']
        # return self.sess.run([act_op, v_op, p_op], feed_dict={self.model['o_ph']: o.reshape(1, -1)})[0]  # TODO: Check

    def get_weights(self):
        """
        Get weights
        """
        weight_vals = self.sess.run(self.model['main_vars'])
        return weight_vals
        

@ray.remote
class RayRolloutWorkerClass(object):
    """
    Rollout Worker with RAY
    """
    def __init__(self, worker_id=0, hdims=[256,256], 
                 actv=tf.nn.relu, ep_len_rollout=1000, batch=True):
        import tensorflow as tf

        # Parse
        self.worker_id = worker_id
        self.ep_len_rollout = ep_len_rollout

        # Each worker should maintain its own environment
        suppress_tf_warning() # suppress TF warnings
        gym.logger.set_level(40) # gym logger 
        self.env = get_env(batch)
        
        if batch:
            odim = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        else:
            odim = self.env_observation_space.shape[0]
        adim = self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Create PPO model
        self.model, self.sess = create_ppo_model(odim=self.odim, adim=self.adim, hdims=hdims, actv=actv)
        self.sess.run(tf.global_variables_initializer())
        print ("Ray Worker [%d] Ready."%(self.worker_id))
        
        # Flag to initialize assign operations for 'set_weights()'
        self.FIRST_SET_FLAG = True

        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True
        
    def get_action(self, o, deterministic=False):
        # act_op = self.model['mu'] if deterministic else self.model['pi']
        # v_op = self.model['v'], 
        # p_op = self.model['pi']
        # return self.sess.run([act_op, v_op, p_op], feed_dict={self.model['o_ph']: o.reshape(1, -1)})[0]  # TODO: Check
        return sess.run(model['get_action_ops'],feed_dict={model['o_ph']:o.reshape(1,-1)})
    
    def set_weights(self, weight_vals):
        """
        Set weights
        """
        import tensorflow as tf

        if self.FIRST_SET_FLAG:
            self.FIRST_SET_FLAG = False
            self.assign_placeholders = []
            self.assign_ops = []
            
            for w_idx, weight_tf_var in enumerate(self.model['main_vars']):
                a = weight_tf_var
                assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                assign_op = a.assign(assign_placeholder)
                self.assign_placeholders.append(assign_placeholder)
                self.assign_ops.append(assign_op)
                
        for w_idx, weight_tf_var in enumerate(self.model['main_vars']):
            # Memory-leakage-free assign
            self.sess.run(self.assign_ops[w_idx],
                          {self.assign_placeholders[w_idx]: weight_vals[w_idx]})
            
    def rollout(self, buf, local_steps_per_epoch):
        """
        Rollout
        """        
        if self.FIRST_ROLLOUT_FLAG:
            self.FIRST_ROLLOUT_FLAG = False
            self.o = self.env.reset() # reset environment

        # Loop
        for t in range(local_steps_per_epoch):
            ops = self.get_action(self.o, deterministic=False)
            print('len(ops)', len(ops))
            a, v_t, p_t = ops[0], ops[1], ops[2]
            o2, r, d, _ = self.env.step(a)

            # save and log
            buf.store(self.o, a, r, v_t, tf.math.log(p_t))

            # Update obs (critical!)
            self.o = o2

            if d: 
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else sess.run(v, feed_dict={o_ph: self.o.reshape(1,-1)})
                buf.finish_path(last_val)
                self.o = self.env.reset() # reset when done
    
    def evaluate(self, max_ep_len_eval, red=None):
        """
        Evaluate
        """
        o, d, ep_ret, ep_len = self.env.reset(red=red), False, 0, 0
        
        while not(d or (ep_len == max_ep_len_eval)):
            a = self.get_action(o, deterministic=True)
            o, r, d, _ = self.env.step(a)
            ep_ret += r # compute return 
            ep_len += 1
            
        blue_health, red_health = self.env.blue_health, self.env.red_health
        eval_res = [ep_ret, ep_len, blue_health, red_health] # evaluation result 
        return eval_res


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
    