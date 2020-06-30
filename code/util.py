import os,time
import numpy as np
import tensorflow as tf

from nn import discount_cumsum

    
def suppress_tf_warning():
    import tensorflow as tf
    import os
    import logging
    from tensorflow.python.util import deprecation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
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
        
        
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, odim, adim, size=5000, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, odim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, adim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


def gpu_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

t_start_tictoc = time.time()
def tic():
    global t_start_tictoc
    t_start_tictoc = time.time()
    
def toc(toc_str=None):
    global t_start_tictoc
    t_elapsed_sec = time.time() - t_start_tictoc
    if toc_str is None:
        print ("Elapsed time is [%.4f]sec."%
        (t_elapsed_sec))
    else:
        print ("[%s] Elapsed time is [%.4f]sec."%
        (toc_str,t_elapsed_sec))
        
        
def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])
    
    
def save_ppo_model_and_buffer(npz_path,R,buffer,VERBOSE=True):
    """
    Save SAC model weights and replay buffers
    """
    
    # PPO model
    tf_vars = R.model['main_vars']
    data2save,var_names,var_vals = dict(),[],[]
    for v_idx,tf_var in enumerate(tf_vars):
        var_name,var_val = tf_var.name,R.sess.run(tf_var)
        var_names.append(var_name)
        var_vals.append(var_val)
        data2save[var_name] = var_val
        if VERBOSE:
            print ("[%02d]  var_name:[%s]  var_shape:%s"%
                (v_idx,var_name,var_val.shape,)) 
            
    # Buffers
    names,vals = buffer.get()
    for name,val in zip(names,vals):
        data2save[name] = val
    
    # Create folder if not exist
    dir_name = os.path.dirname(npz_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print ("[%s] created."%(dir_name))
        
    # Save npz
    np.savez(npz_path,**data2save)
    print ("[%s] saved."%(npz_path))
            
            
def restore_ppo_model_and_buffer(npz_path,R,buffer,VERBOSE=True):
    """
    Restore PPO model weights and replay buffers
    """
    
    # Load npz
    l = np.load(npz_path)
    print ("[%s] loaded."%(npz_path))
    
    # Get values of SAC model  
    tf_vars = R.model['main_vars']
    var_vals = []
    for tf_var in tf_vars:
        var_vals.append(l[tf_var.name])   
        
    # Assign weights of SAC model
    R.set_weights(var_vals)
    
    # Restore buffers
    buffer_names,_ = buffer.get()
    a = []
    for buffer_name in buffer_names:
        a.append(l[buffer_name])
    buffer.restore(a)
    
    
def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = (np.min(x) if len(x) > 0 else np.inf)
        global_max = (np.max(x) if len(x) > 0 else -np.inf)
        return mean, std, global_min, global_max
    return mean, std
