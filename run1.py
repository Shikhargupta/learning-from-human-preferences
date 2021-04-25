
'''
python3 run1.py train_policy_with_preferences MovingDotDiscreteNoFrameskip-v0 --n_envs 16 --render_episodes --load_prefs_dir runs/moving-dot_45cb953'''

import logging
import os
from os import path as osp
import sys
import time
from multiprocessing import Process, Queue

import cloudpickle
import easy_tf_log
from a2c import logger
from a2c.a2c.a2c import learn
from a2c.a2c.policies import CnnPolicy, MlpPolicy
from a2c.common import set_global_seeds
from a2c.common.vec_env.subproc_vec_env import SubprocVecEnv
from params import parse_args, PREFS_VAL_FRACTION
from pref_db import PrefDB, PrefBuffer
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble
from reward_predictor_core_network import net_cnn, net_moving_dot_features, gp_rp
from utils import VideoRenderer, get_port_range, make_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages

def start_episode_renderer():
    episode_vid_queue = Queue()
    renderer = VideoRenderer(
        episode_vid_queue,
        playback_speed=2,
        zoom=2,
        mode=VideoRenderer.play_through_mode)
    return episode_vid_queue, renderer

def configure_a2c_logger(log_dir):
    a2c_dir = osp.join(log_dir, 'a2c')
    os.makedirs(a2c_dir)
    tb = logger.TensorBoardOutputFormat(a2c_dir)
    logger.Logger.CURRENT = logger.Logger(dir=a2c_dir, output_formats=[tb])

def make_envs(env_id, n_envs, seed):
    def wrap_make_env(env_id, rank):
        def _thunk():
            return make_env(env_id, seed + rank)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv(env_id, [wrap_make_env(env_id, i)
                                 for i in range(n_envs)])
    return env


general_params, a2c_params, \
    pref_interface_params, rew_pred_training_params = parse_args()

if general_params['render_episodes']:
    episode_vid_queue, episode_renderer = start_episode_renderer()
else:
    episode_vid_queue = episode_renderer = None

if general_params['debug']:
    logging.getLogger().setLevel(logging.DEBUG)

seg_pipe = Queue(maxsize=1)
pref_pipe = Queue(maxsize=1)
start_policy_training_flag = Queue(maxsize=1)




def make_train_reward_predictor():
    prefs_dir=general_params['prefs_dir']
    train_path = osp.join(prefs_dir, 'train.pkl.gz')
    val_path = osp.join(prefs_dir, 'val.pkl.gz')
    pref_db_train = PrefDB.load(train_path)
    pref_db_val = PrefDB.load(val_path)
    pref_buffer = PrefBuffer(db_train=pref_db_train,
                         db_val=pref_db_val)

    pref_db_train, pref_db_val = pref_buffer.get_dbs()

    gauss_rp = gp_rp()
    gauss_rp.train(pref_db_train)

# make_train_reward_predictor()

def create_cluster_dict(jobs):
    n_ports = len(jobs) + 1
    ports = get_port_range(start_port=2200,
                           n_ports=n_ports,
                           random_stagger=True)
    cluster_dict = {}
    for part, port in zip(jobs, ports):
        cluster_dict[part] = ['localhost:{}'.format(port)]
    return cluster_dict

def start_policy_training(cluster_dict, gen_segments,
                          start_policy_training_pipe, seg_pipe,
                          episode_vid_queue, log_dir, a2c_params):
    env_id = a2c_params['env_id']
    if env_id in ['MovingDotDiscreteNoFrameskip-v0', 'MovingDot-v0']:
        policy_fn = MlpPolicy
    elif env_id in ['PongNoFrameskip-v4', 'EnduroNoFrameskip-v4']:
        policy_fn = CnnPolicy
    else:
        msg = "Unsure about policy network for {}".format(a2c_params['env_id'])
        raise Exception(msg)

    configure_a2c_logger(log_dir)

    # Done here because daemonic processes can't have children
    env = make_envs(a2c_params['env_id'],
                    a2c_params['n_envs'],
                    a2c_params['seed'])
    del a2c_params['env_id'], a2c_params['n_envs']

    ckpt_dir = osp.join(log_dir, 'policy_checkpoints')
    os.makedirs(ckpt_dir)

    def f():
        print("training!!!!!!!!!!!!!!!")
        reward_predictor = make_train_reward_predictor()
        print("Trained rp!!!!!!!!!")
        misc_logs_dir = osp.join(log_dir, 'a2c_misc')
        easy_tf_log.set_dir(misc_logs_dir)
        learn(
            policy=policy_fn,
            env=env,
            seg_pipe=seg_pipe,
            start_policy_training_pipe=start_policy_training_pipe,
            episode_vid_queue=episode_vid_queue,
            reward_predictor=reward_predictor,
            ckpt_save_dir=ckpt_dir,
            gen_segments=gen_segments,
            **a2c_params)

    proc = Process(target=f, daemon=True)
    proc.start()
    return env, proc

env, a2c_proc = start_policy_training(
    cluster_dict=None,
    gen_segments=False,
    start_policy_training_pipe=start_policy_training_flag,
    seg_pipe=seg_pipe,
    episode_vid_queue=episode_vid_queue,
    log_dir=general_params['log_dir'],
    a2c_params=a2c_params)
start_policy_training_flag.put(True)
a2c_proc.join()
env.close()
