from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, inspect
import ray
import gym
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune import run_experiments

import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
import baselines.common.tf_util as U
import datetime
from baselines import logger
import os.path as osp
import tensorflow as tf
import random
import sys

import numpy as np

def getGibsonEnv():
    rank = MPI.COMM_WORLD.Get_rank()
    config_file = os.path.join('/root/mount/gibson/examples/train/', '..', 'configs', 'husky_navigate_rgb_train.yaml')
    print(config_file)
    env = HuskyNavigateEnv(gpu_idx=0,
                               config=config_file)
    env.reset()
    return env

ray.init()
env_name = "test"
register_env(env_name, lambda _ : getGibsonEnv())

config = ppo.DEFAULT_CONFIG.copy()
config.update({
    "model": {
        "conv_filters": [
            [32, [8, 8], 4],
            [64, [4, 4], 2],
            [64, [10, 10], 8],
        ],
    },
    "num_workers": 1,
    "train_batch_size": 2000,
    "sample_batch_size": 100,
    "lambda": 0.95,
    "clip_param": 0.2,
    "num_sgd_iter": 20,
    "lr": 0.0001,
    "sgd_minibatch_size": 32,
    "num_gpus": 1,
    'use_gae': True,
    'horizon': 4096,
    'kl_coeff': 0.0,
    'vf_loss_coeff': 0.0,
    'entropy_coeff': 0.0,
    'tf_session_args': {
        'gpu_options': {'allow_growth': True}
    }
})

alg = ppo.PPOAgent(config=config, env=env_name)
alg.train()
