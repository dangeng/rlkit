"""
Run DQN on grid world.
"""

import gym
import numpy as np

import torch
from torch import nn as nn
from grasper.seg.model.resnet import Res50Dil8
from grasper.seg.model.fcn import Interpolator

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.dqn.dqn import DQN
from rlkit.torch.networks import Mlp
import torch.nn as nn
import torch.nn.functional as F
from grasper.grasperEnv import GraspingWorld

import pdb

from rlkit.torch.core import PyTorchModule

class QFunction(PyTorchModule):
    def __init__(self):
        super(QFunction, self).__init__()
        super().__init__()
        self.save_init_params(locals())

        self.resnet = Res50Dil8()
        self.dimReduc = nn.Conv2d(2048,4,1)
        self.upscale = Interpolator(4,8,odd=False)

    def forward(self, x):
        batches = x.shape[0]
        x = self.resnet(x)
        print(x.shape)
        x = self.dimReduc(x)
        x = self.upscale(x)
        return x.view(batches, -1)

def layer_init(x):
    return x
class QFunction(PyTorchModule):    
    def __init__(self, in_channels=3, out_channels=4, feature_dim=64, num_outputs=1):
        super().__init__()
        self.save_init_params(locals())
        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=5, stride=2))

        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2,padding=1))

        self.conv3 = layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1))
        self.conv4 = layer_init(nn.Conv2d(32, out_channels, kernel_size=3, stride=1,padding=1))
        self.upscale = Interpolator(4,4,odd=False)


    def forward(self, x):
        
        batches = x.shape[0]
        #batch_size = x.shape[0]
        y = F.relu(self.conv1(x))

        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = self.conv4(y)
        y = self.upscale(y)
        #print(y.shape)
        return y.view(batches, -1)

    
def experiment(variant):
    env = GraspingWorld()
    training_env = GraspingWorld()

    qf = QFunction()

    qf_criterion = nn.MSELoss()
    # Use this to switch to DoubleDQN
    # algorithm = DoubleDQN(
    algorithm = DQN(
        env,
        training_env=training_env,
        qf=qf,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=500,
            num_steps_per_epoch=100,
            num_steps_per_eval=10,
            batch_size=4,
            max_path_length=20,
            discount=0.5,
            epsilon=0.2,
            tau=0.001,
            hard_update_period=100,
            #min_num_steps_before_training=10,
            replay_buffer_size=100,
            save_environment=False,  # Can't serialize CartPole for some reason
        ),
    )
    ptu.set_gpu_mode(True, 1)
    setup_logger('dqn_grasper', variant=variant)
    experiment(variant)
