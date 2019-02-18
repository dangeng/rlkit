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
        x = self.resnet(x)
        x = self.dimReduc(x)
        x = self.upscale(x)
        return x.flatten()

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
        replay_buffer_size=10,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            epsilon=0.2,
            tau=0.001,
            hard_update_period=1000,
            save_environment=False,  # Can't serialize CartPole for some reason
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
