# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import pybullet as p

from pybullet_robot_envs.envs.world_envs.world_env import WorldEnv
from pybullet_object_models import ycb_objects


def get_ycb_objects_list():
    obj_list = [
        'YcbMustardBottle',
        'YcbTomatoSoupCan',
        'YcbCrackerBox',
        'YcbChipsCan',
        'YcbBanana',
        'YcbFoamBrick',
        'YcbGelatinBox',
        'YcbHammer',
        'YcbMasterChefCan',
        'YcbMediumClamp',
        'YcbPear',
        'YcbPottedMeatCan',
        'YcbPowerDrill',
        'YcbScissors',
        'YcbStrawberry',
        'YcbTennisBall',
    ]

    return obj_list


class YcbWorldFetchEnv(WorldEnv):

    def __init__(self,
                 physicsClientId,
                 obj_name='YcbMustardBottle',
                 obj_pose_rnd_std=0.05,
                 workspace_lim=None,
                 control_eu_or_quat=0):

        super(WorldEnv, self).__init__(physicsClientId, obj_name, obj_pose_rnd_std, workspace_lim, control_eu_or_quat)

    def load_object(self, obj_name):
        # Load object. Randomize its start position if requested
        self._obj_init_pose = self._sample_pose()
        self.obj_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), obj_name, "model.urdf"),
                                 basePosition=self._obj_init_pose[:3], baseOrientation=self._obj_init_pose[3:7],
                                 physicsClientId=self._physics_client_id)
