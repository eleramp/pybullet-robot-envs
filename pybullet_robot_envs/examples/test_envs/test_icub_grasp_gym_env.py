# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

#!/usr/bin/env python
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, '/home/erampone/workspace/INSTALL/lib/superquadriclib/bindings')

print(os.sys.path)

from pybullet_robot_envs.envs.icub_envs.icub_reach_gym_env import iCubReachGymEnv
from pybullet_robot_envs.envs.icub_envs.icub_grasp_residual_gym_env import iCubGraspResidualGymEnv
from pybullet_robot_envs.envs.icub_envs.icub_grasp_residual_gym_env_1 import iCubGraspResidualGymEnv1
from pybullet_robot_envs.envs.icub_envs.icub_reach_residual_gym_env import iCubReachResidualGymEnv
from pybullet_robot_envs.envs.icub_envs.icub_reach_grasp_residual_gym_env import iCubReachGraspResidualGymEnv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--continueIK', action='store_const', const=1, dest="useIK",
                    help='use continue Inverse Kinematic action')
parser.add_argument('--arm', action='store', default='l', dest="arm",
                    help="choose arm to control: 'l' - left or 'r'-right")

import numpy as np

def main(args):
    eu_or_quat = 0
    env = iCubReachGraspResidualGymEnv(renders=True, control_arm='r', obj_pose_rnd_std=0.05, noise_pcl=0.00,
                                      control_eu_or_quat=eu_or_quat, obj_name=2)

    env.seed(1)
    motorsIds = []

    dv = 1
    if eu_or_quat is 0:
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhRollx", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPitchy", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhYawz", -dv, dv, 0.0))
        #motorsIds.append(env._p.addUserDebugParameter("close_open", -dv, dv, 0.0))
    else:
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("qx", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("qy", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("qz", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("qw", -dv, dv, 0.0))

    done = False
    for t in range(10000000):
        # env.render()
        action = [] #[0.0]*7
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))

        state, reward, done, _ = env.step(action)
        if done:
            env.reset()
        if t % 100 == 0:
            print("reward ", reward)


if __name__ == '__main__':
    main(parser.parse_args())
