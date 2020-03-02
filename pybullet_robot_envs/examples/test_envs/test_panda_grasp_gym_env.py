# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_robot_envs.envs.panda_envs.panda_reach_residual_gym_env import PandaReachResidualGymEnv
from pybullet_robot_envs.envs.panda_envs.panda_grasp_residual_gym_env import PandaGraspResidualGymEnv
import pybullet_data


import time
import math as m

def main():

    env = PandaGraspResidualGymEnv(obj_pose_rnd_std=0.0, renders=True)
    motorsIds = []

    dv = 1
    motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
    motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
    motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
    motorsIds.append(env._p.addUserDebugParameter("lhRollx", -dv, dv, 0.0))
    motorsIds.append(env._p.addUserDebugParameter("lhPitchy", -dv, dv, 0.0))
    motorsIds.append(env._p.addUserDebugParameter("lhYawz", -dv, dv, 0.0))
    motorsIds.append(env._p.addUserDebugParameter("lhFingerLeft", 0, 0.04, 0.04))
    motorsIds.append(env._p.addUserDebugParameter("lhFingerLeft", 0, 0.04, 0.04))


    done = False

    for t in range(10000000):
        #env.render()
        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))

        action = int(action[0])

        #print(env.step(action))

        state, reward, done, _ = env.step(action)
        if t%10==0:
            print("reward ", reward)
            print("done ", done)
        if done:
            env.reset()

if __name__ == '__main__':
    main()
