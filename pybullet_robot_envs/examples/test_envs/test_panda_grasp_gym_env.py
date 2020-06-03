# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_robot_envs.envs.panda_envs.panda_grasp_residual_gym_env import PandaGraspResidualGymEnv
from pybullet_robot_envs.envs.panda_envs.panda_grasp_residual_gym_env_superquadric_obj import PandaGraspResidualGymEnvSqObj


import time
import math as m

def main():

    env = PandaGraspResidualGymEnvSqObj(obj_pose_rnd_std=0.07, renders=True, dset='train', n_control_pt=2, control_eu_or_quat=1)
    motorsIds = []

    dv = 0.05
    dvo = 1.57
    # motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
    # motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
    # motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
    # motorsIds.append(env._p.addUserDebugParameter("lhRollx", -dvo, dvo, 0.0))
    # motorsIds.append(env._p.addUserDebugParameter("lhPitchy", -dvo, dvo, 0.0))
    # motorsIds.append(env._p.addUserDebugParameter("lhYawz", -dvo, dvo, 0.0))
    # motorsIds.append(env._p.addUserDebugParameter("lhFingerLeft", -0.01, 0.01, 0.04))
    # motorsIds.append(env._p.addUserDebugParameter("lhFingerLeft", 0, 0.04, 0.04))


    done = False
    n_episode = 0
    while n_episode < 100:
        # env.render()
        # action = []
        # for motorId in motorsIds:
        #     action.append(env._p.readUserDebugParameter(motorId))

        action = env.action_space.sample() *0.0
        state, reward, done, _ = env.step(action)
        if done:
            env.reset()
            n_episode += 1

if __name__ == '__main__':
    main()
