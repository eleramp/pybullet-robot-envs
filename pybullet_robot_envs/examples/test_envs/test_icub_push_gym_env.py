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

from pybullet_robot_envs.envs.icub_envs.icub_push_gym_env import iCubPushGymEnv
from pybullet_robot_envs.envs.icub_envs.icub_push_gym_goal_env import iCubPushGymGoalEnv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--continueIK', action='store_const', const=1, dest="useIK",
                    help='use continue Inverse Kinematic action')
parser.add_argument('--arm', action='store', default='r', dest="arm",
                    help="choose arm to control: 'l' - left or 'r'-right")

def main(args):

    use_IK = 1 # if args.useIK else 0

    env = iCubPushGymEnv(renders=True, control_arm=args.arm, use_IK=use_IK,
                         discrete_action=0, control_orientation=1, obj_pose_rnd_std=0.05)
    motorsIds = []

    if (env._discrete_action):
        dv = 12
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0))
    elif use_IK:
        dv = 1
        motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhRollx", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhPitchy", -dv, dv, 0.0))
        motorsIds.append(env._p.addUserDebugParameter("lhYawz", -dv, dv, 0.0))

    else:
        dv = 1
        joints_idx = env._robot._motor_idxs

        for j in joints_idx:
            info = env._p.getJointInfo(env._robot.robot_id, j)
            jointName = info[1]
            motorsIds.append(env._p.addUserDebugParameter(jointName.decode("utf-8"), -dv, dv, 0.0))

    done = False
    for t in range(10000000):
        #env.render()
        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))

        action = int(action[0]) if env._discrete_action else action

        state, reward, done, info = env.step(action)
        if t%100==0:
            print("reward ", reward)
            print("done ", done)

if __name__ == '__main__':
    main(parser.parse_args())
