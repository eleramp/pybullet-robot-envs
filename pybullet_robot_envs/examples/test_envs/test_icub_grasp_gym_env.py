#!/usr/bin/env python
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

print(os.sys.path)

from pybullet_robot_envs.envs.icub_envs.icub_grasp_residual_gym_env import iCubGraspResidualGymEnv
from pybullet_robot_envs import robot_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--continueIK', action='store_const', const=1, dest="useIK",
                    help='use continue Inverse Kinematic action')
parser.add_argument('--arm', action='store', default='l', dest="arm",
                    help="choose arm to control: 'l' - left or 'r'-right")

import pymesh
import open3d as o3d
import numpy as np

def main(args):

    env = iCubGraspResidualGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, control_arm='l', useOrientation=1, rnd_obj_pose=0.05,  noise_pcl=0.005)
    env.seed(1)
    motorsIds = []

    dv = 1
    #motorsIds.append(env._p.addUserDebugParameter("lhPosX", -dv, dv, 0.0))
    #motorsIds.append(env._p.addUserDebugParameter("lhPosY", -dv, dv, 0.0))
    #motorsIds.append(env._p.addUserDebugParameter("lhPosZ", -dv, dv, 0.0))
    #motorsIds.append(env._p.addUserDebugParameter("lhRollx", -dv, dv, 0.0))
    #motorsIds.append(env._p.addUserDebugParameter("lhPitchy", -dv, dv, 0.0))
    #motorsIds.append(env._p.addUserDebugParameter("lhYawz", -dv, dv, 0.0))
    #motorsIds.append(env._p.addUserDebugParameter("close_open", -dv, dv, 0.0))

    done = False
    #env._p.addUserDebugText('current hand position', [0, -0.5, 1.4], [1.1, 0, 0])
    #idx = env._p.addUserDebugText(' ', [0, -0.5, 1.2], [1, 0, 0])

    for t in range(10000000):
        #env.render()
        action = [0.0]*7
        #for motorId in motorsIds:
            #action.append(env._p.readUserDebugParameter(motorId))

        state, reward, done, _ = env.step(action)
        if done:
            env.reset()
        if t % 100 == 0:
            print("reward ", reward)
            #env._p.addUserDebugText(' '.join(str(round(e, 2)) for e in state[:6]), [0, -0.5, 1.2], [1, 0, 0], replaceItemUniqueId=idx)


if __name__ == '__main__':
    main(parser.parse_args())
