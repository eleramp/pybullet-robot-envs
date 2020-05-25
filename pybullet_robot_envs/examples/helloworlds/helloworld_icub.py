# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import pybullet_data
from icub_model_pybullet import model_with_hands
from pybullet_object_models import ycb_objects

import time
import math as m

joint_groups = {'torso': ['torso_pitch', 'torso_roll', 'torso_yaw'],
                'r_arm': ['r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
                          'r_elbow', 'r_wrist_pitch', 'r_wrist_prosup', 'r_wrist_yaw'],
                'r_hand': ['r_hand::r_aij6', 'r_hand::r_aij3', 'r_hand::r_aij4', 'r_hand::r_aij5',
                           'r_hand::r_lij6', 'r_hand::r_lij3', 'r_hand::r_lij4', 'r_hand::r_lij5',
                           'r_hand::r_mj6', 'r_hand::r_mj3', 'r_hand::r_mj4', 'r_hand::r_mj5',
                           'r_hand::r_rij6', 'r_hand::r_rij3', 'r_hand::r_rij4', 'r_hand::r_rij5',
                           'r_hand::r_tj2', 'r_hand::r_tj4', 'r_hand::r_tj5', 'r_hand::r_tj6']
                }


def main():

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1.8, 120, -50, [0.0, -0.0, -0.0])
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(1/240.)

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    # Set gravity for simulation
    p.setGravity(0, 0, -9.8)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # load robot model
    icubId = p.loadSDF(os.path.join(model_with_hands.get_data_path(), "icub_model_with_hands.sdf"))[0]

    # set constraint between base_link and world
    p.createConstraint(icubId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[p.getBasePositionAndOrientation(icubId)[0][0],
                                           p.getBasePositionAndOrientation(icubId)[0][1],
                                           p.getBasePositionAndOrientation(icubId)[0][2] * 1.2],
                       parentFrameOrientation=p.getBasePositionAndOrientation(icubId)[1])

    # set starting pose for standing
    home_pos_torso = [0.0, 0.0, 0.0]  # degrees
    home_pos_head = [0.47, 0, 0]
    home_left_arm = [-29.4, 28.8, 0, 44.59, 0, 0, 0]
    home_right_arm = [-29.4, 30.4, 0, 44.59, 0, 0, 0]
    home_left_hand = [0] * 20
    home_right_hand = [0] * 20

    init_pos = [0.0]*12 + home_pos_torso + home_left_arm + home_left_hand + home_pos_head + home_right_arm + home_right_hand

    # reset joint position to starting pose
    joint_name_to_ids = {}
    joints_num = p.getNumJoints(icubId)

    idx = 0
    for i in range(joints_num):
        jointInfo = p.getJointInfo(icubId, i)
        jointName = jointInfo[1].decode("UTF-8")
        jointType = jointInfo[2]

        if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
            p.resetJointState(icubId, i, init_pos[idx]/180*m.pi)
            joint_name_to_ids[jointName] = i
            idx += 1

    # set end-effector index
    hand_idx = joint_name_to_ids['r_wrist_yaw']

    p.stepSimulation()

    # -------------------------- #
    # --- Load other objects --- #
    # -------------------------- #

    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])
    p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbFoamBrick', "model.urdf"), [0.5, -0.03, 0.7])

    # Run the world for a bit
    for _ in range(100):
        p.stepSimulation()

    # ------------------ #
    # --- Start Demo --- #
    # ------------------ #

    # pre-grasp configuration of the hand
    idxs_pregrasp = [joint_name_to_ids['r_wrist_pitch'], joint_name_to_ids['r_hand::r_tj2']]

    p.setJointMotorControlArray(icubId, idxs_pregrasp, p.POSITION_CONTROL, targetPositions=[1, 1.57], forces=[50, 50])
    for _ in range(10):
        p.stepSimulation()
        time.sleep(1/240)

    # 1: go above the object
    pos_1 = [0.49, 0.0, 0.8]
    quat_1 = p.getQuaternionFromEuler([0, 0, m.pi/2])

    jointPoses = p.calculateInverseKinematics(icubId, hand_idx, pos_1, quat_1)

    p.setJointMotorControlArray(icubId, joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=jointPoses)
    p.setJointMotorControlArray(icubId, idxs_pregrasp, p.POSITION_CONTROL, targetPositions=[1, 1.57])

    for _ in range(60):
        p.stepSimulation()
        time.sleep(1 / 240)

    # 2: turn hand above the object
    pos_2 = [0.485, 0.0, 0.75]
    quat_2 = p.getQuaternionFromEuler([0, m.pi/2, m.pi/2])

    jointPoses = p.calculateInverseKinematics(icubId, hand_idx, pos_2, quat_2)

    p.setJointMotorControlArray(icubId, joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=jointPoses,
                                forces=[50] * len(joint_name_to_ids.values()))

    p.setJointMotorControlArray(icubId, idxs_pregrasp, p.POSITION_CONTROL, targetPositions=[1, 1.57])

    for _ in range(40):
        p.stepSimulation()
        time.sleep(1/240)

    # 3: close fingers
    pos_3 = [0, 0.3, 0.5, 0.9, 0,  0.3, 0.5, 0.9, 0,  0.3, 0.5, 0.9, 0,  0.3, 0.5, 0.9, 1.57, 0.6, 0.4, 0.7]
    idx_fingers = [joint_name_to_ids[jn] for jn in joint_groups['r_hand']]

    p.setJointMotorControlArray(icubId, joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=jointPoses)
    p.setJointMotorControlArray(icubId, idx_fingers, p.POSITION_CONTROL, targetPositions=pos_3)
    p.setJointMotorControlArray(icubId, idxs_pregrasp, p.POSITION_CONTROL, targetPositions=[1, 1.57])

    for _ in range(60):
        p.stepSimulation()
        time.sleep(1/240)

    # 4: go up
    pos_4 = [0.45, 0, 0.9]
    quat_4 = p.getQuaternionFromEuler([0, m.pi/2, m.pi/2])

    jointPoses = list(p.calculateInverseKinematics(icubId, hand_idx, pos_4, quat_4))
    jointPoses[-len(idx_fingers):] = pos_3

    p.setJointMotorControlArray(icubId, joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=jointPoses)

    for _ in range(60):
        p.stepSimulation()
        time.sleep(1/240)

    # 5: go right
    pos_5 = [0.3, -0.2, 0.9]
    quat_5 = p.getQuaternionFromEuler([0, 0, m.pi/2])

    jointPoses = list(p.calculateInverseKinematics(icubId, hand_idx, pos_5, quat_5))
    jointPoses[-len(idx_fingers):] = pos_3

    p.setJointMotorControlArray(icubId, joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=jointPoses)

    for _ in range(60):
        p.stepSimulation()
        time.sleep(1/240)

    # 6: open fingers
    pos_6 = [0.0] * len(idx_fingers)
    for i, idx in enumerate(idx_fingers):
        if idx == joint_name_to_ids['r_hand::r_tj2']:
            pos_6[i] = 1.57

    p.setJointMotorControlArray(icubId, joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=jointPoses)
    p.setJointMotorControlArray(icubId, idx_fingers, p.POSITION_CONTROL, targetPositions=pos_6)

    for _ in range(50):
        p.stepSimulation()
        time.sleep(1/240)

    jointPoses[-len(idx_fingers):] = pos_6

    # ------------------------ #
    # --- Play with joints --- #
    # ------------------------ #

    paramIds = []
    num_joints = p.getNumJoints(icubId)
    idx = 0
    for i in range(num_joints):
        jointInfo = p.getJointInfo(icubId, i)
        jointName = jointInfo[1]
        jointType = jointInfo[2]

        if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
            paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), jointInfo[8], jointInfo[9], jointPoses[idx]))
            idx += 1

    while True:
        new_pos = []
        for i in paramIds:
            new_pos.append(p.readUserDebugParameter(i))
        p.setJointMotorControlArray(icubId, joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=new_pos)

        p.stepSimulation()
        time.sleep(0.01)


if __name__ == '__main__':
    main()
