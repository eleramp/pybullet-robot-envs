# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
from icub_model_pybullet import model_with_hands
from pybullet_robot_envs.envs.icub_envs.icub_env import iCubEnv

import numpy as np
import quaternion
import math as m
import time


class iCubHandsEnv(iCubEnv):

    def __init__(self, use_IK=0, control_arm='l', control_orientation=0, control_eu_or_quat=0):

        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._use_simulation = 1

        self._indices_torso = range(12, 15)
        self._indices_left_arm = range(15, 22)
        self._indices_left_hand = range(22, 42)
        self._indices_right_arm = range(45, 52)
        self._indices_right_hand = range(52, 72)
        self._indices_head = range(42, 45)
        self._end_eff_idx = []

        self._home_pos_torso = [0.0, 0.0, 0.0]  # degrees
        self._home_pos_head = [0.47, 0, 0]

        self._home_left_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self._home_right_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]

        self._home_left_hand = [0] * len(self._indices_left_hand)
        self._home_right_hand = [0] * len(self._indices_right_hand)

        self._home_hand_pose = []
        self._home_motor_pose = []

        self._grasp_pos = [0, 0.4, 0.35, 0.05, 0, 0.35, 0.4, 0.05, 0, 0.35, 0.4, 0.05, 0, 0.35, 0.4, 0.05, 1.57, 0.6, 0.1, 0.05]

        self._workspace_lim = [[0.25, 0.52], [-0.3, 0.3], [0.5, 1.0]]
        self._eu_lim = [[-m.pi/2, m.pi/2], [-m.pi/2, m.pi/2], [-m.pi, m.pi]]

        self._control_arm = control_arm if control_arm == 'r' or control_arm == 'l' else 'l'  # left arm by default

        self.robot_id = None

        # Initialize base class
        # super().__init__(use_IK=0, control_arm='l', control_orientation=0)

        self.reset()

    def reset(self):

        self.robot_id = p.loadSDF(os.path.join(model_with_hands.get_data_path(), "icub_model_with_hands.sdf"))[0]
        assert self.robot_id is not None, "Failed to load the icub model"

        self._num_joints = p.getNumJoints(self.robot_id)

        # set constraint between base_link and world
        constr_id = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                       parentFramePosition=[0, 0, 0],
                                       childFramePosition=[p.getBasePositionAndOrientation(self.robot_id)[0][0],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][1],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][2] * 1.2],
                                       parentFrameOrientation=p.getBasePositionAndOrientation(self.robot_id)[1])

        # Set all joints initial values
        for count, i in enumerate(self._indices_torso):
            p.resetJointState(self.robot_id, i, self._home_pos_torso[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_head):
            p.resetJointState(self.robot_id, i, self._home_pos_head[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_left_arm):
            p.resetJointState(self.robot_id, i, self._home_left_arm[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_right_arm):
            p.resetJointState(self.robot_id, i, self._home_right_arm[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_left_hand):
            p.resetJointState(self.robot_id, i, self._home_left_hand[count] / 180 * m.pi)

        for count, i in enumerate(self._indices_right_hand):
            p.resetJointState(self.robot_id, i, self._home_right_hand[count] / 180 * m.pi)

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        # save indices of only the joints to control
        control_arm_indices = list(self._indices_left_arm) + list(self._indices_left_hand) if self._control_arm == 'l' \
            else list(self._indices_right_arm) + list(self._indices_right_hand)

        control_arm_poses = self._home_left_arm + self._home_left_hand if self._control_arm == 'l' else \
            self._home_right_arm + self._home_right_hand

        self._motor_idxs = [i for i in self._indices_torso] + [j for j in control_arm_indices]
        self._end_eff_idx = self._indices_left_arm[-1] if self._control_arm == 'l' else self._indices_right_arm[-1]

        self._joints_to_block = list(self._indices_left_arm) if self._control_arm == 'r' else list(self._indices_right_arm)
        self._joints_to_block += list(self._indices_left_hand) + list(self._indices_right_hand)

        self._home_motor_pose = self._home_pos_torso + control_arm_poses

        self._motor_names = []
        for i in self._indices_torso:
            jointInfo = p.getJointInfo(self.robot_id, i)
            if jointInfo[3] > -1:
                self._motor_names.append(str(jointInfo[1]))
        for i in control_arm_indices:
            jointInfo = p.getJointInfo(self.robot_id, i)
            if jointInfo[3] > -1:
                self._motor_names.append(str(jointInfo[1]))

        # set initial hand pose
        if self._control_arm == 'l':
            self._home_hand_pose = [0.2, 0.26, 0.85, 0, 0, m.pi/2]  # x,y,z, roll,pitch,yaw
            self._eu_lim = [[-m.pi/2, m.pi/2], [-m.pi/2, m.pi/2], [0, m.pi]]
        else:
            self._home_hand_pose = [0.2, -0.26, 0.85, 0, 0, m.pi/2]
            self._eu_lim = [[-m.pi / 2, m.pi / 2], [-m.pi / 2, m.pi / 2], [0, m.pi]]

        # self.eu_lim[0] = np.add(self.eu_lim[0], self.home_hand_pose[3])
        # self.eu_lim[1] = np.add(self.eu_lim[1], self.home_hand_pose[4])
        # self.eu_lim[2] = np.add(self.eu_lim[2], self.home_hand_pose[5])

        if self._use_IK:
            self.apply_action(self._home_hand_pose[:3])

    def _com_to_link_hand_frame(self):
        if self._control_arm is 'r':
            com_T_link_hand = (-0.011682, 0.051682, -0.000577)
        else:
            com_T_link_hand = (-0.011682, 0.051355, 0.000577)

        return com_T_link_hand

    def get_finger_joints_poses(self):
        joint_states = p.getJointStates(self.robot_id, self._motor_idxs[-20:])
        joint_poses = [x[0] for x in joint_states]
        return joint_poses

    def open_hand(self):
        # open fingers
        pos = [0, 0.6, 0.5, 0.8, 0, 0.6, 0.5, 0.8, 0, 0.6, 0.5, 0.9, 0, 0.6, 0.5, 0.8, 1.57, 0.6, 0.4, 0.7]
        pos = [0]*20

        if self._control_arm is 'l':
            idx_thumb = self._indices_left_hand[-4]
        else:
            idx_thumb = self._indices_right_hand[-4]

        steps = [i / 100 for i in range(100, -1, -1)]
        for s in steps:
            next_pos = np.multiply(pos, s)
            p.setJointMotorControlArray(self.robot_id, range(52, 72), p.POSITION_CONTROL, targetPositions=next_pos,
                                        forces=[20] * len(range(52, 72)))
            p.setJointMotorControl2(self.robot_id, idx_thumb, p.POSITION_CONTROL, targetPosition=1.57, force=50)
            for _ in range(4):
                p.stepSimulation()

    def pre_grasp(self):
        # move fingers to pre-grasp configuration
        if self._control_arm is 'l':
            idx = self._indices_left_hand[-4]
        else:
            idx = self._indices_right_hand[-4]

        p.resetJointState(self.robot_id, idx, 1.57)
        p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=1.57, force=50)

    def grasp(self, step):
        # close fingers
        if self._control_arm is 'l':
            idx_thumb = self._indices_left_hand[-4]
        else:
            idx_thumb = self._indices_right_hand[-4]

        joint_states = p.getJointStates(self.robot_id, self._motor_idxs[-20:])
        joint_poses = [x[0] for x in joint_states]
        p.setJointMotorControlArray(self.robot_id, self._motor_idxs[-20:], p.POSITION_CONTROL, targetPositions=joint_poses,
                                    forces=[50] * len(self._motor_idxs[-20:]))

        next_pos = np.multiply(self._grasp_pos, step)
        p.setJointMotorControlArray(self.robot_id, range(52, 72), p.POSITION_CONTROL, targetPositions=next_pos,
                                    forces=[50] * len(range(52, 72)))
        p.setJointMotorControl2(self.robot_id, idx_thumb, p.POSITION_CONTROL, targetPosition=1.57, force=50)

    def checkContacts(self, obj_id):
        points = p.getContactPoints(self.robot_id, obj_id)
        if len(points) > 0:
            print("contacts! {}".format(len(points)))
            return True

        return False
