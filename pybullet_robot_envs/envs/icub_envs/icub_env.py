# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import icub_model_pybullet
from gym.utils import seeding

import numpy as np
import quaternion
import math as m
import time



class iCubEnv:

    def __init__(self, physicsClientId, use_IK=0, control_arm='l', control_orientation=0, control_eu_or_quat=0):

        self._physics_client_id = physicsClientId
        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._use_simulation = 1

        self._indices_torso = range(12, 15)
        self._indices_left_arm = range(15, 22)
        self._indices_right_arm = range(25, 32)
        self._indices_head = range(22, 25)
        self._end_eff_idx = []

        self._home_pos_torso = [0.0, 0.0, 0.0]  # degrees
        self._home_pos_head = [0.47, 0, 0]

        self._home_left_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]
        self._home_right_arm = [-29.4, 40.0, 0, 70, 0, 0, 0]

        self._home_hand_pose = []

        self._workspace_lim = [[0.25, 0.52], [-0.3, 0.3], [0.5, 1.0]]
        self._eu_lim = [[-m.pi/2, m.pi/2], [-m.pi/2, m.pi/2], [-m.pi, m.pi]]

        self._control_arm = control_arm if control_arm == 'r' or control_arm == 'l' else 'l'  # left arm by default

        self.robot_id = None

        self._num_joints = 0
        self._motor_idxs = 0
        self.ll, self.ul, self.jr, self.rs = None, None, None, None

        self.seed()
        self.reset()

    def reset(self):

        self.robot_id = p.loadSDF(os.path.join(icub_model_pybullet.get_data_path(), "icub_model.sdf"),
                                  physicsClientId=self._physics_client_id)[0]

        assert self.robot_id is not None, "Failed to load the icub model"

        self._num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)

        # set constraint between base_link and world
        constr_id = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0],
                                       parentFramePosition=[0, 0, 0],
                                       childFramePosition=[p.getBasePositionAndOrientation(self.robot_id)[0][0],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][1],
                                                           p.getBasePositionAndOrientation(self.robot_id)[0][2] * 1.2],
                                       parentFrameOrientation=p.getBasePositionAndOrientation(self.robot_id)[1],
                                       physicsClientId=self._physics_client_id)

        # Set all joints initial values
        for count, i in enumerate(self._indices_torso):
            p.resetJointState(self.robot_id, i, self._home_pos_torso[count] / 180 * m.pi,
                              physicsClientId=self._physics_client_id)

        for count, i in enumerate(self._indices_head):
            p.resetJointState(self.robot_id, i, self._home_pos_head[count] / 180 * m.pi,
                              physicsClientId=self._physics_client_id)

        for count, i in enumerate(self._indices_left_arm):
            p.resetJointState(self.robot_id, i, self._home_left_arm[count] / 180 * m.pi,
                              physicsClientId=self._physics_client_id)

        for count, i in enumerate(self._indices_right_arm):
            p.resetJointState(self.robot_id, i, self._home_right_arm[count] / 180 * m.pi,
                              physicsClientId=self._physics_client_id)

        # save indices of only the joints to control
        control_arm_indices = self._indices_left_arm if self._control_arm == 'l' else self._indices_right_arm
        self._motor_idxs = list(self._indices_torso) + list(control_arm_indices)

        self._end_eff_idx = self._indices_left_arm[-1] if self._control_arm == 'l' else self._indices_right_arm[-1]

        self._joints_to_block = list(self._indices_left_arm) if self._control_arm == 'r' else list(self._indices_right_arm)

        self._motor_names = []
        for i in self._indices_torso:
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            if jointInfo[3] > -1:
                self._motor_names.append(str(jointInfo[1]))
        for i in control_arm_indices:
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            if jointInfo[3] > -1:
                self._motor_names.append(str(jointInfo[1]))

        self.ll, self.ul, self.jr, self.rs, self.jd = self.get_joint_ranges()

        # set initial hand pose
        if self._control_arm == 'l':
            self._home_hand_pose = [0.3, 0.26, 0.8, 0, 0, 0]  # x,y,z, roll,pitch,yaw
        else:
            self._home_hand_pose = [0.3, -0.26, 0.8, 0, 0, m.pi]

        if self._use_IK:
            self.apply_action(self._home_hand_pose)

    def delete_simulated_robot(self):
        # Remove the robot from the simulation
        p.removeBody(self.robot_id, physicsClientId=self._physics_client_id)

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses, joint_dumping = [], [], [], [], []
        for i in range(self._num_joints):
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)

            if jointInfo[3] > -1:
                ll, ul = jointInfo[8:10]
                jr = ul - ll
                # For simplicity, assume resting state == initial state
                rp = p.getJointState(self.robot_id, i, physicsClientId=self._physics_client_id)[0]
                lower_limits.append(ll)
                upper_limits.append(ul)
                joint_ranges.append(jr)
                rest_poses.append(rp)
                joint_dumping.append(0.1 if i in self._motor_idxs else 100.)

        return lower_limits, upper_limits, joint_ranges, rest_poses, joint_dumping

    def get_ws_lim(self):
        return self._workspace_lim

    def get_action_dim(self):
        if not self._use_IK:
            return len(self._motor_idxs)
        if self._control_orientation and self._control_eu_or_quat is 0:
            return 6  # position x,y,z + roll/pitch/yaw of hand frame
        elif self._control_orientation and self._control_eu_or_quat is 1:
            return 7  # position x,y,z + quat of hand frame
        return 3  # position x,y,z

    def get_observation_dim(self):
        return len(self.get_observation())

    def get_observation(self):
        # Cartesian world pos/orn of left hand center of mass
        observation = []
        observation_lim = []
        state = p.getLinkState(self.robot_id, self._end_eff_idx, computeLinkVelocity=1,
                                computeForwardKinematics=1, physicsClientId=self._physics_client_id)
        pos = state[0]
        orn = state[1]

        vel_l = state[6]
        vel_a = state[7]

        # cartesian position
        observation.extend(list(pos))
        observation_lim.extend(list(self._workspace_lim))

        # cartesian orientation
        if self._control_eu_or_quat is 0:
            euler = p.getEulerFromQuaternion(orn)
            observation.extend(list(euler))  # roll, pitch, yaw
            observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])
        else:
            observation.extend(list(orn))  # roll, pitch, yaw
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        # cartesian velocities (linear and angular)
        observation.extend(list(vel_l))
        observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])
        observation.extend(list(vel_a))
        observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])

        # finger tips
        # tips_idxs = [3,7,11,15,19]
        # if self._control_arm is 'l':
        #     finger_idxs = self._indices_left_hand
        # else:
        #     finger_idxs = self._indices_right_hand

        # tip_1 = p.getLinkState(self.robot_id, finger_idxs[tips_idxs[0]], physicsClientId=self._physics_client_id)
        # tip_2 = p.getLinkState(self.robot_id, finger_idxs[tips_idxs[1]], physicsClientId=self._physics_client_id)
        # tip_3 = p.getLinkState(self.robot_id, finger_idxs[tips_idxs[2]], physicsClientId=self._physics_client_id)
        # tip_4 = p.getLinkState(self.robot_id, finger_idxs[tips_idxs[3]], physicsClientId=self._physics_client_id)
        # tip_5 = p.getLinkState(self.robot_id, finger_idxs[tips_idxs[4]], physicsClientId=self._physics_client_id)

        # # finger tips positions
        # observation.extend(list(tip_1[0]))
        # observation_lim.extend(list(self._workspace_lim))
        # observation.extend(list(tip_2[0]))
        # observation_lim.extend(list(self._workspace_lim))
        # observation.extend(list(tip_3[0]))
        # observation_lim.extend(list(self._workspace_lim))
        # observation.extend(list(tip_4[0]))
        # observation_lim.extend(list(self._workspace_lim))
        # observation.extend(list(tip_5[0]))
        # observation_lim.extend(list(self._workspace_lim))
        #
        # # finger tips orientations
        # if self._control_eu_or_quat is 0:
        #     observation.extend(list(p.getEulerFromQuaternion(tip_1[1])))
        #     observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])
        #     observation.extend(list(p.getEulerFromQuaternion(tip_2[1])))
        #     observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])
        #     observation.extend(list(p.getEulerFromQuaternion(tip_3[1])))
        #     observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])
        #     observation.extend(list(p.getEulerFromQuaternion(tip_4[1])))
        #     observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])
        #     observation.extend(list(p.getEulerFromQuaternion(tip_5[1])))
        #     observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])
        # else:
        #     observation.extend(list(tip_1[1]))
        #     observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
        #     observation.extend(list(tip_2[1]))
        #     observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
        #     observation.extend(list(tip_3[1]))
        #     observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
        #     observation.extend(list(tip_4[1]))
        #     observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
        #     observation.extend(list(tip_5[1]))
        #     observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        # joints poses
        joint_states = p.getJointStates(self.robot_id, self._motor_idxs, physicsClientId=self._physics_client_id)
        joint_poses = [x[0] for x in joint_states]
        observation.extend(list(joint_poses))
        observation_lim.extend([[self.ll[i], self.ul[i]] for i in self._motor_idxs])

        return observation, observation_lim

    def _com_to_link_hand_frame(self):
        if self._control_arm is 'r':
            com_T_link_hand = (0.064668, -0.0056, -0.022681)
        else:
            com_T_link_hand = (-0.064768, -0.00563, -0.02266)

        return com_T_link_hand

    def apply_action(self, action):
        if self._use_IK:

            if not (len(action) == 3 or len(action) == 6 or len(action) == 7):
                raise AssertionError('number of action commands must be \n- 3: (dx,dy,dz)'
                                     '\n- 6: (dx,dy,dz,droll,dpitch,dyaw)'
                                     '\n- 7: (dx,dy,dz,qx,qy,qz,w)'
                                     '\ninstead it is: ', len(action))

            dx, dy, dz = action[:3]

            new_pos = [min(self._workspace_lim[0][1], max(self._workspace_lim[0][0], dx)),
                       min(self._workspace_lim[1][1], max(self._workspace_lim[1][0], dy)),
                       min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

            if not self._control_orientation:
                new_quat_orn = p.getQuaternionFromEuler(self._home_hand_pose[3:6])

            elif len(action) == 6:
                droll, dpitch, dyaw = action[3:6]

                new_eu_orn = [min(self._eu_lim[0][1], max(self._eu_lim[0][0], droll)),
                              min(self._eu_lim[1][1], max(self._eu_lim[1][0], dpitch)),
                              min(self._eu_lim[2][1], max(self._eu_lim[2][0], dyaw))]

                new_quat_orn = p.getQuaternionFromEuler(new_eu_orn)

            elif len(action) == 7:
                new_quat_orn = action[3:7]
            else:
                new_quat_orn = p.getLinkState(self.robot_id, self._end_eff_idx, physicsClientId=self._physics_client_id)[5]

            # transform the new pose from COM coordinate to link coordinate
            com_T_link_hand = self._com_to_link_hand_frame()

            link_hand_pose = p.multiplyTransforms(new_pos, new_quat_orn,
                                                  com_T_link_hand, (0., 0., 0., 1.))

            # compute joint positions with IK
            jointPoses = p.calculateInverseKinematics(self.robot_id, self._end_eff_idx,
                                                      link_hand_pose[0], link_hand_pose[1],
                                                      jointDamping=self.jd,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)

            # workaround to block joints of not-controlled arm
            if self._use_simulation:
                for i in range(self._num_joints):
                    jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
                    if i in self._joints_to_block:
                        continue
                    if jointInfo[3] > -1:
                        # minimize error is:
                        # error = position_gain * (desired_position - actual_position) +
                        #         velocity_gain * (desired_velocity - actual_velocity)

                        p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=jointPoses[i],
                                                targetVelocity=0,
                                                force=500,
                                                physicsClientId=self._physics_client_id)

            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self._num_joints):
                    p.resetJointState(self.robot_id, i, jointPoses[i], physicsClientId=self._physics_client_id)

        else:
            if not len(action) == len(self._motor_idxs):
                raise AssertionError('number of motor commands differs from number of motor to control',
                                     len(action), len(self._motor_idxs))

            for idx, val in enumerate(action):
                motor = self._motor_idxs[idx]

                # curr_motor_pos = p.getJointState(self.robot_id, motor, physicsClientId=self._physics_client_id)[0]
                new_motor_pos = min(self.ul[motor], max(self.ll[motor], val))

                p.setJointMotorControl2(self.robot_id,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        force=50,
                                        physicsClientId=self._physics_client_id)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def debug_gui(self):

        ws = self._workspace_lim
        p1 = [ws[0][0], ws[1][0], ws[2][0]]  # xmin,ymin
        p2 = [ws[0][1], ws[1][0], ws[2][0]]  # xmax,ymin
        p3 = [ws[0][1], ws[1][1], ws[2][0]]  # xmax,ymax
        p4 = [ws[0][0], ws[1][1], ws[2][0]]  # xmin,ymax

        p.addUserDebugLine(p1, p2, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p2, p3, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0,physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1,
                           physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1,
                           physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], parentObjectUniqueId=self.robot_id, parentLinkIndex=-1,
                           physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_right_arm[-1], physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_right_arm[-1], physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_right_arm[-1], physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_left_arm[-1], physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_left_arm[-1], physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self._indices_left_arm[-1], physicsClientId=self._physics_client_id)
