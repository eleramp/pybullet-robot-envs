# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import pybullet as p
import pybullet_data
from gym.utils import seeding

import numpy as np
import math as m

from pybullet_robot_envs.envs.utils import goal_distance

class pandaEnv:

    def __init__(self, physicsClientId, use_IK=0, base_position=(0.0, 0, 0.625), joint_action_space=7, includeVelObs = True,
                 control_eu_or_quat=0):

        self._physics_client_id = physicsClientId
        self._use_IK = use_IK
        self._control_orientation = 1
        self._use_simulation = 1
        self._base_position = base_position

        self.joint_action_space = joint_action_space
        self._include_vel_obs = includeVelObs
        self._control_eu_or_quat = control_eu_or_quat

        self._workspace_lim = [[0.3, 0.7], [-0.3, 0.3], [0.65, 1.5]]
        self._eu_lim = [[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]]

        self.endEffLink = 11  # 8
        self._home_pos_joints = [0, -0.54, 0, -2.6, -0.30, 2, 1, 0.02, 0.02]

        self._home_hand_pose = []

        self._num_dof = 7
        self._motor_idxs = []
        self.robot_id = None

        self.seed()
        self.reset()

    def reset(self):
        # load robot
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
        self.robot_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
                                   basePosition=self._base_position, useFixedBase=True, flags=flags,
                                   physicsClientId=self._physics_client_id)

        # reset joints to home position
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        idx = 0
        self._motor_idxs = []
        for i in range(num_joints):
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            jointType = jointInfo[2]
            if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
                p.resetJointState(self.robot_id, i, self._home_pos_joints[idx], physicsClientId=self._physics_client_id)
                self._motor_idxs.append(i)
                idx += 1

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        if self._use_IK:
            self._home_hand_pose = [self._base_position[0] + 0.2,
                                    self._base_position[1] - 0.2,
                                    self._base_position[2] + 0.4,
                                    -2/3*m.pi, -m.pi/4, 0]  # x,y,z,roll,pitch,yaw

            self._home_hand_pose = [min(self._workspace_lim[0][1], max(self._workspace_lim[0][0], self._base_position[0] + 0.2)),
                                    min(self._workspace_lim[1][1], max(self._workspace_lim[1][0], self._base_position[1])),
                                    min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], self._base_position[2] + 0.4)),
                                    min(m.pi, max(-m.pi, m.pi)),
                                    min(m.pi, max(-m.pi, 0)),
                                    min(m.pi, max(-m.pi, 0))]

            self.apply_action(self._home_hand_pose)
            p.stepSimulation(physicsClientId=self._physics_client_id)

        self.debug_gui()

    def delete_simulated_robot(self):
        # Remove the robot from the simulation
        p.removeBody(self.robot_id, physicsClientId=self._physics_client_id)

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)):
            jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            jointType = jointInfo[2]

            if jointType is p.JOINT_REVOLUTE or jointType is p.JOINT_PRISMATIC:
                ll, ul = jointInfo[8:10]
                jr = ul - ll
                # For simplicity, assume resting state == initial state
                rp = p.getJointState(self.robot_id, i, physicsClientId=self._physics_client_id)[0]
                lower_limits.append(ll)
                upper_limits.append(ul)
                joint_ranges.append(jr)
                rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses

    def get_action_dimension(self):
        if not self._use_IK:
            return self.joint_action_space
        if self._control_orientation and self._control_eu_or_quat is 0:
            return 6  # position x,y,z + roll/pitch/yaw of hand frame
        elif self._control_orientation and self._control_eu_or_quat is 1:
            return 7  # position x,y,z + quat of hand frame
        return 3  # position x,y,z

    def get_observation_dimension(self):
        return len(self.get_observation())

    def get_observation(self):
        observation = []
        observation_lim = []
        state = p.getLinkState(self.robot_id, self.endEffLink, computeLinkVelocity=1,
                               computeForwardKinematics=1, physicsClientId=self._physics_client_id)
        pos = state[0]
        orn = state[1]

        euler = p.getEulerFromQuaternion(orn)

        # cartesian position
        observation.extend(list(pos))
        observation_lim.extend(list(self._workspace_lim))

        # cartesian orientation
        if self._control_eu_or_quat is 0:
            euler = p.getEulerFromQuaternion(orn)
            observation.extend(list(euler))  # roll, pitch, yaw
            observation_lim.extend([[-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi]])
        else:
            observation.extend(list(orn))  # roll, pitch, yaw
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        if self._include_vel_obs:
            vel_l = state[6]
            vel_a = state[7]
            observation.extend(list(vel_l))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])
            observation.extend(list(vel_a))
            observation_lim.extend([[-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi]])

        jointStates = p.getJointStates(self.robot_id, self._motor_idxs, physicsClientId=self._physics_client_id)
        jointPoses = [x[0] for x in jointStates]

        observation.extend(list(jointPoses))
        observation_lim.extend([[self.ll[i], self.ul[i]] for i in range(0, len(self._motor_idxs))])

        return observation, observation_lim

    def pre_grasp(self):
        self.apply_action_fingers((0.04, 0.04))
        for _ in range(0,10):
            p.stepSimulation(physicsClientId=self._physics_client_id)
        f1 = p.getLinkState(self.robot_id, self._motor_idxs[-2], physicsClientId=self._physics_client_id)[0]
        f2 = p.getLinkState(self.robot_id, self._motor_idxs[-1], physicsClientId=self._physics_client_id)[0]

    def apply_action_fingers(self, action):
        assert len(action) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))

        for i in self._motor_idxs[-2:]:
            p.setJointMotorControl2(self.robot_id,
                                    i,
                                    p.POSITION_CONTROL,
                                    targetPosition=action[0],
                                    force=10,
                                    physicsClientId=self._physics_client_id)

    def apply_action(self, action, max_vel=1):

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
                droll, dpitch, dyaw = action[3:]

                eu_orn = [min(m.pi, max(-m.pi, droll)),
                          min(m.pi, max(-m.pi, dpitch)),
                          min(m.pi, max(-m.pi, dyaw))]

                new_quat_orn = p.getQuaternionFromEuler(eu_orn)

            elif len(action) == 7:
                new_quat_orn = action[3:7]

            else:
                new_quat_orn = p.getLinkState(self.robot_id, self.endEffLink, physicsClientId=self._physics_client_id)[5]

            jointPoses = p.calculateInverseKinematics(self.robot_id, self.endEffLink, new_pos, new_quat_orn,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001,
                                                      physicsClientId=self._physics_client_id)
            jointPoses = np.multiply(1, jointPoses)
            if self._use_simulation:
                    for i in range(self._num_dof):
                        jointInfo = p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
                        if jointInfo[3] > -1:
                            p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                    jointIndex=i,
                                                    controlMode=p.POSITION_CONTROL,
                                                    targetPosition=jointPoses[i],
                                                    targetVelocity=0,
                                                    maxVelocity=max_vel,
                                                    force=500,
                                                    physicsClientId=self._physics_client_id)
            else:
                for i in range(self._num_dof):
                    p.resetJointState(self.robot_id, i, jointPoses[i], physicsClientId=self._physics_client_id)

        else:
            assert len(action) == self.joint_action_space, ('number of motor commands differs from number of motor to control', len(action))

            for a in range(len(action)):
                curr_motor_pos = p.getJointState(self.robot_id, a, physicsClientId=self._physics_client_id)[0]
                new_motor_pos = curr_motor_pos + action[a]  # supposed to be a delta

                p.setJointMotorControl2(self.robot_id,
                                        a,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        targetVelocity=0,
                                        force=500,
                                        physicsClientId=self._physics_client_id)

    def check_contact_fingertips(self, obj_id):
        p0 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=self._motor_idxs[-2], physicsClientId=self._physics_client_id)
        p1 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=self._motor_idxs[-1], physicsClientId=self._physics_client_id)

        fingers_in_contact = 0

        p0_f = 0
        if len(p0) > 0:
            fingers_in_contact += 1
            print("p0! {}".format(len(p0)))
            for pp in p0:
                p0_f += pp[9]
            p0_f /= len(p0)
            #print("\t\t p0 normal force! {}".format(p0_f))

        p1_f = 0
        if len(p1) > 0:
            fingers_in_contact += 1
            print("p1! {}".format(len(p1)))
            for pp in p1:
                p1_f += pp[9]
            p1_f /= len(p1)
            #print("\t\t p1 normal force! {}".format(p1_f))

        return fingers_in_contact, [p0_f, p1_f]

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
        p.addUserDebugLine(p3, p4, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)
        p.addUserDebugLine(p4, p1, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0, physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=-1, physicsClientId=self._physics_client_id)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.endEffLink, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.endEffLink, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.robot_id,
                           parentLinkIndex=self.endEffLink, physicsClientId=self._physics_client_id)
