# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import pybullet as p
import pybullet_data
from gym.utils import seeding
from icub_model_pybullet import franka_panda

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
        self.robot_id = p.loadURDF(os.path.join(franka_panda.get_data_path(), "panda_model.urdf"),
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

            self._home_hand_pose = [min(self._workspace_lim[0][1], max(self._workspace_lim[0][0], self._base_position[0])),
                                    min(self._workspace_lim[1][1], max(self._workspace_lim[1][0], self._base_position[1])),
                                    min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], self._base_position[2] + 0.5)),
                                    min(m.pi, max(-m.pi, m.pi)),
                                    min(m.pi, max(-m.pi, -m.pi/4)),
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

    def get_workspace(self):
        return [i[:] for i in self._workspace_lim]

    def set_workspace(self, ws):
        self._workspace_lim = [i[:] for i in ws]

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
            observation_lim.extend(self._eu_lim)
        else:
            observation.extend(list(orn))  # roll, pitch, yaw
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        if self._include_vel_obs:
            # standardize by subtracting the mean and dividing by the std
            vel_std = [0.04, 0.07, 0.03]
            vel_mean = [0.0, 0.01, 0.0]
            vel_l = np.subtract(state[6], vel_mean)
            vel_l = np.divide(vel_l, vel_std)
            observation.extend(list(vel_l))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

        jointStates = p.getJointStates(self.robot_id, self._motor_idxs, physicsClientId=self._physics_client_id)
        jointPoses = [x[0] for x in jointStates]

        observation.extend(list(jointPoses))
        observation_lim.extend([[self.ll[i], self.ul[i]] for i in range(0, len(self._motor_idxs))])

        return observation, observation_lim

    def pre_grasp(self):
        self.apply_action_fingers((0.04, 0.04))
        for _ in range(0,10):
            p.stepSimulation(physicsClientId=self._physics_client_id)

    def get_fingertips_pose(self):
        state = p.getLinkState(self.robot_id, self._motor_idxs[-2], physicsClientId=self._physics_client_id)
        f1_pos = list(state[4])
        f1_orn = list(state[5])

        matrix = p.getMatrixFromQuaternion(f1_orn)
        dcm = np.array([matrix[0:3], matrix[3:6], matrix[6:9]])
        delta_pos = np.array(list(dcm.dot([0, 0, 0.045])))
        f1_pos[0] += delta_pos[0]
        f1_pos[1] += delta_pos[1]
        f1_pos[2] += delta_pos[2]

        state = p.getLinkState(self.robot_id, self._motor_idxs[-1], physicsClientId=self._physics_client_id)
        f2_pos = list(state[4])
        f2_orn = list(state[5])

        matrix = p.getMatrixFromQuaternion(f2_orn)
        dcm = np.array([matrix[0:3], matrix[3:6], matrix[6:9]])
        delta_pos = np.array(list(dcm.dot([0, 0, 0.045])))
        f2_pos[0] += delta_pos[0]
        f2_pos[1] += delta_pos[1]
        f2_pos[2] += delta_pos[2]

        return tuple(((f1_pos + f1_orn), (f2_pos + f2_orn)))

    def apply_action_fingers(self, action):
        assert len(action) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))

        for i in self._motor_idxs[-2:]:
            p.setJointMotorControl2(self.robot_id,
                                    i,
                                    p.POSITION_CONTROL,
                                    targetPosition=action[0],
                                    force=10,
                                    physicsClientId=self._physics_client_id)

    def apply_action(self, action, max_vel=None):

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
                            if max_vel:
                                p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                        jointIndex=i,
                                                        controlMode=p.POSITION_CONTROL,
                                                        targetPosition=jointPoses[i],
                                                        targetVelocity=0,
                                                        maxVelocity=max_vel,
                                                        force=500,
                                                        physicsClientId=self._physics_client_id)
                            else:
                                p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                                        jointIndex=i,
                                                        controlMode=p.POSITION_CONTROL,
                                                        targetPosition=jointPoses[i],
                                                        targetVelocity=0,
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

    def check_collision(self, obj_id):
        # check if there is any collision with an object

        contact_pts = p.getContactPoints(obj_id, self.robot_id, physicsClientId=self._physics_client_id)

        # check if the contact is on the fingertip(s)
        n_fingertips_contact = self.check_contact_fingertips(obj_id)

        return (len(contact_pts) - n_fingertips_contact) > 0

    def check_contact_fingertips(self, obj_id):
        # check if there is any contact on the internal part of the fingers, to check if they are correctly touching an object

        p0 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=self._motor_idxs[-2], physicsClientId=self._physics_client_id)
        p1 = p.getContactPoints(obj_id, self.robot_id, linkIndexB=self._motor_idxs[-1], physicsClientId=self._physics_client_id)

        p0_contact = 0
        if len(p0) > 0:
            # get cartesian position of the finger link frame in world coordinates
            w_pos_f0 = p.getLinkState(self.robot_id, self._motor_idxs[-2], physicsClientId=self._physics_client_id)[4:6]
            f0_pos_w = p.invertTransform(w_pos_f0[0], w_pos_f0[1])

            for pp in p0:
                # compute relative position of the contact point wrt the finger link frame
                f0_pos_pp = p.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])
                p0_contact += 1 if (f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005) else 0

        p1_contact = 0
        if len(p1) > 0:
            w_pos_f1 = p.getLinkState(self.robot_id, self._motor_idxs[-1], physicsClientId=self._physics_client_id)[4:6]
            f1_pos_w = p.invertTransform(w_pos_f1[0], w_pos_f1[1])

            for pp in p1:
                # compute relative position of the contact point wrt the finger link frame
                f1_pos_pp = p.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])
                p1_contact += 1 if (f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005) else 0

        return (p0_contact > 0) + (p1_contact > 0)

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
