# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import numpy as np
import math as m
import pybullet as p
from gym.utils import seeding

import pybullet_data
from pybullet_object_models import ycb_objects, superquadric_objects


def get_objects_list():
    obj_list = [dir for dir in os.listdir(pybullet_data.getDataPath()) if not dir.startswith('__')]
    return obj_list


def get_ycb_objects_list():
    obj_list = [
        'YcbMustardBottle',
        'YcbTomatoSoupCan',
        'YcbCrackerBox',
        'YcbChipsCan',
        'YcbBanana',
        'YcbFoamBrick',
        'YcbGelatinBox',
        'YcbHammer',
        'YcbMasterChefCan',
        'YcbMediumClamp',
        'YcbPear',
        'YcbPottedMeatCan',
        'YcbPowerDrill',
        'YcbScissors',
        'YcbStrawberry',
        'YcbTennisBall',
    ]
    return obj_list


def get_sq_objects_list():
    obj_list = [dir for dir in os.listdir(superquadric_objects.getDataPath()) if not dir.startswith('__')]
    return obj_list


class WorldEnv:

    def __init__(self,
                 physicsClientId,
                 obj_name='duck_vhacd',
                 obj_pose_rnd_std=0.0,
                 obj_orientation_rnd=0.0,
                 workspace_lim=None,
                 control_eu_or_quat=0):

        if workspace_lim is None:
            workspace_lim = [[0.25, 0.52], [-0.3, 0.3], [0.5, 1.0]]

        self._physics_client_id = physicsClientId
        self._ws_lim = tuple(workspace_lim)
        self._h_table = []
        self._obj_name = obj_name
        self._prev_obj_name = None
        self._obj_pose_rnd_std = obj_pose_rnd_std
        self._obj_orientation_rnd = obj_orientation_rnd
        self._obj_init_pose = []

        self.obj_id = None
        self.table_id = None
        self._is_initialized = False

        self._control_eu_or_quat = control_eu_or_quat

        # initialize
        self.seed()
        self.reset()

    def reset(self):
        if not self._is_initialized:
            self._is_initialized = True
            p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0], physicsClientId=self._physics_client_id)

            # Load table and object
            self.table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                                  basePosition=[0.85, 0.0, 0.0], useFixedBase=True, physicsClientId=self._physics_client_id)

            table_info = p.getCollisionShapeData(self.table_id, -1, physicsClientId=self._physics_client_id)[0]
            self._h_table = table_info[5][2] + table_info[3][2]/2

            # set ws limit on z according to table height
            self._ws_lim[2][:] = [self._h_table, self._h_table + 0.3]


        self.load_object(self._obj_name)

    def load_object(self, new_obj):

        # Load object. Randomize its start position if requested
        if new_obj == self._prev_obj_name and self.obj_id is not None:
            obj_pos = self._sample_pose()
            p.resetBasePositionAndOrientation(self.obj_id,
                                              obj_pos[:3], obj_pos[3:7],
                                              physicsClientId=self._physics_client_id)
        else:
            self._prev_obj_name = new_obj
            if self.obj_id is not None:
                p.removeBody(self.obj_id, physicsClientId=self._physics_client_id)
            self._obj_name = new_obj
            obj_pos = self._sample_pose()
            self.obj_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), new_obj + ".urdf"),
                                     basePosition=obj_pos[:3], baseOrientation=obj_pos[3:7],
                                     flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                                     physicsClientId=self._physics_client_id)

        # Run the simulation a bit to let the object stabilize its pose
        for _ in range(600):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self._physics_client_id)

        self._obj_init_pose = obj_pos + obj_orn

    def get_object_init_pose(self):
        pos = self._obj_init_pose[:3]
        quat = self._obj_init_pose[3:7]
        return pos, quat

    def set_obj_pose(self, new_pos, new_quat):
        p.resetBasePositionAndOrientation(self.obj_id, new_pos, new_quat, physicsClientId=self._physics_client_id)

    def get_table_height(self):
        return self._h_table

    def get_object_shape_info(self):
        info = list(p.getCollisionShapeData(self.obj_id, -1, physicsClientId=self._physics_client_id)[0])
        info[4] = p.getVisualShapeData(self.obj_id, -1, physicsClientId=self._physics_client_id)[0][4]
        return info

    def get_obj_bbox(self):
        bbox_min, bbox_max = p.getAABB(self.obj_id, physicsClientId=self._physics_client_id)
        return bbox_min, bbox_max

    def get_workspace(self):
        return [i[:] for i in self._ws_lim]

    def get_observation_dimension(self):
        obs, _ = self.get_observation()
        return len(obs)

    def get_observation(self):
        observation = []
        observation_lim = []

        # get object position
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self._physics_client_id)
        observation.extend(list(obj_pos))
        observation_lim.extend(self._ws_lim)

        if self._control_eu_or_quat is 0:
            obj_euler = p.getEulerFromQuaternion(obj_orn)  # roll, pitch, yaw
            observation.extend(list(obj_euler))
            observation_lim.extend([[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]])
        else:
            observation.extend(list(obj_orn))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        return observation, observation_lim

    def check_contact(self, body_id, obj_id=None):
        if obj_id is None:
            obj_id = self.obj_id

        pts = p.getContactPoints(obj_id, body_id, physicsClientId=self._physics_client_id)

        return len(pts) > 0

    def debug_gui(self):
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.obj_id, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.obj_id, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.obj_id, physicsClientId=self._physics_client_id)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sample_pose(self):
        # ws_lim = self._ws_lim
        x_min = self._ws_lim[0][0] + 0.05
        x_max = self._ws_lim[0][1] - 0.1
        y_min = self._ws_lim[1][0] + 0.05
        y_max = self._ws_lim[1][1] - 0.05

        px = x_min + 0.6 * (x_max - x_min)
        py = y_min + 0.5 * (y_max - y_min)
        pz = self._h_table + 0.07

        eu = [0.0, 0.0, 0.0]
        quat = p.getQuaternionFromEuler(eu)

        if self._obj_pose_rnd_std > 0:
            # Add a Gaussian noise to position
            mu, sigma = 0, self._obj_pose_rnd_std
            # noise = self.np_random.normal(mu, sigma, 2)
            noise = [self.np_random.uniform(low=-self._obj_pose_rnd_std, high=self._obj_pose_rnd_std),
                     self.np_random.uniform(low=-self._obj_pose_rnd_std, high=self._obj_pose_rnd_std)]

            px = px + noise[0]
            py = py + noise[1]

        if self._obj_orientation_rnd > 0:

            # Add uniform noise to yaw orientation
            idx = self.np_random.randint(0, 2)
            eu[idx] = self.np_random.uniform(low=-m.pi/2, high=m.pi/2)
            quat = p.getQuaternionFromEuler(eu)

            # quat = p.getQuaternionFromEuler([self.np_random.uniform(low=-m.pi/2, high=m.pi/2),
            #                                  self.np_random.uniform(low=-m.pi/2, high=m.pi/2),
            #                                  self.np_random.uniform(low=-m.pi/2, high=m.pi/2)])

        px = np.clip(px, x_min, x_max)
        py = np.clip(py, y_min, y_max)

        obj_pose = (px, py, pz) + quat

        return obj_pose


class YcbWorldEnv(WorldEnv):

    def __init__(self,
                 physicsClientId,
                 obj_name='YcbMustardBottle',
                 obj_pose_rnd_std=0.05,
                 obj_orientation_rnd=0.0,
                 workspace_lim=None,
                 control_eu_or_quat=0):
        super().__init__(physicsClientId, obj_name, obj_pose_rnd_std, obj_orientation_rnd, workspace_lim,
                                       control_eu_or_quat)


    def load_object(self, new_obj):

        # Load object. Randomize its start position if requested
        if new_obj == self._prev_obj_name and self.obj_id is not None:
            obj_pos = self._sample_pose()
            p.resetBasePositionAndOrientation(self.obj_id,
                                              obj_pos[:3], obj_pos[3:7],
                                              physicsClientId=self._physics_client_id)
        else:
            self._prev_obj_name = new_obj
            if self.obj_id is not None:
                p.removeBody(self.obj_id, physicsClientId=self._physics_client_id)
            self._obj_name = new_obj
            obj_pos = self._sample_pose()
            self.obj_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), new_obj, "model.urdf"),
                                     basePosition=obj_pos[:3], baseOrientation=obj_pos[3:7],
                                     flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                                     physicsClientId=self._physics_client_id)

        # Run the simulation a bit to let the object stabilize its pose
        for _ in range(600):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self._physics_client_id)

        self._obj_init_pose = obj_pos + obj_orn


class SqWorldEnv(WorldEnv):

    def __init__(self,
                 physicsClientId,
                 obj_name='sq_0.12_0.12_0.12_1_1',
                 obj_pose_rnd_std=0.05,
                 obj_orientation_rnd=0.0,
                 workspace_lim=None,
                 control_eu_or_quat=0):
        super().__init__(physicsClientId, obj_name, obj_pose_rnd_std, obj_orientation_rnd, workspace_lim,
                                       control_eu_or_quat)

    def load_object(self, new_obj):

        # Load object. Randomize its start position if requested
        if new_obj == self._prev_obj_name and self.obj_id is not None:
            obj_pos = self._sample_pose()
            p.resetBasePositionAndOrientation(self.obj_id,
                                              obj_pos[:3], obj_pos[3:7],
                                              physicsClientId=self._physics_client_id)
        else:
            self._prev_obj_name = new_obj
            if self.obj_id is not None:
                p.removeBody(self.obj_id, physicsClientId=self._physics_client_id)
            self._obj_name = new_obj
            obj_pos = self._sample_pose()
            self.obj_id = p.loadURDF(os.path.join(superquadric_objects.getDataPath(), new_obj, "model.urdf"),
                                     basePosition=obj_pos[:3], baseOrientation=obj_pos[3:7],
                                     physicsClientId=self._physics_client_id)

        # Run the simulation a bit to let the object stabilize its pose
        for _ in range(600):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self._physics_client_id)

        self._obj_init_pose = obj_pos + obj_orn
