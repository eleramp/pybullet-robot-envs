import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,currentdir)

import numpy as np
import pybullet as p
import pybullet_data
# import robot_data
import pymesh
import superquadric_bindings
from superquadric_bindings import PointCloud, SuperqEstimatorApp


class WorldFetchEnv:

    def __init__(self,
                 rnd_obj_pose=1,
                 workspace_lim=None):

        if workspace_lim is None:
            workspace_lim = [[0.25, 0.52], [-0.2, 0.2], [0.5, 1.0]]

        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._ws_lim = workspace_lim
        self._h_table = []
        self._obj_id = []
        self._rnd_obj_pose = rnd_obj_pose

        self._pointcloud = PointCloud()
        self._sqestimator = SuperqEstimatorApp()

        # initialize
        self.reset()

    def reset(self):
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

        # Load table and object for simulation
        table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.85, 0.0, 0.0])

        table_info = p.getCollisionShapeData(table_id, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2]/2

        # Load object. Randomize its start position if requested
        obj_pose = self._sample_pose()
        self._obj_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "duck_vhacd.urdf"), obj_pose)
        # self._obj_id = p.loadURDF(os.path.join(robot_data.getDataPath(), "objects/bottle/bottle.urdf"),obj_pose)

    def get_table_height(self):
        return self._h_table

    def get_object_shape_info(self):
        return p.getCollisionShapeData(self._obj_id, -1)[0]

    def get_observation_dimension(self):
        return len(self.getObservation())

    def get_observation(self):
        observation = []

        # get object position
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self._obj_id)
        obj_euler = p.getEulerFromQuaternion(obj_orn)  # roll, pitch, yaw

        observation.extend(list(obj_pos))
        observation.extend(list(obj_euler))
        return observation

    def _sample_pose(self):
        if self._rnd_obj_pose:
            px = self.np_random.uniform(low=self._ws_lim[0][0], high=self._ws_lim[0][1], size=(1))
            py = self.np_random.uniform(low=self._ws_lim[1][0]+0.005*self.np_random.rand(), high=self._ws_lim[1][1]-0.005*self.np_random.rand(), size=(1))
        else:
            px = self._ws_lim[0][0] + 0.5*(self._ws_lim[0][1]-self._ws_lim[0][0])
            py = self._ws_lim[1][0] + 0.5*(self._ws_lim[1][1]-self._ws_lim[1][0])
        pz = self._h_table
        obj_pose = [px, py, pz]

        return obj_pose

    def debug_gui(self):
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self._obj_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self._obj_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self._obj_id)