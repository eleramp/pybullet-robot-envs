import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import numpy as np
import math as m
import pybullet as p
import pybullet_data
import robot_data


class WorldFetchEnv:

    def __init__(self,
                 rnd_obj_pose=0.05,
                 workspace_lim=None):

        if workspace_lim is None:
            workspace_lim = [[0.25, 0.52], [-0.2, 0.2], [0.5, 1.0]]

        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._ws_lim = workspace_lim
        self._h_table = []
        self.obj_id = []
        self._rnd_obj_pose = rnd_obj_pose

        # initialize
        self.reset()

    def reset(self):
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

        # Load table and object for simulation
        table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.85, 0.0, 0.0])

        table_info = p.getCollisionShapeData(table_id, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2]/2

        # Load object. Randomize its start position if requested
        #obj_pose = self._sample_pose()
        #self.obj_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "duck_vhacd.urdf"), obj_pose)

        obj_pose = self._sample_pose()
        self.obj_id = p.loadURDF(os.path.join(robot_data.getDataPath(), "objects/006_mustard_bottle/mustard_bottle.urdf"),
                                basePosition=obj_pose[:3], baseOrientation=obj_pose[3:7])

    def get_table_height(self):
        return self._h_table

    def get_object_shape_info(self):
        return p.getCollisionShapeData(self.obj_id, -1)[0]

    def get_observation_dimension(self):
        return len(self.getObservation())

    def get_observation(self):
        observation = []

        # get object position
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id)
        obj_euler = p.getEulerFromQuaternion(obj_orn)  # roll, pitch, yaw

        observation.extend(list(obj_pos))
        observation.extend(list(obj_euler))
        return observation

    def _sample_pose(self):
        px = self._ws_lim[0][0] + 0.5 * (self._ws_lim[0][1] - self._ws_lim[0][0])
        py = self._ws_lim[1][0] + 0.5 * (self._ws_lim[1][1] - self._ws_lim[1][0])
        pz = self._h_table
        quat = p.getQuaternionFromEuler([0.0, 0.0, 0])

        if self._rnd_obj_pose >= 0:
            # Add a Gaussian noise to position
            mu, sigma = 0, self._rnd_obj_pose
            noise = np.random.normal(mu, sigma, 2)
            px = px + noise[0]
            px = np.clip(px, self._ws_lim[0][0], self._ws_lim[0][1])
            py = py + noise[1]
            py = np.clip(py, self._ws_lim[1][0], self._ws_lim[1][1])
            # Add uniofrm noise to yaw orientation
            # quat = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=0, high=2.0 * m.pi)])

        obj_pose = (px, py, pz) + quat

        return obj_pose

    def check_object_moved(self, obj_pose):
        if goal_distance(np.array(obj_pose[:3]), np.array(self._obj_pose[:3])) > 0.02 or \
                goal_distance(np.array(obj_pose[3:6]), np.array(self._obj_pose[3:6])) > 0.5:
            # let the simulation run a bit to stabilize the object motion
            for _ in range(10):
                p.stepSimulation()
            return True

        return False

    def seed(self, seed=None):
        np.random.seed(seed)

    def debug_gui(self):
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.obj_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.obj_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.obj_id)