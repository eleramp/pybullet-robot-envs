import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import numpy as np
import math as m
import pybullet as p
import pybullet_data


def get_objects_list():
    obj_list = [
        'duck_vhacd',
        'cube_small',
        'teddy_vhacd',
        'domino/domino',
    ]

    return obj_list

# # #
class WorldFetchEnv:

    def __init__(self,
                 obj_name='duck_vhacd',
                 obj_pose_rnd_std=0.05,
                 workspace_lim=None,
                 control_eu_or_quat=0):

        if workspace_lim is None:
            workspace_lim = [[0.25, 0.52], [-0.3, 0.3], [0.5, 1.0]]

        self._ws_lim = tuple(workspace_lim)
        self._h_table = []
        self._obj_name = obj_name
        self._obj_pose_rnd_std = obj_pose_rnd_std
        self._obj_init_pose = []

        self.obj_id = []

        self._control_eu_or_quat = control_eu_or_quat

        # initialize
        self.reset()

    def reset(self):
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

        # Load table and object
        table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                              basePosition=[0.85, 0.0, 0.0], useFixedBase=True)

        table_info = p.getCollisionShapeData(table_id, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2]/2

        # Load object. Randomize its start position if requested
        self._obj_init_pose = self._sample_pose()
        self.obj_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), self._obj_name + ".urdf"),
                                basePosition=self._obj_init_pose[:3], baseOrientation=self._obj_init_pose[3:7],
                                 flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

    def get_object_init_pose(self):
        pos = self._obj_init_pose[:3]
        quat = self._obj_init_pose[3:7]
        return pos, quat

    def get_table_height(self):
        return self._h_table

    def get_object_shape_info(self):
        info = list(p.getCollisionShapeData(self.obj_id, -1)[0])
        info[4] = p.getVisualShapeData(self.obj_id, -1)[0][4]
        return info

    def get_observation_dimension(self):
        obs, _ = self.get_observation()
        return len(obs)

    def get_observation(self):
        observation = []
        observation_lim = []

        # get object position
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id)
        observation.extend(list(obj_pos))
        observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

        if self._control_eu_or_quat is 0:
            obj_euler = p.getEulerFromQuaternion(obj_orn)  # roll, pitch, yaw
            observation.extend(list(obj_euler))
            observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])
        else:
            observation.extend(list(obj_orn))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        return observation, observation_lim

    def set_obj_pose(self, new_pos, new_quat):
        p.resetBasePositionAndOrientation(self.obj_id, new_pos, new_quat)

    def check_contact(self, link_id):
        pts = p.getContactPoints(self.obj_id, link_id)
        if len(pts) > 0:
            print("<<----------->> contact with object!!!!! <<----------------->>")
            return True
        return False

    def debug_gui(self):
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.obj_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.obj_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.obj_id)

    def _sample_pose(self):
        # ws_lim = self._ws_lim
        x_min = self._ws_lim[0][0] + 0.064668
        x_max = self._ws_lim[0][1] - 0.05

        px = x_min + 0.4 * (x_max - x_min)
        py = self._ws_lim[1][0] + 0.5 * (self._ws_lim[1][1] - self._ws_lim[1][0])
        pz = self._h_table+0.06
        quat = p.getQuaternionFromEuler([0.0, 0.0, 2/3*m.pi])

        if self._obj_pose_rnd_std > 0:
            # Add a Gaussian noise to position
            mu, sigma = 0, self._obj_pose_rnd_std
            noise = np.random.normal(mu, sigma, 2)

            px = px + noise[0]
            px = np.clip(px, x_min, x_max)

            py = py + noise[1]
            py = np.clip(py, self._ws_lim[1][0], self._ws_lim[1][1])

            # Add uniform noise to yaw orientation
            quat = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=1/4*m.pi, high=3/4*m.pi)])

        obj_pose = (px, py, pz) + quat

        return obj_pose
