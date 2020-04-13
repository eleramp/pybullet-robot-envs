import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import pybullet as p
import pybullet_data


from pybullet_robot_envs.envs.world_envs.fetch_env import WorldFetchEnv
import ycb_objects_models_sim
from ycb_objects_models_sim import objects


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


class YcbWorldFetchEnv(WorldFetchEnv):

    def __init__(self,
                 obj_name='YcbMustardBottle',
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
        self.seed()
        self.reset()

    def reset(self):
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

        # Load table and object
        table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.85, 0.0, 0.0])

        table_info = p.getCollisionShapeData(table_id, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2]/2

        # Load object. Randomize its start position if requested
        self._obj_init_pose = self._sample_pose()
        self.obj_id = p.loadURDF(os.path.join(ycb_objects_models_sim.objects.getDataPath(), self._obj_name,  "model.urdf"),
                                basePosition=self._obj_init_pose[:3], baseOrientation=self._obj_init_pose[3:7])
