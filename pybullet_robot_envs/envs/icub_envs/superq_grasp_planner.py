import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import numpy as np
import pybullet as p
import math as m
from __init__ import *

import pymesh
import superquadric_bindings
from superquadric_bindings import PointCloud, SuperqEstimatorApp, GraspEstimatorApp, Visualizer
import config_superq_grasp_planner as cfg

import threading


# class th_visualizer(threading.Thread):

#    def __init__(self):
#        threading.Thread.__init__(self, daemon=True)
#        self.visualizer = Visualizer()
#        self.visualizer.visualize()

#    def run(self):
#       pass


class SuperqGraspPlanner:

    def __init__(self, robot_base_pose=((0.0,) * 3, (0.0,)*4)):

        self._grasping_hand = "right"
        self._superqs = superquadric_bindings.vector_superquadric()
        self._robot_base_pose = robot_base_pose
        self._icub_hand_right_orn = [-m.pi/2, 0.0, m.pi]
        self._icub_hand_left_orn = [m.pi/2, 0.0, 0.0]
        self._starting_pose = []
        self._best_grasp_pose = []
        self._i_path = 0
        self._obj_pose = []
        self._obj_info = []
        self._visualizer = Visualizer()
        self._pointcloud = PointCloud()
        self._sq_estimator = SuperqEstimatorApp()

        self._grasp_estimator = GraspEstimatorApp()
        # initialize
        self.reset()

    def reset(self, starting_pose=[0.0]*6):

        self._starting_pose = starting_pose

        self._grasping_hand = cfg.mode['control_arms']

        # self._visualizer.setPosition(cfg.visualizer['x'], cfg.visualizer['y'])
        # self._visualizer.setSize(cfg.visualizer['width'], cfg.visualizer['height'])

        # ------ Set Superquadric Model parameters ------ #
        self._sq_estimator.SetNumericValue("tol", cfg.sq_model['tol'])
        self._sq_estimator.SetIntegerValue("print_level", 0)
        self._sq_estimator.SetStringValue("object_class", cfg.sq_model['object_class'])
        self._sq_estimator.SetIntegerValue("optimizer_points", cfg.sq_model['optimizer_points'])
        self._sq_estimator.SetBoolValue("random_sampling", cfg.sq_model['random_sampling'])
        self._sq_estimator.SetBoolValue("merge_model", cfg.sq_model['merge_model'])
        self._sq_estimator.SetIntegerValue("minimum_points", cfg.sq_model['minimum_points'])
        self._sq_estimator.SetIntegerValue("fraction_pc", cfg.sq_model['fraction_pc'])
        self._sq_estimator.SetNumericValue("threshold_axis", cfg.sq_model['threshold_axis'])
        self._sq_estimator.SetNumericValue("threshold_section1", cfg.sq_model['threshold_section1'])
        self._sq_estimator.SetNumericValue("threshold_section2", cfg.sq_model['threshold_section2'])

        # ------ Set Superquadric Grasp parameters ------ #
        self._grasp_estimator.SetIntegerValue("print_level", 0)
        self._grasp_estimator.SetNumericValue("tol", cfg.sq_grasp['tol'])
        self._grasp_estimator.SetIntegerValue("max_superq", cfg.sq_grasp['max_superq'])
        self._grasp_estimator.SetNumericValue("constr_tol", cfg.sq_grasp['constr_tol'])
        self._grasp_estimator.SetStringValue("left_or_right", self._grasping_hand)
        self._grasp_estimator.setVector("plane", np.array(cfg.sq_grasp['plane_table']))
        self._grasp_estimator.setVector("displacement", np.array(cfg.sq_grasp['displacement']))
        self._grasp_estimator.setVector("hand", np.array(cfg.sq_grasp['hand_sq']))

        # Start visualization
        # self._visualizer.visualize()
        return

    def set_object_info(self, obj_info):
        self._obj_info = obj_info

    def compute_object_pointcloud(self, obj_pose):
        # Create object's point cloud from its mesh

        if not self._obj_info:
            print("compute_object_pointcloud(): no object information provided. Please call set_object_info() first.")
            return False

        self._pointcloud.deletePoints()
        self._visualizer.resetPoints()

        # Current object link pose wrt pybullet world frame
        self._obj_pose = obj_pose
        obj_pos = obj_pose[:3]
        obj_orn = obj_pose[3:6]

        # Object collision shape info
        obj_scale = self._obj_info[3]
        # object collision frame wrt object link frame
        collision_pose = self._obj_info[5]
        collision_orn = self._obj_info[6]
        collision_matrix = p.getMatrixFromQuaternion(collision_orn)
        collision_dcm = np.array([collision_matrix[0:3], collision_matrix[3:6], collision_matrix[6:9]])

        # trasform object pose from pybullet world to icub world (on the hip)
        w_robot_T_w_py = p.invertTransform(self._robot_base_pose[0], self._robot_base_pose[1])
        w_robot_T_obj = p.multiplyTransforms(w_robot_T_w_py[0], w_robot_T_w_py[1], obj_pos, p.getQuaternionFromEuler(obj_orn))

        # object transform matrix
        obj_matrix = p.getMatrixFromQuaternion(w_robot_T_obj[1])
        obj_dcm = np.array([obj_matrix[0:3], obj_matrix[3:6], obj_matrix[6:9]])

        # Get points
        obj_mesh = pymesh.load_mesh(self._obj_info[4])
        points = superquadric_bindings.deque_Vector3d()
        for i, v in enumerate(obj_mesh.vertices):
            v0 = v * obj_scale
            v1 = collision_dcm.dot(v0) + collision_pose
            v2 = obj_dcm.dot(v1) + w_robot_T_obj[0]
            points.push_back(v2)
            if i % 1000 is 0:
                p.addUserDebugLine(v2, [v2[0] + 0.001, v2[1], v2[2]], lineColorRGB=[0, 1, 0], lineWidth=4.0,
                                   lifeTime=30, parentObjectUniqueId=1)

        if points.size() >= cfg.sq_model['minimum_points']:
            self._pointcloud.setPoints(points)
            print("new point cloud with {} points".format(self._pointcloud.getNumberPoints()))
            self._visualizer.addPoints(self._pointcloud, False)

            return True
        else:
            return False

    def estimate_superq(self):
        self._superqs = self._sq_estimator.computeSuperq(self._pointcloud)

        # Visualize estimated superq
        sq = superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))
        for i, s in enumerate(self._superqs):
            sq[i] = s
        self._visualizer.addSuperq(sq)

        # trasform superquadric pose from icub world to pybullet world
        sq_out = superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))
        for i, s in enumerate(self._superqs):
            sq_out[i] = s

            # axis angle to quaterion
            quat_sq = axis_angle_to_quaternion((s.axisangle[0][0], s.axisangle[1][0], s.axisangle[2][0], s.axisangle[3][0]))

            w_py_T_sq = p.multiplyTransforms(self._robot_base_pose[0], self._robot_base_pose[1], (s.center[0][0], s.center[1][0], s.center[2][0]), quat_sq)
            vec_aa_sq = quaternion_to_axis_angle(w_py_T_sq[1])

            sq_out[i].setSuperqOrientation(np.array(vec_aa_sq))
            sq_out[i].setSuperqCenter(np.array(w_py_T_sq[0]))

        return sq_out

    def estimate_grasp(self):

        # transform sq from world coordinates to robot base coordinates
        sq = superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))

        for i, s in enumerate(self._superqs):
           sq[i] = s

        grasp_res_hand = self._grasp_estimator.computeGraspPoses(sq)

        self._visualizer.addPoses(grasp_res_hand.grasp_poses)

        self._visualizer.addPlane(self._grasp_estimator.getPlaneHeight())

        # ------> Estimate pose cost <-------- #
        # Compute pose hat
        # ...

        # Refine pose cost
        self._grasp_estimator.refinePoseCost(grasp_res_hand)
        # ------> Select best pose <-------- #
        self._best_grasp_pose = grasp_res_hand.grasp_poses[grasp_res_hand.best_pose]

        if self._grasping_hand is "right":
            self._visualizer.highlightBestPose("right", "right", grasp_res_hand.best_pose)
        elif self._grasping_hand is "left":
            self._visualizer.highlightBestPose("left", "left", grasp_res_hand.best_pose)

        # transform grasp pose from from icub world to pybullet world
        # axis angle to quaterion
        quat_gp_icub = axis_angle_to_quaternion((self._best_grasp_pose.axisangle[0][0], self._best_grasp_pose.axisangle[1][0],
                                            self._best_grasp_pose.axisangle[2][0], self._best_grasp_pose.axisangle[3][0]))

        pos_gp_icub = (self._best_grasp_pose.position[0][0], self._best_grasp_pose.position[1][0],
                                          self._best_grasp_pose.position[2][0])

        if self._grasping_hand is "right":
            w_icub_T_gp_py = p.multiplyTransforms(pos_gp_icub, quat_gp_icub,
                                                  (0, 0, 0), p.getQuaternionFromEuler(self._icub_hand_right_orn))
        else:
            w_icub_T_gp_py = p.multiplyTransforms(pos_gp_icub, quat_gp_icub,
                                                  (0, 0, 0), p.getQuaternionFromEuler(self._icub_hand_left_orn))

        w_py_T_gp_py = p.multiplyTransforms(self._robot_base_pose[0], self._robot_base_pose[1],
                                            w_icub_T_gp_py[0], w_icub_T_gp_py[1])

        gp_py_orn = p.getEulerFromQuaternion(w_py_T_gp_py[1])

        gp_out = [w_py_T_gp_py[0][0], w_py_T_gp_py[0][1], w_py_T_gp_py[0][2], gp_py_orn[0], gp_py_orn[1], gp_py_orn[2]]

        return gp_out

    def check_object_moved(self, obj_pose):
        if goal_distance(np.array(obj_pose[:3]), np.array(self._obj_pose[:3])) > 0.02 or \
                goal_distance(np.array(obj_pose[3:6]), np.array(self._obj_pose[3:6])) > 0.5:
            return True

        return False

    def get_next_action(self, hand_pose):
        next_action = hand_pose
        #if goal_distance(np.array(hand_pose[:3]), np.array(self._best_grasp_pose[:3]))>0.05:
        #    next_action[:3] = self._compute_approach_path()
        #else:
        #    pass # here something like close fingers

        return next_action

    def _compute_approach_path(self):
        # linear path from initial to grasping pose
        self._i_path = min(self._i_path+0.1, 1)
        p = self._starting_pose[:3] + self._i_path * (self._best_grasp_pose[:3] - self._starting_pose[:3])
        # print("starting position {} next position {} final position {}".format(self._starting_pose[:3], p, self._best_grasp_pose[:3]))
        return p