#!/usr/bin/env python3

import os, inspect
import numpy as np
import pybullet as p
import math as m
import time
import trimesh
import superquadric_bindings
from gym.utils import seeding

import rospy
# import tf_conversions
import std_msgs.msg
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from pybullet_robot_envs.envs.utils import goal_distance, axis_angle_to_quaternion, quaternion_to_axis_angle, sph_coord
from superquadric_bindings import PointCloud, SuperqEstimatorApp, GraspEstimatorApp, Visualizer
from pybullet_robot_envs.envs import config_superq_grasp_planner as cfg


def get_real_icub_to_sim_robot():
    R = {
        cfg.robots[0]: [[-m.pi/2, 0.0, m.pi], [m.pi/2, 0.0, 0.0]],
        cfg.robots[1]: [[0.0, -m.pi/2, m.pi/2], [0.0, -m.pi/2, m.pi/2]],
        cfg.robots[2]: [[0., 0., -m.pi/2], [[m.pi, 0., -m.pi/2]]]
    }
    return R


class SuperqGraspPlanner:

    def __init__(self, robot_id, obj_id, robot_name, object_name=None,
                 robot_base_pose=((0.0,) * 3, (0.0,)*4),
                 grasping_hand='r',
                 noise_pcl=0.02,
                 render=True):

        self._grasping_hand = grasping_hand
        self._noise_pcl = noise_pcl
        self._robot_base_pose = robot_base_pose
        self._real_icub_R_sim_robot = get_real_icub_to_sim_robot()[robot_name]
        self._starting_pose = []
        self._best_grasp_pose = []
        self._approach_path = []
        self._n_control_pt = 4
        self._action = [np.zeros(3), np.array([0, 0, 0, 1]), np.zeros(1)]
        self._obj_info = []

        self._robot_id = robot_id
        self._obj_id = obj_id
        self._robot_name = robot_name
        self._object_name = object_name
        self._render = render

        self._pointcloud = PointCloud()
        self._superqs = superquadric_bindings.vector_superquadric()
        self._sq_estimator = SuperqEstimatorApp()
        self._grasp_estimator = GraspEstimatorApp()
        if self._render:
            self._visualizer = Visualizer()
        self._gp_reached = 0

        # ------ Set ROS node ------ #
        rospy.init_node('pybullet_sq_grasp_planner', anonymous=True)
        self._pub = rospy.Publisher('pybullet_poses', PoseArray, queue_size=1)
        self._sub = rospy.Subscriber('move_group_plan_waypoints', PoseArray, callback=self.trajectory_callback, queue_size=1)

        rate = rospy.Rate(10)  # 10hz

        self._new_trajectory_msg = False
        self._trajectory_points = []

        # ------ Set Visualizer parameters ------ #
        if self._render:
            self._visualizer.setPosition(cfg.visualizer['x'], cfg.visualizer['y'])
            self._visualizer.setSize(cfg.visualizer['width'], cfg.visualizer['height'])

            self._visualizer.resetPoints()
            self._visualizer.resetSuperq()
            self._visualizer.resetPoses()

        # ------ Set Superquadric Model parameters ------ #
        self._sq_estimator.SetNumericValue("tol", cfg.sq_model['tol'])
        self._sq_estimator.SetIntegerValue("print_level", 0)
        self._sq_estimator.SetIntegerValue("optimizer_points", cfg.sq_model['optimizer_points'])
        self._sq_estimator.SetBoolValue("random_sampling", cfg.sq_model['random_sampling'])

        # ------ Set Superquadric Grasp parameters ------ #
        self._grasp_estimator.SetIntegerValue("print_level", 0)
        self._grasp_estimator.SetNumericValue("tol", cfg.sq_grasp[self._robot_name]['tol'])
        self._grasp_estimator.SetIntegerValue("max_superq", cfg.sq_grasp[self._robot_name]['max_superq'])
        self._grasp_estimator.SetNumericValue("constr_tol", cfg.sq_grasp[self._robot_name]['constr_tol'])
        self._grasp_estimator.SetStringValue("left_or_right", "right" if self._grasping_hand is 'r' else "left")
        self._grasp_estimator.setVector("plane", np.array(cfg.sq_grasp[self._robot_name]['plane_table']))
        self._grasp_estimator.setVector("displacement", np.array(cfg.sq_grasp[self._robot_name]['displacement']))
        self._grasp_estimator.setVector("hand", np.array(cfg.sq_grasp[self._robot_name]['hand_sq']))
        self._grasp_estimator.setMatrix("bounds_right", np.array(cfg.sq_grasp[self._robot_name]['bounds_right']))
        self._grasp_estimator.setMatrix("bounds_left", np.array(cfg.sq_grasp[self._robot_name]['bounds_left']))

        # initialize
        self.seed()
        self.reset(self._robot_id, self._obj_id)

    def reset(self, robot_id, obj_id, object_name=None, starting_pose=np.array(np.zeros(6)), n_control_pt=2):

        self._starting_pose = starting_pose
        self._n_control_pt = n_control_pt
        self._robot_id = robot_id
        self._obj_id = obj_id
        self._object_name = object_name

        # reset variables
        self._best_grasp_pose = []
        self._approach_path = []
        self._action = [np.zeros(3), np.array([0, 0, 0, 1]), np.zeros(1)]
        self._obj_info = []

        # Update object class
        if self._object_name is not None and self._object_name in cfg.objects.keys():
            self._sq_estimator.SetStringValue("object_class", cfg.objects[self._object_name])
        else:
            self._sq_estimator.SetStringValue("object_class", cfg.sq_model['object_class'])

        return

    def set_robot_base_pose(self, pose):
        self._robot_base_pose = [list(pose[0]), list(pose[1])]
        self._robot_base_pose[0][2] -= 0.05  # take into account the displacement between COM and URDF link pose

    def set_object_info(self, obj_info):
        self._obj_info = obj_info

    def compute_object_pointcloud(self, obj_pose):
        # Create object's point cloud from its mesh

        if not self._obj_info:
            print("compute_object_pointcloud(): no object information provided. Please call set_object_info() first.")
            return False

        # reset point cloud object and visualizer
        self._pointcloud.deletePoints()
        if self._render:
            self._visualizer.resetPoints()

        # Current object link pose wrt pybullet world frame
        obj_pos = obj_pose[:3]
        obj_quat = obj_pose[3:7]

        # Object collision shape info
        obj_scale = self._obj_info[3]

        # object collision frame wrt object link frame
        collision_pose = self._obj_info[5]
        collision_orn = self._obj_info[6]
        collision_matrix = p.getMatrixFromQuaternion(collision_orn)
        collision_dcm = np.array([collision_matrix[0:3], collision_matrix[3:6], collision_matrix[6:9]])

        # NOTE: these transform are necessary since the superquadric-lib code consider the icub world frame as reference frame

        # trasform object pose from pybullet world to icub world (on the hip)
        w_py_R_w_icub = p.getQuaternionFromEuler([0, 0, -m.pi])
        w_robot_T_w_py = p.invertTransform(self._robot_base_pose[0], w_py_R_w_icub)

        # object wrt robot world
        w_robot_T_obj = p.multiplyTransforms(w_robot_T_w_py[0], w_robot_T_w_py[1], obj_pos, obj_quat)

        # object transform matrix
        obj_matrix = p.getMatrixFromQuaternion(w_robot_T_obj[1])
        w_robot_R_obj = np.array([obj_matrix[0:3], obj_matrix[3:6], obj_matrix[6:9]])

        # Get points
        encoding = 'utf-8'
        mesh_file_name = (self._obj_info[4]).decode(encoding)
        obj_mesh = trimesh.load(mesh_file_name)

        # Create gaussian noise to add to the points distribution
        mu, sigma = 0.0, self._noise_pcl
        noise = self.np_random.normal(mu, sigma, [obj_mesh.vertices.size, 1])
        #np.random.shuffle(noise)

        points = superquadric_bindings.deque_Vector3d()
        colors = superquadric_bindings.vector_vector_uchar()

        for i, v in enumerate(obj_mesh.vertices):
            v0 = v * obj_scale
            v1 = collision_dcm.dot(v0) + collision_pose
            v2 = w_robot_R_obj.dot(v1)
            sph_vec = sph_coord(v2[0], v2[1], v2[2])
            # sample only points visible to the robot eyes, to simulate partial observability of the object
            if 1: #  sph_vec[1] <= m.pi/8 or -m.pi/2 < sph_vec[2] < m.pi/2 or v1[0] < 0:
                v3 = v2 + w_robot_T_obj[0]
                v3[0] += noise[i]

                points.push_back(v3)
                colors.push_back([255, 255, 0])
                # if i % 100 is 0:
                #     p.addUserDebugLine(v1, [v1[0] + 0.001, v1[1], v1[2]], lineColorRGB=[0, 1, 0], lineWidth=4.0,
                #                       lifeTime=300)

        if points.size() >= cfg.sq_model['minimum_points']:
            self._pointcloud.setPoints(points)
            self._pointcloud.setColors(colors)
            # print("new point cloud with {} points".format(self._pointcloud.getNumberPoints()))
            if self._render:
                self._visualizer.addPoints(self._pointcloud, False)

            return True
        else:
            return False

    def estimate_superq(self):
        # Compute superquadric model from the object's point cloud

        # Check point cloud validity
        if self._pointcloud.getNumberPoints() < cfg.sq_model['minimum_points']:
            print("current object's point cloud has only {} points. Please re-call compute_object_pointcloud()."
                  .format(self._pointcloud.getNumberPoints() ))
            return superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))

        # Compute superquadrics
        self._superqs = self._sq_estimator.computeSuperq(self._pointcloud)

        # Visualize estimated superq
        sq = superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))
        for i, s in enumerate(self._superqs):
            sq[i] = s

        if self._render:
            self._visualizer.resetSuperq()
            self._visualizer.addSuperq(sq)

        return np.size(self._superqs, 0) >= 0

    def estimate_grasp(self):
        # Compute best candidate grasp pose

        self._best_grasp_pose = []

        if np.size(self._superqs, 0) is 0:
            print("No available superquadrics. Please call estimate_superq()")
            return []

        # ------> Compute candidate grasping poses <-------- #
        sq = superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))
        for i, s in enumerate(self._superqs):
            sq[i] = s

        grasp_res_hand = self._grasp_estimator.computeGraspPoses(sq)

        if np.size(grasp_res_hand.grasp_poses, 0) is 0:
            return False

        gp = grasp_res_hand.grasp_poses[0]
        print("grasp poses \n {}".format(gp.position))
        if np.linalg.norm((gp.position[0][0], gp.position[1][0], gp.position[2][0])) == 0.:
            return False

        sq_hand = grasp_res_hand.hand_superq
        pointcloud = PointCloud()
        points = superquadric_bindings.vector_deque_Vector3d()
        points = grasp_res_hand.points_on[0]
        pointcloud.setPoints(points)

        # visualize them
        if self._render:
            self._visualizer.resetPoses()
            self._visualizer.addPoses(grasp_res_hand.grasp_poses)
            self._visualizer.addPlane(self._grasp_estimator.getPlaneHeight())
            self._visualizer.addSuperqHands(sq_hand)
            self._visualizer.addPointsHands(pointcloud)

        # ------> Estimate pose cost <-------- #
        # TODO: Compute pose hat
        # ...

        # Refine pose cost
        self._grasp_estimator.refinePoseCost(grasp_res_hand)

        # ------> Select best pose <-------- #
        best_grasp_pose = grasp_res_hand.grasp_poses[grasp_res_hand.best_pose]

        if self._grasping_hand is 'r' and self._render:
            self._visualizer.highlightBestPose("right", "right", grasp_res_hand.best_pose)
        elif self._grasping_hand is 'l' and self._render:
            self._visualizer.highlightBestPose("left", "left", grasp_res_hand.best_pose)

        # ------> transform grasp pose from icub world to pybullet world coordinates <-------- #

        # axis angle to quaterion
        quat_gp_icub = axis_angle_to_quaternion((best_grasp_pose.axisangle[0][0], best_grasp_pose.axisangle[1][0],
                                                 best_grasp_pose.axisangle[2][0], best_grasp_pose.axisangle[3][0]))

        pos_gp_icub = (best_grasp_pose.position[0][0], best_grasp_pose.position[1][0], best_grasp_pose.position[2][0])

        if self._grasping_hand is 'r':
            w_icub_T_gp_py = p.multiplyTransforms(pos_gp_icub, quat_gp_icub,
                                                  (0, 0, 0), p.getQuaternionFromEuler(self._real_icub_R_sim_robot[0]))
        else:
            w_icub_T_gp_py = p.multiplyTransforms(pos_gp_icub, quat_gp_icub,
                                                  (0, 0, 0), p.getQuaternionFromEuler(self._real_icub_R_sim_robot[1]))

        w_py_R_w_icub = p.getQuaternionFromEuler([0, 0, -m.pi])
        w_py_T_gp_py = p.multiplyTransforms(self._robot_base_pose[0], w_py_R_w_icub,
                                            w_icub_T_gp_py[0], w_icub_T_gp_py[1])

        gp_py_orn = p.getEulerFromQuaternion(w_py_T_gp_py[1])

        gp_out = [w_py_T_gp_py[0][0], w_py_T_gp_py[0][1], w_py_T_gp_py[0][2], gp_py_orn[0], gp_py_orn[1], gp_py_orn[2]]
        self._best_grasp_pose = gp_out

        return True

    def get_superqs(self):
        # Return superquadric model

        # trasform superquadric pose from icub world to pybullet world
        sq_out = superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))
        for i, s in enumerate(self._superqs):
            sq_out[i] = s

            # axis angle to quaterion
            quat_sq = axis_angle_to_quaternion((s.axisangle[0][0], s.axisangle[1][0], s.axisangle[2][0], s.axisangle[3][0]))

            w_py_R_w_icub = p.getQuaternionFromEuler([0, 0, -m.pi])
            w_py_T_sq = p.multiplyTransforms(self._robot_base_pose[0], w_py_R_w_icub,
                                             (s.center[0][0], s.center[1][0], s.center[2][0]), quat_sq)

            vec_aa_sq = quaternion_to_axis_angle(w_py_T_sq[1])

            sq_out[i].setSuperqOrientation(np.array(vec_aa_sq))
            sq_out[i].setSuperqCenter(np.array(w_py_T_sq[0]))

        return sq_out

    def get_grasp_pose(self):
        return self._best_grasp_pose.copy()

    def trajectory_callback(self, ros_poses):
        rospy.loginfo("Received poses: {}".format(len(ros_poses.poses)))

        self._approach_path = []
        link_states = p.getLinkStates(self._robot_id, [8, 11])
        ph_T_w = p.invertTransform(link_states[0][4], link_states[0][5])
        ph_T_pgt = p.multiplyTransforms(ph_T_w[0], ph_T_w[1], link_states[1][4], link_states[1][5])

        for ros_pose in ros_poses.poses:
            r_pose_ph = ((ros_pose.position.x, ros_pose.position.y, ros_pose.position.z),
                        (ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w))

            # ---> Transform from panda_hand to panda_grasptarget
            r_pose_pgt = p.multiplyTransforms(r_pose_ph[0], r_pose_ph[1], ph_T_pgt[0], ph_T_pgt[1])

            # ... and from robot base to pybullet world
            py_pose_pgt = p.multiplyTransforms(self._robot_base_pose[0], self._robot_base_pose[1], r_pose_pgt[0], r_pose_pgt[1])
            self._approach_path.append(py_pose_pgt)

        self._approach_path.reverse()

        self._new_trajectory_msg = True
        return

    def compute_approach_path(self):
        # Send starting pose and grasping pose to moveit to compute the trajectory

        # ---> Express poses  wrt robot base frame
        r_T_py = p.invertTransform(self._robot_base_pose[0], self._robot_base_pose[1])

        # starting pose
        r_T_sp = p.multiplyTransforms(r_T_py[0], r_T_py[1],
                                      self._starting_pose[:3], p.getQuaternionFromEuler(self._starting_pose[3:6]))
        # grasping pose
        r_T_gp = p.multiplyTransforms(r_T_py[0], r_T_py[1],
                                       self._best_grasp_pose[:3], p.getQuaternionFromEuler(self._best_grasp_pose[3:6]))

        # ---> Transform from panda_grasptarget to panda_hand
        link_states = p.getLinkStates(self._robot_id, [8,11])
        pgt_T_w = p.invertTransform(link_states[1][4], link_states[1][5])
        pgt_T_ph = p.multiplyTransforms(pgt_T_w[0], pgt_T_w[1], link_states[0][4], link_states[0][5])

        # starting pose
        r_T_sp_ph = p.multiplyTransforms(r_T_sp[0], r_T_sp[1], pgt_T_ph[0], pgt_T_ph[1])

        # grasping pose
        r_T_gp_ph = p.multiplyTransforms(r_T_gp[0], r_T_gp[1], pgt_T_ph[0], pgt_T_ph[1])

        # ---> Create ROS PoseArray messages
        posearray = PoseArray()
        posearray.header.stamp = rospy.Time.now()  # timestamp of creation of the message
        posearray.header.frame_id = "panda_link0"  # frame id in which the array is published

        # start pose
        pose_msg_start = Pose()

        pose_msg_start.position.x = r_T_sp_ph[0][0]
        pose_msg_start.position.y = r_T_sp_ph[0][1]
        pose_msg_start.position.z = r_T_sp_ph[0][2]
        pose_msg_start.orientation.x = r_T_sp_ph[1][0]
        pose_msg_start.orientation.y = r_T_sp_ph[1][1]
        pose_msg_start.orientation.z = r_T_sp_ph[1][2]
        pose_msg_start.orientation.w = r_T_sp_ph[1][3]

        posearray.poses.append(pose_msg_start)
        pose_msg_goal = Pose()

        # goal pose
        pose_msg_goal.position.x = r_T_gp_ph[0][0]
        pose_msg_goal.position.y = r_T_gp_ph[0][1]
        pose_msg_goal.position.z = r_T_gp_ph[0][2]
        pose_msg_goal.orientation.x = r_T_gp_ph[1][0]
        pose_msg_goal.orientation.y = r_T_gp_ph[1][1]
        pose_msg_goal.orientation.z = r_T_gp_ph[1][2]
        pose_msg_goal.orientation.w = r_T_gp_ph[1][3]

        posearray.poses.append(pose_msg_goal)

        # ---> Send msg to ros topic
        self._new_trajectory_msg = False
        rospy.loginfo(posearray)
        self._pub.publish(posearray)

        # wait for response from moveit
        count = 10
        while not self._new_trajectory_msg and count is not 0:
            count -= 1
            time.sleep(1)

        if count is 0 or self.is_approach_path_empty():
            return False

        self._debug_gui(self._approach_path)
        return True

    def is_last_approach_step(self):
        if len(self._approach_path) == 1:
            return True

        return False

    def is_approach_path_empty(self):
        if len(self._approach_path) == 0:
            return True

        return False

    def get_next_action(self, robot_obs=None, world_obs=None, atol=1e-2):
        """
        State machine that returnes the next action to apply to the robot's hand, based on the current state of the system

        Returns
        -------
        action : [np.array([float]*3), np.array([float]*3), np.array([float])] - (delta_pos + delta_euler + open/close fingers)
        """

        tg_h_obj = 0.9

        done = False

        # Approach the object
        if self._approach_path:
            # print("APPROACH")
            next_pose = self._approach_path.pop()
            self._action = [np.array(next_pose[0]), np.array(next_pose[1]), np.array([-1])]
            return self._action, done

        # Lift the object
        # print("LIFT")
        done = True
        action = self._action
        #action[0][2] = tg_h_obj
        action = [action[0], action[1], np.array([0])]
        return action, done

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _debug_gui(self, points):
        for pt in points:
            matrix = p.getMatrixFromQuaternion(pt[1])
            dcm = np.array([matrix[0:3], matrix[3:6], matrix[6:9]])
            np_pose = np.array(list(pt[0]))
            pax = np_pose + np.array(list(dcm.dot([0.1, 0, 0])))
            pay = np_pose + np.array(list(dcm.dot([0, 0.1, 0])))
            paz = np_pose + np.array(list(dcm.dot([0, 0, 0.1])))

            p.addUserDebugLine(pt[0], pax.tolist(), [1, 0, 0], lifeTime=0)
            p.addUserDebugLine(pt[0], pay.tolist(), [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(pt[0], paz.tolist(), [0, 0, 1], lifeTime=0)
