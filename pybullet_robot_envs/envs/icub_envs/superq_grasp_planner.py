import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, '/home/erampone/workspace/INSTALL/lib/superquadriclib/bindings')

import numpy as np
import quaternion
import pybullet as p
import math as m
from pybullet_robot_envs.envs.utils import goal_distance, axis_angle_to_quaternion, quaternion_to_axis_angle, sph_coord

import pymesh

import superquadric_bindings
from superquadric_bindings import PointCloud, SuperqEstimatorApp, GraspEstimatorApp, Visualizer
import config_superq_grasp_planner as cfg

class SuperqGraspPlanner:

    def __init__(self, icub_id, obj_id,
                 robot_base_pose=((0.0,) * 3, (0.0,)*4),
                 grasping_hand='l', noise_pcl=0.02, render=True):

        self._grasping_hand = grasping_hand
        self._noise_pcl = noise_pcl
        self._superqs = superquadric_bindings.vector_superquadric()
        self._robot_base_pose = robot_base_pose
        # offset between icub hand's ref.frame in PyBullet and on the real robot --> TO DO: make it less hard coded
        self._icub_hand_right_orn = [0.0, -m.pi/2, m.pi/2]
        self._icub_hand_left_orn =[0.0, -m.pi/2, m.pi/2]
        self._starting_pose = []
        self._best_grasp_pose = []
        self._approach_path = []
        self._action = [np.zeros(3), np.zeros(3), np.zeros(1)]
        self._obj_info = []
        self._icub_id = icub_id
        self._obj_id = obj_id
        self._render = render
        if self._render:
            self._visualizer = Visualizer()
        self._pointcloud = PointCloud()
        self._sq_estimator = SuperqEstimatorApp()
        self._grasp_estimator = GraspEstimatorApp()
        self._gp_reached = 0

        # initialize
        self.reset(self._icub_id, self._obj_id)

    def reset(self, robot_id, obj_id, starting_pose=np.array(np.zeros(6))):

        self._starting_pose = starting_pose
        self._best_grasp_pose = []
        self._approach_path = []
        self._action = [np.zeros(3), np.zeros(3), np.zeros(1)]
        self._obj_info = []
        self._icub_id = robot_id
        self._obj_id = obj_id

        #self._grasping_hand = cfg.mode['control_arms']

        if self._render:
            self._visualizer.setPosition(cfg.visualizer['x'], cfg.visualizer['y'])
            self._visualizer.setSize(cfg.visualizer['width'], cfg.visualizer['height'])

            self._visualizer.resetPoints()
            self._visualizer.resetSuperq()
            self._visualizer.resetPoses()

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
        self._grasp_estimator.SetStringValue("left_or_right", "right" if self._grasping_hand is 'r' else "left")
        self._grasp_estimator.setVector("plane", np.array(cfg.sq_grasp['plane_table']))
        self._grasp_estimator.setVector("displacement", np.array(cfg.sq_grasp['displacement']))
        self._grasp_estimator.setVector("hand", np.array(cfg.sq_grasp['hand_sq']))

        return

    def set_robot_base_pose(self, pose):
        self._robot_base_pose = pose

    def set_object_info(self, obj_info):
        self._obj_info = obj_info

    def compute_object_pointcloud(self, obj_pose):
        # Create object's point cloud from its mesh

        if not self._obj_info:
            print("compute_object_pointcloud(): no object information provided. Please call set_object_info() first.")
            return False

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

        # trasform object pose from pybullet world to icub world (on the hip)
        w_robot_T_w_py = p.invertTransform(self._robot_base_pose[0], self._robot_base_pose[1])
        w_robot_T_obj = p.multiplyTransforms(w_robot_T_w_py[0], w_robot_T_w_py[1], obj_pos, obj_quat)

        # object transform matrix
        obj_matrix = p.getMatrixFromQuaternion(w_robot_T_obj[1])
        w_robot_R_obj = np.array([obj_matrix[0:3], obj_matrix[3:6], obj_matrix[6:9]])

        # Get points
        obj_mesh = pymesh.load_mesh(self._obj_info[4])

        # Create gaussian noise to add to the point's distribution
        mu, sigma = 0.0, self._noise_pcl
        noise = np.random.normal(mu, sigma, [obj_mesh.vertices.size, 3])

        points = superquadric_bindings.deque_Vector3d()
        colors = superquadric_bindings.vector_vector_uchar()
        counter = 1
        rnd = 0
        for i, v in enumerate(obj_mesh.vertices):
            v0 = v * obj_scale
            v1 = collision_dcm.dot(v0) + collision_pose
            v2 = w_robot_R_obj.dot(v1)
            sph_vec = sph_coord(v2[0], v2[1], v2[2])
            # sample only points visible to the robot eyes, to simulate partial observability of the object
            if sph_vec[1] <= m.pi/4 or -m.pi/2 <= sph_vec[2] <= m.pi/2:
                v3 = v2 + w_robot_T_obj[0]

               # if np.random.random()<0.2:
                #if rnd > 0 and counter > 0:
                 #   counter -= 1
                v3 += noise[i]
                    #continue
                #else:
                #    counter = 10000
                #    rnd = np.random.random() < 0.01

                points.push_back(v3)
                colors.push_back([255, 255, 0])
                # if i % 100 is 0:
                #    p.addUserDebugLine(v3, [v3[0] + 0.001, v3[1], v3[2]], lineColorRGB=[0, 1, 0], lineWidth=4.0,
                #                       lifeTime=30, parentObjectUniqueId=0)

        if points.size() >= cfg.sq_model['minimum_points']:
            self._pointcloud.setPoints(points)
            self._pointcloud.setColors(colors)
            print("new point cloud with {} points".format(self._pointcloud.getNumberPoints()))
            if self._render:
                self._visualizer.addPoints(self._pointcloud, False)

            return True
        else:
            return False

    def estimate_superq(self):
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

        # visualize them
        if self._render:
            self._visualizer.resetPoses()
            self._visualizer.addPoses(grasp_res_hand.grasp_poses)
            self._visualizer.addPlane(self._grasp_estimator.getPlaneHeight())

        # ------> Estimate pose cost <-------- #
        # Compute pose hat
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
                                                  (0, 0, 0), p.getQuaternionFromEuler(self._icub_hand_right_orn))
        elif self._grasping_hand is 'l':
            w_icub_T_gp_py = p.multiplyTransforms(pos_gp_icub, quat_gp_icub,
                                                  (0, 0, 0), p.getQuaternionFromEuler(self._icub_hand_left_orn))

        w_py_T_gp_py = p.multiplyTransforms(self._robot_base_pose[0], self._robot_base_pose[1],
                                            w_icub_T_gp_py[0], w_icub_T_gp_py[1])

        gp_py_orn = p.getEulerFromQuaternion(w_py_T_gp_py[1])

        gp_out = [w_py_T_gp_py[0][0], w_py_T_gp_py[0][1], w_py_T_gp_py[0][2], gp_py_orn[0], gp_py_orn[1], gp_py_orn[2]]
        self._best_grasp_pose = gp_out

        return True

    def get_superqs(self):
        # trasform superquadric pose from icub world to pybullet world
        sq_out = superquadric_bindings.vector_superquadric(np.size(self._superqs, 0))
        for i, s in enumerate(self._superqs):
            sq_out[i] = s

            # axis angle to quaterion
            quat_sq = axis_angle_to_quaternion(
                (s.axisangle[0][0], s.axisangle[1][0], s.axisangle[2][0], s.axisangle[3][0]))

            w_py_T_sq = p.multiplyTransforms(self._robot_base_pose[0], self._robot_base_pose[1],
                                             (s.center[0][0], s.center[1][0], s.center[2][0]), quat_sq)
            vec_aa_sq = quaternion_to_axis_angle(w_py_T_sq[1])

            sq_out[i].setSuperqOrientation(np.array(vec_aa_sq))
            sq_out[i].setSuperqCenter(np.array(w_py_T_sq[0]))

        return sq_out

    def get_grasp_pose(self):
        return self._best_grasp_pose.copy()

    def compute_approach_path(self):
        # reset current path
        self._approach_path = []

        # Transform from starting pose to grasp pose
        sp_inv_pose = p.invertTransform(self._starting_pose[:3], p.getQuaternionFromEuler(self._starting_pose[3:6]))
        sp_P_gp = p.multiplyTransforms(sp_inv_pose[0], sp_inv_pose[1],
                                       self._best_grasp_pose[:3], p.getQuaternionFromEuler(self._best_grasp_pose[3:6]))

        # linear path from initial to grasping pose
        dist_object = np.linalg.norm(sp_P_gp[0])
        dist_intra_path_points = 0.02
        n_pt = int(dist_object / dist_intra_path_points)
        i_path = [i / n_pt for i in range(0, n_pt + 1)]

        for idx in i_path[-10:]:
            # --- Position --- #
            delta_pos = idx * np.array(sp_P_gp[0])

            # --- Orientation --- #
            delta_ax = quaternion_to_axis_angle(sp_P_gp[1])
            delta_ax[-1] = idx * delta_ax[-1]
            delta_quat = axis_angle_to_quaternion(delta_ax)

            next_pose = p.multiplyTransforms(self._starting_pose[:3], p.getQuaternionFromEuler(self._starting_pose[3:6]),
                                            delta_pos, delta_quat)

            self._approach_path.append(next_pose)

        self._debug_gui(self._approach_path)

        self._approach_path.reverse()
        return True

    def get_next_action(self, robot_obs, world_obs, atol=1e-2):
        """
        Returns
        -------
        action : [np.array([float]*3), np.array([float]*3), np.array([float])] - (delta_pos + delta_euler + open/close fingers)
        """
        hand_pose = robot_obs[:6]
        obj_pose = world_obs[:6]
        tg_h_obj = 0.85

        # Check if done
        if obj_pose[2] >= (tg_h_obj - atol*2):
            #print("DONE")
            self._action = [self._action[0], self._action[1], np.array([-0.5])]
            return self._action

        # Approach the object
        if self._approach_path:
            #print("APPROACH")
            next_pose = self._approach_path.pop()
            self._action = [np.array(next_pose[0]), np.array(next_pose[1]), np.array([1])]
            return self._action

        # Grasp the object
        if not self._object_grasped():
            #print("GRASP")
            self._action = [self._action[0], self._action[1], np.array([-1])]
            return self._action

        # Lift the object
       # print("LIFT")
        action = self._action
        action[0][2] = tg_h_obj
        action = [action[0], action[1], np.array([-1])]
        return action

    def _next_pose(self, hand_pose):  # not used
        if not self._approach_path:
            print("Approach path is empty. Can't get next way-point")
            return [np.zeros(3), np.zeros(3), np.array([0.5])]

        self._gp_reached = 0
        # Find the nearest way-point in the approach trajectory
        dist = 1000
        idx = 0
        for i, pose in enumerate(self._approach_path):
            temp_dist = goal_distance(np.array(hand_pose[:3]), np.array(pose[0]))
            if temp_dist < dist:
                dist = temp_dist
                idx = min(i+1, len(self._approach_path)-1)

        if idx is len(self._approach_path)-1:
            self._gp_reached = 1

        next_pose = self._approach_path[idx]

        # relative position from current hand position to next target position
        rel_pos = np.subtract(next_pose[0], hand_pose[:3])

        # quaternion of current pose
        q_cp = p.getQuaternionFromEuler(hand_pose[3:6])
        w_q_cp = np.quaternion(q_cp[3], q_cp[0], q_cp[1], q_cp[2])

        # quaternion of next target pose
        w_q_tp = np.quaternion(next_pose[1][3], next_pose[1][0], next_pose[1][1], next_pose[1][2])

        # relative quaternion from starting to grasping pose
        cp_q_tp = np.conj(w_q_cp) * w_q_tp
        cp_q_tp = quaternion.as_float_array(cp_q_tp)
        cp_eu_tp = p.getEulerFromQuaternion([cp_q_tp[1], cp_q_tp[2], cp_q_tp[3], cp_q_tp[0]])

        return [rel_pos, np.array(list(cp_eu_tp)), np.array([0.5])]

    def _object_approached(self, hand_pose, obj_pose, atol): #not used
        return goal_distance(np.array(hand_pose[:3]), np.array(obj_pose[:3])) < 0.1

    def _object_grasped(self):
        # check if there is a constraint between hand and object
        id = p.getConstraintUniqueId(p.getNumConstraints()-1)
        info = p.getConstraintInfo(id)
        if info[0] is self._icub_id and info[2] is self._obj_id:
            return True

        return False

    def _debug_gui(self, points):
        for pt in points:
            pose = pt[0]
            p.addUserDebugLine([pose[0], pose[1], pose[2]], [pose[0] + 0.1, pose[1], pose[2]], [1, 0, 0], lifeTime=0)
            p.addUserDebugLine([pose[0], pose[1], pose[2]], [pose[0], pose[1] + 0.1, pose[2]], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine([pose[0], pose[1], pose[2]], [pose[0], pose[1], pose[2] + 0.1], [0, 0, 1], lifeTime=0)
