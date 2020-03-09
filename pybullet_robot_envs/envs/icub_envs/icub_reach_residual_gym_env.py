import os, inspect
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import math as m
import quaternion

from pybullet_robot_envs.envs.icub_envs.icub_env_with_hands import iCubHandsEnv
from pybullet_robot_envs.envs.icub_envs.icub_env import iCubEnv
from pybullet_robot_envs.envs.world_envs.ycb_fetch_env import get_ycb_objects_list, YcbWorldFetchEnv
from pybullet_robot_envs.envs.icub_envs.superq_grasp_planner import SuperqGraspPlanner
from pybullet_robot_envs.envs.utils import goal_distance, quat_multiplication, axis_angle_to_quaternion, quaternion_to_axis_angle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

DO_LOGGING = False


class iCubReachResidualGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 log_file=os.path.join(currentdir),
                 action_repeat=20,
                 control_arm='l',
                 control_orientation=1,
                 control_eu_or_quat=0,
                 obj_name=None,
                 obj_pose_rnd_std=0.05,
                 noise_pcl=0.00,
                 renders=False,
                 max_steps=1000,
                 use_superq=1,
                 n_control_pt=4,
                 r_weights=(-5, -10, 10)):

        self._time_step = 1. / 240.  # 4 ms

        self._control_arm = control_arm
        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._action_repeat = action_repeat
        self._n_control_pt = n_control_pt
        self._observation = []
        self._r_weights = r_weights

        if obj_name is not None:
            self._obj_name = get_ycb_objects_list()[obj_name]
        else:
            self._obj_name = None

        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self._t_grasp, self._t_lift = 0, 0
        self._obj_pose_rnd_std = obj_pose_rnd_std
        self._noise_pcl = noise_pcl
        self._last_frame_time = 0
        self._use_superq = use_superq
        self._distance_threshold = 0.05

        self._log_file = []
        self._log_file_path = []
        if DO_LOGGING:
            self._log_file_path.append(os.path.join(log_file, 'nominal.txt'))
            self._log_file_path.append(os.path.join(log_file, 'learned.txt'))
            self._log_file.append(open(self._log_file_path[0], "w+"))
            self._log_file.append(open(self._log_file_path[1], "w+"))

            self._log_file[0].close()
            self._log_file[1].close()

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
            self._cid = p.connect(p.SHARED_MEMORY)
            if self._cid < 0:
                self._cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.0, -0.0, -0.0])
        else:
            self._cid = p.connect(p.DIRECT)

        # Load robot
        self._robot = iCubHandsEnv(use_IK=1, control_arm=self._control_arm,
                                   control_orientation=self._control_orientation,
                                   control_eu_or_quat=self._control_eu_or_quat)

        # Load world environment
        if self._obj_name is None:
            obj_name = get_ycb_objects_list()[0]
        else:
            obj_name = self._obj_name
        self._world = YcbWorldFetchEnv(obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                       workspace_lim=self._robot._workspace_lim,
                                       control_eu_or_quat=self._control_eu_or_quat)

        # Load base controller
        self._base_controller = SuperqGraspPlanner(self._robot.robot_id, self._world.obj_id,
                                                   robot_name='icub_hands',
                                                   object_name=obj_name,
                                                   render=self._renders,
                                                   grasping_hand=self._control_arm,
                                                   noise_pcl=self._noise_pcl)

        # limit iCub workspace to table plane
        self._robot._workspace_lim[2][0] = self._world.get_table_height()

        self._superqs = []
        self._grasp_pose = []

        # initialize simulation environment
        self._first_call = 1
        self.seed()
        self.reset()
        self._first_call = 0

        # Define spaces
        self.observation_space, self.action_space = self.create_spaces()

    def create_spaces(self):
        # Configure observation limits
        obs, obs_lim = self.get_extended_observation()

        observation_low = []
        observation_high = []
        for el in obs_lim:
            observation_low.extend([el[0]])
            observation_high.extend([el[1]])

        # Configure the observation space
        observation_space = spaces.Box(np.array(observation_low), np.array(observation_high), dtype='float32')

        # Configure action space
        action_dim = self._robot.get_action_dim()
        action_bound = 1
        action_high = np.array([0.03, 0.03, 0.03, 0.785, 0.2, 1])
        action_low = np.array([-0.03, -0.03, -0.03, -0.785, -0.2, -1])
        action_space = spaces.Box(action_low, action_high, dtype='float32')

        return observation_space, action_space

    def reset(self):
        if DO_LOGGING:
            if self.logId is not None:
                p.stopStateLogging(self.logId)
            if self.logId_ct is not None:
                p.stopStateLogging(self.logId_ct)
            print("logging closed")

            if not self._log_file[0].closed:
                self._log_file[0].close()
            self._log_file[0] = open(self._log_file_path[0], "a+")

            if not self._log_file[1].closed:
                self._log_file[1].close()
            self._log_file[1] = open(self._log_file_path[1], "a+")

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        self._env_step_counter = 0

        p.setGravity(0, 0, -9.8)

        self._robot.reset()

        # Let the world run for a bit
        for _ in range(50):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._time_step)

        self._robot.pre_grasp()

        obj_name = get_ycb_objects_list()[self.np_random.randint(0, 3)] if self._obj_name is None else self._obj_name
        self._world._obj_name = obj_name
        print("obj_name {}".format(obj_name))

        self._world.reset()
        # Let the world run for a bit
        for _ in range(200):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._time_step)

        self._robot.debug_gui()
        self._world.debug_gui()
        robot_obs, _ = self._robot.get_observation()

        # if self._first_call:
        self._base_controller.reset(robot_id=self._robot.robot_id, obj_id=self._world.obj_id, object_name=obj_name,
                                    starting_pose=self._robot._home_hand_pose, n_control_pt=self._n_control_pt)

        self._base_controller.set_robot_base_pose(p.getBasePositionAndOrientation(self._robot.robot_id))

        self.compute_grasp_pose()

        self._base_controller.compute_approach_path()

        self.debug_gui()
        p.stepSimulation()

        robot_obs, _ = self._robot.get_observation()
        world_obs, _ = self._world.get_observation()

        self._t_grasp, self._t_lift = 0, 0

        obs, _ = self.get_extended_observation()

        if DO_LOGGING:
            print("------------------------>>>>>start logging")

            self.logId = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "log_successful_grasp.bin")
            self.logId_ct = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "log_successful_grasp_ct.bin",
                                                bodyUniqueIdA=self._robot.robot_id,
                                                bodyUniqueIdB=self._world.obj_id)

        return obs

    def compute_grasp_pose(self):

        self._base_controller.set_object_info(self._world.get_object_shape_info())

        # TO DO: add check on outputs!
        world_obs, _ = self._world.get_observation()
        if self._control_eu_or_quat is 0:
            obj_pose = world_obs[:3] + list(p.getQuaternionFromEuler(world_obs[3:6]))
        else:
            obj_pose = world_obs[:3] + world_obs[3:7]

        ok = self._base_controller.compute_object_pointcloud(obj_pose)
        if not ok:
            print("Can't get good point cloud of the object")
            return self.reset()

        ok = self._base_controller.estimate_superq()
        if not ok:
            print("can't compute good superquadrics")
            return self.reset()

        ok = self._base_controller.estimate_grasp()
        if not ok:
            print("can't compute any grasp pose")
            return self.reset()

        self._superqs = self._base_controller.get_superqs()
        self._grasp_pose = self._base_controller.get_grasp_pose()

        print("object pose: {}".format(world_obs))
        print("superq pose: {} {}".format(self._superqs[0].center, self._superqs[0].ea))
        print("grasp pose: {}".format(self._grasp_pose))

        if self._renders:
            self._base_controller._visualizer.visualize()

    def get_extended_observation(self):
        self._observation = []
        observation_lim = []

        # get observation form robot and world
        robot_observation, robot_obs_lim = self._robot.get_observation()

        if self._control_eu_or_quat is 0:
            r_quat = p.getQuaternionFromEuler(robot_observation[3:6])
        else:
            r_quat = robot_observation[3:7]

        self._observation.extend(list(robot_observation))
        observation_lim.extend(robot_obs_lim)

        self._observation.extend(list(self._grasp_pose.copy()))
        observation_lim.extend([[-1, 1], [-1, 1], [-1, 1],
                                [-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi],
                                [-2 * m.pi, 2 * m.pi]])

        # get superquadric params of dimension and shape
        if self._use_superq:
            # get superquadric params
            sq_pos = [self._superqs[0].center[0][0], self._superqs[0].center[1][0], self._superqs[0].center[2][0]]
            sq_quat = axis_angle_to_quaternion((self._superqs[0].axisangle[0][0], self._superqs[0].axisangle[1][0],
                                                self._superqs[0].axisangle[2][0], self._superqs[0].axisangle[3][0]))
            sq_eu = p.getEulerFromQuaternion(sq_quat)
            sq_dim = self._superqs[0].dim
            sq_exp = self._superqs[0].exp

            #
            self._observation.extend(list(sq_pos))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

            #
            if self._control_eu_or_quat is 0:
                self._observation.extend(list(sq_eu))
                observation_lim.extend([[-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi]])
            else:
                self._observation.extend(list(sq_quat))
                observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

            #
            self._observation.extend([sq_dim[0][0], sq_dim[1][0], sq_dim[2][0],
                                      sq_exp[0][0], sq_exp[1][0]])
            # check dim limits of sq dim params
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [0, 2], [0, 2]])

            # relative superq position wrt hand c.o.m. frame
            inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3], r_quat)
            sq_pos_in_hand, sq_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn,
                                                                  sq_pos, sq_quat)

            self._observation.extend(list(sq_pos_in_hand))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

            if self._control_eu_or_quat is 0:
                sq_euler_in_hand = p.getEulerFromQuaternion(sq_orn_in_hand)
                self._observation.extend(list(sq_euler_in_hand))
                observation_lim.extend([[-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi]])

            else:
                self._observation.extend(list(sq_orn_in_hand))
                observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        else:
            world_observation, world_obs_lim = self._world.get_observation()

            if self._control_eu_or_quat is 0:
                w_quat = p.getQuaternionFromEuler(world_observation[3:6])
            else:
                w_quat = world_observation[3:7]

            self._observation.extend(list(world_observation))
            observation_lim.extend(world_obs_lim)

            # relative object position wrt hand c.o.m. frame
            inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3], r_quat)
            obj_pos_in_hand, obj_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn,
                                                                    world_observation[:3], w_quat)

            self._observation.extend(list(obj_pos_in_hand))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])

            if self._control_eu_or_quat is 0:
                obj_euler_in_hand = p.getEulerFromQuaternion(obj_orn_in_hand)
                self._observation.extend(list(obj_euler_in_hand))
                observation_lim.extend([[-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi], [-2 * m.pi, 2 * m.pi]])

            else:
                self._observation.extend(list(obj_orn_in_hand))
                observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        return np.array(self._observation), observation_lim

    def apply_action(self, action):
        # process action and send it to the robot

        if self._renders:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._action_repeat * self._time_step - time_spent

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        # set new action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # action[:3] = np.clip(action[:3], -0.05, 0.05)
        pos_action = action[:3]
        if self._control_eu_or_quat is 0:
            # action[3] = np.clip(action[3], -0.785, 0.785)
            # action[4] = np.clip(action[4], -0.2, 0.2)
            quat_action = p.getQuaternionFromEuler(action[3:6])
        else:
            quat_action = action[3:7]
            if quat_action[0] == 0 and quat_action[1] == 0 and quat_action[2] == 0:
                quat_action[3] = 1

        # get action from base controller
        base_action, done = self._base_controller.get_next_action()

        final_action_pos = np.add(base_action[0], pos_action)
        final_action_quat = np.quaternion(base_action[1][3], base_action[1][0], base_action[1][1], base_action[1][2]) * \
                            np.quaternion(quat_action[3], quat_action[0], quat_action[1], quat_action[2])

        final_action_quat = quaternion.as_float_array(final_action_quat)
        final_action_quat_1 = [final_action_quat[1], final_action_quat[2], final_action_quat[3], final_action_quat[0]]

        # final_action = np.add(base_action[0].tolist() + base_action[1].tolist(), action[:6])

        for _ in range(self._action_repeat):
            self._robot.apply_action(final_action_pos.tolist() + final_action_quat_1)
            self._robot.pre_grasp()
            p.stepSimulation()
            if self._renders:
                time.sleep(self._time_step)

            w_obs, _ = self._world.get_observation()
            r_obs, _ = self._robot.get_observation()

            if self._termination(w_obs, r_obs):
                break

            self._env_step_counter += 1

        # dump data
        if DO_LOGGING:
            self.dump_data([base_action, [final_action_pos.tolist() + final_action_quat_1]])

        return final_action_pos.tolist() + final_action_quat_1 + base_action[2]

    def step(self, action):

        # apply action on the robot
        applied_action = self.apply_action(action)

        obs, _ = self.get_extended_observation()

        w_obs, _ = self._world.get_observation()
        r_obs, _ = self._robot.get_observation()

        info = {
            'is_success': self._is_success(w_obs, r_obs),
        }

        done = self._termination(w_obs, r_obs)
        reward = self._compute_reward(w_obs, r_obs)

        # print("reward")
        # print(reward)

        return obs, np.array(reward), np.array(done), info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._world.seed(seed)
        self._robot.seed(seed)
        self._base_controller.seed(seed)
        return [seed]

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self._p.getBasePositionAndOrientation(self._robot.robot_id)

        cam_dist = 1.3
        cam_yaw = 180
        cam_pitch = -40
        RENDER_HEIGHT = 720
        RENDER_WIDTH = 960

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=cam_dist,
                                                                yaw=cam_yaw,
                                                                pitch=cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self, w_obs, r_obs):

        # early termination if object falls
        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
        else:
            eu = w_obs[3:6]

        # cost: object falls
        if self._object_fallen(eu[0], eu[1]):
            print("FALLEN")
            return np.float32(1.)

        # rew 1: distance between hand and grasp pose
        d = goal_distance(np.array(r_obs[:3]), np.array(self._grasp_pose[:3]))
        if d <= self._distance_threshold and self._t_grasp >= 1:
            print("SUCCESS")
            return np.float32(1.)

        if self._env_step_counter > self._max_steps:
            print("MAX STEPS")
            return np.float32(1.)

        # here check lift for termination
        # if self._object_lifted(world_obs[2], world_obs[-1]) and self._t_lift >= 2:
        #    print("SUCCESS")
        #    return np.float32(1.)

        return np.float32(0.)

    def _is_success(self, w_obs, r_obs):
        d = goal_distance(np.array(r_obs[:3]), np.array(self._grasp_pose[:3]))
        if d <= self._distance_threshold and self._t_grasp >= 1:
            return np.float32(1.)
        else:
            return np.float32(0.)


    def _compute_reward(self, w_obs, r_obs):
        c1, c2, r = np.float32(0.0), np.float32(0.0), np.float32(0.0)

        # cost 1: object touched
        if self._world.check_contact(self._robot.robot_id):
            c1 = self._r_weights[0]

        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
        else:
            eu = w_obs[3:6]

        if self._object_fallen(eu[0], eu[1]):
            c2 = self._r_weights[1]

        # rew 1: distance between hand and grasp pose
        # Compute distance between goal and the achieved goal.
        d = goal_distance(np.array(r_obs[:3]), np.array(self._grasp_pose[:3]))
        if d <= self._distance_threshold:
            r += self._r_weights[2]
            self._t_grasp += self._time_step * self._action_repeat
        else:
            self._t_grasp = 0

        if d <= self._distance_threshold and self._t_grasp >= 1:
            r = np.float32(100.0)

        reward = r + c1 + c2

        return reward

    def _object_fallen(self, obj_roll, obj_pitch):
        return obj_roll <= -0.785 or obj_roll >= 0.785 or obj_pitch <= -0.785 or obj_pitch >= 0.785

    def _object_lifted(self, z_obj, h_target, atol=0.05):
        return z_obj >= h_target - atol

    def debug_gui(self):

        quat_sq = axis_angle_to_quaternion((self._superqs[0].axisangle[0][0], self._superqs[0].axisangle[1][0],
                                            self._superqs[0].axisangle[2][0], self._superqs[0].axisangle[3][0]))

        matrix = p.getMatrixFromQuaternion(quat_sq)
        dcm = np.array([matrix[0:3], matrix[3:6], matrix[6:9]])
        pose = self._superqs[0].center + [[0.03], [0], [0]]
        np_pose = np.array([pose[0][0], pose[1][0], pose[2][0]])
        pax = np_pose + np.array(list(dcm.dot([0.1, 0, 0])))
        pay = np_pose + np.array(list(dcm.dot([0, 0.1, 0])))
        paz = np_pose + np.array(list(dcm.dot([0, 0, 0.1])))

        p.addUserDebugLine(pose, pax.tolist(), [1, 0, 0])
        p.addUserDebugLine(pose, pay.tolist(), [0, 1, 0])
        p.addUserDebugLine(pose, paz.tolist(), [0, 0, 1])

        pose = self._grasp_pose[:3]

        matrix = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self._grasp_pose[3:6]))
        dcm = np.array([matrix[0:3], matrix[3:6], matrix[6:9]])
        np_pose = np.array(list(pose))
        pax = np_pose + np.array(list(dcm.dot([0.1, 0, 0])))
        pay = np_pose + np.array(list(dcm.dot([0, 0.1, 0])))
        paz = np_pose + np.array(list(dcm.dot([0, 0, 0.1])))

        p.addUserDebugLine(pose, pax.tolist(), [1, 0, 0])
        p.addUserDebugLine(pose, pay.tolist(), [0, 1, 0])
        p.addUserDebugLine(pose, paz.tolist(), [0, 0, 1])

    def dump_data(self, data):
        if len(data) is 2:
            for ii in data[0]:
                for i in ii:
                    self._log_file[0].write(str(i))
                    self._log_file[0].write(" ")
            self._log_file[0].write("\n")
            for j in data[1]:
                self._log_file[1].write(str(j))
                self._log_file[1].write(" ")
            self._log_file[1].write("\n")