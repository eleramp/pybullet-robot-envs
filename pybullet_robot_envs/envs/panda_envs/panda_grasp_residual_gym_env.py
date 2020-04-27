import os, inspect
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import math as m
import quaternion

from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv
from pybullet_robot_envs.envs.world_envs.ycb_fetch_env import get_ycb_objects_list, YcbWorldFetchEnv
from pybullet_robot_envs.envs.panda_envs.superq_grasp_planner import SuperqGraspPlanner
from pybullet_robot_envs.envs.utils import goal_distance, quat_multiplication, axis_angle_to_quaternion, quaternion_to_axis_angle, scale_gym_data

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

DO_LOGGING = False

class PandaGraspResidualGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 log_file=os.path.join(currentdir),
                 action_repeat=60,
                 control_orientation=1,
                 control_eu_or_quat=0,
                 normalize_obs=True,
                 obj_name=None,
                 obj_pose_rnd_std=0.05,
                 noise_pcl=0.00,
                 renders=False,
                 max_steps=500,
                 use_superq=1,
                 n_control_pt=2,
                 r_weights=(-5, -10, 10)):

        self._time_step = 1. / 240.  # 4 ms

        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._normalize_obs = normalize_obs
        self._action_repeat = action_repeat
        self._n_control_pt = n_control_pt+1
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
        self._distance_threshold = 0.1
        self._cum_reward = np.float32(0.0)

        self._log_file = []
        self._log_file_path = []
        if DO_LOGGING:
            self._log_file_path.append(os.path.join(log_file, 'eu_hand.txt'))
            self._log_file_path.append(os.path.join(log_file, 'pos_sq_wrt_hand.txt'))
            self._log_file_path.append(os.path.join(log_file, 'eu_sq_wrt_hand.txt'))
            self._log_file.append(open(self._log_file_path[0], "w+"))
            self._log_file.append(open(self._log_file_path[1], "w+"))
            self._log_file.append(open(self._log_file_path[2], "w+"))

            self._log_file[0].close()
            self._log_file[1].close()
            self._log_file[2].close()

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
            self._physics_client_id = p.connect(p.SHARED_MEMORY)
            if self._physics_client_id < 0:
                self._physics_client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.0, -0.0, -0.0], physicsClientId=self._physics_client_id)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        self._traj_client_id = p.connect(p.DIRECT)

        # Load robot
        self._robot = pandaEnv(self._physics_client_id, use_IK=1, control_eu_or_quat=self._control_eu_or_quat)

        self._robot_traj = pandaEnv(self._traj_client_id, use_IK=1, control_eu_or_quat=self._control_eu_or_quat)

        # Load world environment
        if self._obj_name is None:
            obj_name = get_ycb_objects_list()[0]
        else:
            obj_name = self._obj_name

        self._world = YcbWorldFetchEnv(self._physics_client_id,
                                       obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                       workspace_lim=self._robot.get_workspace(),
                                       control_eu_or_quat=self._control_eu_or_quat)

        # Load base controller
        self._base_controller = SuperqGraspPlanner(self._physics_client_id,
                                                   self._traj_client_id,
                                                   self._robot_traj, self._world.obj_id, robot_name='panda',
                                                   render=self._renders,
                                                   grasping_hand='r',
                                                   noise_pcl=self._noise_pcl)

        # limit iCub workspace to table plane
        workspace = self._robot.get_workspace()
        workspace[2][0] = self._world.get_table_height()
        self._robot.set_workspace(workspace)

        self._superqs = []
        self._grasp_pose = []
        self.are_gym_spaces_set = False

        # initialize simulation environment
        self._first_call = 1
        self.seed()
        self.reset()
        self._first_call = 0

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
        action_dim = self._robot.get_action_dimension()
        if action_dim == 6:  # position and orientation (euler angles)
            action_high = np.array([0.04, 0.04, 0.04, 0.2, 0.2, 0.5])
            action_low = np.array([-0.04, -0.04, -0.04, -0.2, -0.2, -0.5])

        elif action_dim == 7:  # position and orientation (quaternion)
            action_high = np.array([0.04, 0.04, 0.04, 1, 1, 1, 1])
            action_low = np.array([-0.04, -0.04, -0.04, -1, -1, -1, -1])
        else:  # only position
            action_high = np.array([0.04, 0.04, 0.04])
            action_low = np.array([-0.04, -0.04, -0.04])

        action_space = spaces.Box(action_low, action_high, dtype='float32')

        return observation_space, action_space

    def reset(self):
        if DO_LOGGING:
            # if self.logId is not None:
            #     p.stopStateLogging(self.logId, physicsClientId=self._physics_client_id)
            # if self.logId_ct is not None:
            #     p.stopStateLogging(self.logId_ct, physicsClientId=self._physics_client_id)
            # print("logging closed")

            if not self._log_file[0].closed:
                self._log_file[0].close()
            self._log_file[0] = open(self._log_file_path[0], "a+")

            if not self._log_file[1].closed:
                self._log_file[1].close()
            self._log_file[1] = open(self._log_file_path[1], "a+")

            if not self._log_file[2].closed:
                self._log_file[2].close()
            self._log_file[2] = open(self._log_file_path[2], "a+")

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._physics_client_id)
        p.setTimeStep(self._time_step, physicsClientId=self._physics_client_id)

        p.resetSimulation(physicsClientId=self._traj_client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._traj_client_id)
        p.setTimeStep(self._time_step, physicsClientId=self._traj_client_id)

        self._env_step_counter = 0

        p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self._traj_client_id)

        self._robot.reset()
        self._robot_traj.reset()

        # Let the world run for a bit
        for _ in range(50):
            p.stepSimulation(physicsClientId=self._physics_client_id)
            p.stepSimulation(physicsClientId=self._traj_client_id)
            if self._renders:
                time.sleep(self._time_step)

        self._robot.pre_grasp()

        # reset world with new object
        obj_name = get_ycb_objects_list()[self.np_random.randint(0, 4)] if self._obj_name is None else self._obj_name
        self._world._obj_name = obj_name
        print("obj_name {}".format(obj_name))

        self._world.reset()
        # Let the world run for a bit
        for _ in range(200):
            p.stepSimulation(physicsClientId=self._physics_client_id)
            if self._renders:
                time.sleep(self._time_step)

        self._robot.debug_gui()
        self._world.debug_gui()
        robot_obs, _ = self._robot.get_observation()

        # if self._first_call:
        self._base_controller.reset(obj_id=self._world.obj_id,
                                    starting_pose=self._robot._home_hand_pose, n_control_pt=self._n_control_pt)

        self._base_controller.set_robot_base_pose(p.getBasePositionAndOrientation(self._robot.robot_id, physicsClientId=self._physics_client_id))

        ok = self.compute_grasp_pose()

        if ok:
            base_action, _ = self._base_controller.get_next_action()
            for _ in range(self._action_repeat*3):
                self._robot.apply_action(base_action[0].tolist() + base_action[1].tolist())
                p.stepSimulation(physicsClientId=self._physics_client_id)
                if self._renders:
                    time.sleep(self._time_step)

        self.debug_gui()
        p.stepSimulation(physicsClientId=self._physics_client_id)


        robot_obs, _ = self._robot.get_observation()
        world_obs, _ = self._world.get_observation()

        self._target_h_lift = world_obs[2] + 0.15

        self._t_grasp, self._t_lift = 0, 0
        self._grasping_step = 5
        self.last_approach_step = False

        # Define spaces if not done already
        if not self.are_gym_spaces_set:
            self.observation_space, self.action_space = self.create_spaces()

        obs, _ = self.get_extended_observation()
        scaled_obs = obs
        if self._normalize_obs:
            scaled_obs = scale_gym_data(self.observation_space, obs)

        if DO_LOGGING:
            self.dump_obs(obs[39:42], obs[6:9], obs[9:12])

        # if DO_LOGGING:
        #     print("------------------------>>>>>start logging")
        #
        #     self.logId = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "log_successful_grasp.bin",
        #                                      physicsClientId=self._physics_client_id)
        #     self.logId_ct = p.startStateLogging(p.STATE_LOGGING_CONTACT_POINTS, "log_successful_grasp_ct.bin",
        #                                         bodyUniqueIdA=self._robot.robot_id,
        #                                         bodyUniqueIdB=self._world.obj_id,
        #                                         physicsClientId=self._physics_client_id)

        return scaled_obs

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
            self.reset()
            return False

        ok = self._base_controller.estimate_superq()
        if not ok:
            print("can't compute good superquadrics")
            self.reset()
            return False

        ok = self._base_controller.estimate_grasp()
        if not ok:
            print("can't compute any grasp pose")
            self.reset()
            return False

        ok = self._base_controller.compute_approach_path()
        if not ok:
            print("can't compute the approach path")
            self.reset()
            return False

        self._superqs = self._base_controller.get_superqs()
        self._grasp_pose = self._base_controller.get_grasp_pose()

        print("object pose: {}".format(world_obs))
        print("superq pose: {} {}".format(self._superqs[0].center, self._superqs[0].ea))
        print("grasp pose: {}".format(self._grasp_pose))

        if self._renders:
            self._base_controller._visualizer.render()

        return True

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

        # grasp pose
        gp = self._grasp_pose.copy()

        self._observation.extend(list(gp[:3]))
        observation_lim.extend(robot_obs_lim[:3])

        if self._control_eu_or_quat is 0:
            self._observation.extend(list(gp[3:6]))
            observation_lim.extend(robot_obs_lim[3:6])
        else:
            self._observation.extend(list(p.getQuaternionFromEuler(gp[3:6])))
            observation_lim.extend(robot_obs_lim[3:7])

        # object pose
        world_observation, world_obs_lim = self._world.get_observation()

        self._observation.extend(list(world_observation))
        observation_lim.extend(world_obs_lim)

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
            observation_lim.extend(world_obs_lim[:3])

            #
            if self._control_eu_or_quat is 0:
                self._observation.extend(list(sq_eu))
                observation_lim.extend(world_obs_lim[3:6])
            else:
                self._observation.extend(list(sq_quat))
                observation_lim.extend(world_obs_lim[3:7])

            #
            self._observation.extend([sq_dim[0][0], sq_dim[1][0], sq_dim[2][0],
                                      sq_exp[0][0], sq_exp[1][0]])

            # check dim limits of sq dim params
            observation_lim.extend([[0.0, 0.3], [0.0, 0.3], [0, 0.3], [-1, 1], [-1, 1]])

            # relative superq position wrt hand c.o.m. frame
            inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3], r_quat)
            sq_pos_in_hand, sq_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn,
                                                                  sq_pos, sq_quat)

            self._observation.extend(list(sq_pos_in_hand))
            observation_lim.extend([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

            if self._control_eu_or_quat is 0:
                sq_euler_in_hand = p.getEulerFromQuaternion(sq_orn_in_hand)
                self._observation.extend(list(sq_euler_in_hand))
                observation_lim.extend(robot_obs_lim[3:6])

            else:
                self._observation.extend(list(sq_orn_in_hand))
                observation_lim.extend(robot_obs_lim[3:7])

        else:
            # world_observation, world_obs_lim = self._world.get_observation()

            if self._control_eu_or_quat is 0:
                w_quat = p.getQuaternionFromEuler(world_observation[3:6])
            else:
                w_quat = world_observation[3:7]

            # self._observation.extend(list(world_observation))
            # observation_lim.extend(world_obs_lim)

            # relative object position wrt hand c.o.m. frame
            inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3], r_quat)
            obj_pos_in_hand, obj_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn,
                                                                    world_observation[:3], w_quat)

            self._observation.extend(list(obj_pos_in_hand))
            observation_lim.extend([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

            if self._control_eu_or_quat is 0:
                obj_euler_in_hand = p.getEulerFromQuaternion(obj_orn_in_hand)
                self._observation.extend(list(obj_euler_in_hand))
                observation_lim.extend(robot_obs_lim[3:6])

            else:
                self._observation.extend(list(obj_orn_in_hand))
                observation_lim.extend(robot_obs_lim[3:7])

        # add next step on the trajectory
        # if not self._base_controller._approach_path:  ## todo: CHAAAANGE IN A PROPER WAY
        #     next_step = self._grasp_pose.copy()
        #     next_step[2] += 0.2
        #
        #     self._observation.extend(list(next_step[:3]))
        #     observation_lim.extend(robot_obs_lim[:3])
        #
        #     if self._control_eu_or_quat is 0:
        #         self._observation.extend(list(next_step[3:6]))
        #         observation_lim.extend(robot_obs_lim[3:6])
        #     else:
        #         self._observation.extend(list(p.getQuaternionFromEuler(next_step[3:6])))
        #         observation_lim.extend(robot_obs_lim[3:7])
        #
        # else:
        #
        #     next_step = self._base_controller._approach_path[-1]
        #     self._observation.extend(list(next_step[0]))
        #     observation_lim.extend(robot_obs_lim[:3])
        #
        #     if self._control_eu_or_quat is 0:
        #         self._observation.extend(list(p.getEulerFromQuaternion(next_step[1])))
        #         observation_lim.extend(robot_obs_lim[3:6])
        #     else:
        #         self._observation.extend(list(next_step[1]))
        #         observation_lim.extend(robot_obs_lim[3:7])

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

        # ------------------------------ #
        # --- read corrective action --- #
        # ------------------------------ #

        pos_action = action[:3]

        if self._control_eu_or_quat is 0:
            quat_action = p.getQuaternionFromEuler(action[3:6])

        else:
            quat_action = action[3:7]
            # check if it is an invalid quaternion
            if quat_action[0] == 0 and quat_action[1] == 0 and quat_action[2] == 0:
                quat_action[3] = 1

        # --------------------------------------- #
        # --- get action from base controller --- #
        # --------------------------------------- #

        self.last_approach_step = False
        if self._base_controller.is_approach_path_empty():
            self.last_approach_step = True

        base_action, done = self._base_controller.get_next_action()

        # --------------------------- #
        # --- superimpose actions --- #
        # --------------------------- #

        final_action_pos = np.add(base_action[0], pos_action)
        final_action_quat = quat_multiplication(np.array(base_action[1]), np.array(quat_action))

        # ------------------------------------------ #
        # --- send actions to robot and simulate --- #
        # ------------------------------------------ #

        terminate = False
        it_step = 0
        while it_step < self._action_repeat and not terminate:

            it_step += 1

            self._robot.apply_action(final_action_pos.tolist() + final_action_quat.tolist())

            p.stepSimulation(physicsClientId=self._physics_client_id)
            if self._renders:
                time.sleep(self._time_step)

            self._env_step_counter += 1

            w_obs, _ = self._world.get_observation()
            # r_obs, _ = self._robot.get_observation()
            # self._cum_reward += self._compute_reward(w_obs, r_obs, action)

            if self._termination(w_obs):
                terminate = True

        # --- if it is the last step, try to grasp and lift the object --- #

        if self.last_approach_step and not terminate:

            # --> do grasp
            self._grasping_step = 5

            while self._grasping_step > 0:
                # move fingers
                action_f = [0.0, 0.0]
                self._robot.apply_action_fingers(action_f)

                p.stepSimulation(physicsClientId=self._physics_client_id)
                if self._renders:
                    time.sleep(self._time_step)

                self._grasping_step -= 1

            # --> do lift
            final_action_pos[2] += 0.2

            it_step = 0
            while it_step < self._action_repeat*3 and not terminate:

                it_step += 1

                self._robot.apply_action(final_action_pos.tolist() + final_action_quat.tolist(), max_vel=1)

                p.stepSimulation(physicsClientId=self._physics_client_id)
                if self._renders:
                    time.sleep(self._time_step)

                # w_obs, _ = self._world.get_observation()
                # r_obs, _ = self._robot.get_observation()
                # self._cum_reward += self._compute_reward(w_obs, r_obs, action)

                # if self._termination(w_obs):
                #     terminate = True

            self._env_step_counter = self._max_steps+1

        # dump data
        # if DO_LOGGING:
        #    self.dump_data([base_action, [final_action_pos.tolist() + final_action_quat.tolist()]])

        return final_action_pos.tolist() + final_action_quat.tolist()

    def step(self, action):

        # self._cum_reward = np.float32(0.0)

        # apply action on the robot
        applied_action = self.apply_action(action)

        w_obs, _ = self._world.get_observation()
        r_obs, _ = self._robot.get_observation()

        info = {
            'is_success': self._is_success(w_obs),
        }

        done = self._termination(w_obs)
        reward = self._compute_reward(w_obs, r_obs, action)

        obs, _ = self.get_extended_observation()

        if DO_LOGGING:
            self.dump_obs(r_obs[3:6], obs[-6:-3], obs[-3:])
            # self.dump_obs(obs[39:42], r_obs[6:9], r_obs[9:12]) #  DUMP SQ_DIM, VEL_L, VEL_A

        scaled_obs = obs
        if self._normalize_obs:
            scaled_obs = scale_gym_data(self.observation_space, obs)

        # print("reward")
        # print(self._cum_reward)

        return scaled_obs, np.array(reward), np.array(done), info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._world.seed(seed)
        self._robot.seed(seed)
        self._base_controller.seed(seed)
        return [seed]

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])

        base_pos, _ = self._p.getBasePositionAndOrientation(self._robot.robot_id, physicsClientId=self._physics_client_id)

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
                                                                upAxisIndex=2,
                                                                physicsClientId=self._physics_client_id)

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1, farVal=100.0,
                                                         physicsClientId=self._physics_client_id)

        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                                                  physicsClientId=self._physics_client_id)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self, w_obs):
        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
        else:
            eu = w_obs[3:6]

        # cost: object falls
        if self._object_fallen(eu[0], eu[1]):
            print("FALLEN")
            return np.float32(1.)

        # here check lift for termination
        if self._object_lifted(w_obs[2], self._target_h_lift) and self.last_approach_step:
            print("SUCCESS")
            return np.float32(1.)

        if self._env_step_counter > self._max_steps:
            print("MAX STEPS")
            return np.float32(1.)

        return np.float32(0.)

    def _is_success(self, w_obs):
        if self._object_lifted(w_obs[2], self._target_h_lift) and self.last_approach_step:
            print("SUCCESS")
            return np.float32(1.)
        else:
            return np.float32(0.)

    def _compute_reward(self, w_obs, r_obs, action):

        c1, c2, c3 = np.float32(0.0), np.float32(0.0), np.float32(0.0)
        r1, r2 = np.float32(0.0), np.float32(0.0)

        # cost 1: object touched
        if self._robot.check_collision(self._world.obj_id):
            # print("<<----------->> 1. object collision <<----------------->>")
            c1 = -np.float32(5)

        # cost 1.1: table touched
        if self._world.check_contact(self._robot.robot_id, self._world.table_id):
            # print("<<----------->> 2. table collision <<----------------->>")
            c1 -= np.float32(5)

        # cost 2: object falls
        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
        else:
            eu = w_obs[3:6]

        if self._object_fallen(eu[0], eu[1]):
            c2 = -np.float32(30.0)

        # cost 3: distance between hand and object
        if not self.last_approach_step:
            d = goal_distance(np.array(r_obs[:3]), np.array(w_obs[:3]))
            w_d = 1  # / self._action_repeat
            c3 = w_d * (-1 + self._compute_distance_reward(d, max_dist=self._distance_threshold))

            # cost 4: magnitude of action correction
            d_a = np.linalg.norm(action)
            w_d_a = 1  # / self._action_repeat
            c3 += w_d_a * (-1 + self._compute_distance_reward(d_a, max_dist=0.04))

        # reward 1: contact between object and fingertips
        r1 = np.float32(5 * self._robot.check_contact_fingertips(self._world.obj_id))
        # if r1 > 0:
        #     print("<<----------->> 3. contact fingertips <<----------------->>")

        # reward 2: when object lifted of target_h_object
        if self.last_approach_step:
            # d_lift = goal_distance(np.array([w_obs[2]]), np.array([self._target_h_lift]))
            # r2 = self._compute_distance_reward(d_lift, max_dist=0.13)

            if self._object_lifted(w_obs[2], self._target_h_lift):
                # self._t_lift += self._time_step
            # else:
                # self._t_lift = 0

            # if self._t_lift >= 0.3:
                r2 += np.float32(100.0)

        reward = r1 + r2 + (c1 + c2 + c3)

        return reward

    def _compute_distance_reward(self, dist, max_dist):

        w = - np.log(10e-5) / (max_dist ** 2)

        return np.exp(-w * (dist ** 2))


    def _object_fallen(self, obj_roll, obj_pitch):
        return obj_roll <= -0.785 or obj_roll >= 0.785 or obj_pitch <= -0.785 or obj_pitch >= 0.785

    def _object_lifted(self, z_obj, h_target, atol=0.1):
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

        p.addUserDebugLine(pose, pax.tolist(), [1, 0, 0], physicsClientId=self._physics_client_id)
        p.addUserDebugLine(pose, pay.tolist(), [0, 1, 0], physicsClientId=self._physics_client_id)
        p.addUserDebugLine(pose, paz.tolist(), [0, 0, 1], physicsClientId=self._physics_client_id)

        pose = self._grasp_pose[:3]

        matrix = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self._grasp_pose[3:6]))
        dcm = np.array([matrix[0:3], matrix[3:6], matrix[6:9]])
        np_pose = np.array(list(pose))
        pax = np_pose + np.array(list(dcm.dot([0.1, 0, 0])))
        pay = np_pose + np.array(list(dcm.dot([0, 0.1, 0])))
        paz = np_pose + np.array(list(dcm.dot([0, 0, 0.1])))

        p.addUserDebugLine(pose, pax.tolist(), [1, 0, 0], physicsClientId=self._physics_client_id)
        p.addUserDebugLine(pose, pay.tolist(), [0, 1, 0], physicsClientId=self._physics_client_id)
        p.addUserDebugLine(pose, paz.tolist(), [0, 0, 1], physicsClientId=self._physics_client_id)

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

    def dump_obs(self, sq_dim, vel_l, vel_a):
        assert 3 == len(self._log_file)

        for i in sq_dim:
            self._log_file[0].write(str(i))
            self._log_file[0].write(" ")
        self._log_file[0].write("\n")

        for j in vel_l:
            self._log_file[1].write(str(j))
            self._log_file[1].write(" ")
        self._log_file[1].write("\n")

        for k in vel_a:
            self._log_file[2].write(str(k))
            self._log_file[2].write(" ")
        self._log_file[2].write("\n")
