import os, inspect
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import pickle
import math as m

from pybullet_object_models import superquadric_objects

from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv
from pybullet_robot_envs.envs.world_envs.world_env import get_ycb_objects_list, SqWorldEnv
from pybullet_robot_envs.envs.panda_envs.superq_grasp_planner import SuperqGraspPlanner
from pybullet_robot_envs.envs.utils import *

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)


def get_dataset_list(dset):
    try:
        f = open(os.path.join(superquadric_objects.getDataPath(), dset + '.pkl'), 'rb')
        itemlist = pickle.load(f)
        return itemlist

    except Exception:
        print("Cannot load dataset file {}".format(os.path.join(superquadric_objects.getDataPath(), dset + '.pkl')))


class PandaGraspResidualGymEnvSqObj(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 log_file=currentdir,
                 dset='train',
                 action_repeat=60,
                 control_orientation=1,
                 control_eu_or_quat=0,
                 normalize_obs=True,
                 obj_pose_rnd_std=0.0,
                 obj_orn_rnd=0.0,
                 noise_pcl=0.00,
                 renders=False,
                 max_steps=500,
                 use_superq=1,
                 n_control_pt=2):

        self._time_step = 1. / 240.  # 4 ms

        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._normalize_obs = normalize_obs
        self._use_superq = use_superq

        self._action_repeat = action_repeat
        self._n_control_pt = n_control_pt + 1
        self._observation = []

        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self._t_grasp, self._t_lift = 0, 0
        self._distance_threshold = 0.1
        self._target_h_lift = 1

        self._obj_pose_rnd_std = obj_pose_rnd_std
        self._obj_orn_rnd = obj_orn_rnd
        self._noise_pcl = noise_pcl

        self._last_frame_time = 0

        self._obj_list = get_dataset_list(dset)

        # self._cum_reward = np.float32(0.0)

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
            self._physics_client_id = p.connect(p.SHARED_MEMORY)
            if self._physics_client_id < 0:
                self._physics_client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.0, -0.0, -0.0], physicsClientId=self._physics_client_id)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        # this client is used only to compute the trajectory to the grasp pose and sample some way-points
        self._traj_client_id = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self._traj_client_id)

        # Load robot
        self._robot = pandaEnv(self._physics_client_id, use_IK=1, control_eu_or_quat=self._control_eu_or_quat)

        self._robot_traj = pandaEnv(self._traj_client_id, use_IK=1, control_eu_or_quat=self._control_eu_or_quat)

        # load world
        self._obj_iterator = 0
        self._obj_list = get_dataset_list(dset)
        self._obj_name = self._obj_list[self._obj_iterator]

        self._world = SqWorldEnv(self._physics_client_id,
                                       obj_name=self._obj_name,
                                        obj_pose_rnd_std=obj_pose_rnd_std, obj_orientation_rnd=self._obj_orn_rnd,
                                       workspace_lim=self._robot.get_workspace(),
                                       control_eu_or_quat=self._control_eu_or_quat)

        # Load base controller
        self._base_controller = SuperqGraspPlanner(self._physics_client_id,
                                                   self._traj_client_id,
                                                   self._robot_traj, self._world.obj_id, robot_name='panda',
                                                   render=self._renders,
                                                   grasping_hand='r',
                                                   noise_pcl=self._noise_pcl)

        # limit robot workspace to table plane
        workspace = self._robot.get_workspace()
        workspace[2][0] = self._world.get_table_height()
        self._robot.set_workspace(workspace)

        self._superqs = []
        self._grasp_pose = []
        self.are_gym_spaces_set = False

        # initialize simulation environment
        self.seed()
        self.reset()

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
        # --- reset simulation --- #
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._physics_client_id)
        p.setTimeStep(self._time_step, physicsClientId=self._physics_client_id)

        p.resetSimulation(physicsClientId=self._traj_client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._traj_client_id)
        p.setTimeStep(self._time_step, physicsClientId=self._traj_client_id)

        p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self._traj_client_id)

        self._env_step_counter = 0

        # --- reset robot --- #
        self._robot.reset()
        self._robot_traj.reset()

        # configure gripper in pre-grasp mode
        self._robot.pre_grasp()

        # --- reset world --- #
        # sample a object
        obj_name = self._obj_list[self._obj_iterator]
        self._obj_iterator += 1
        if self._obj_iterator == len(self._obj_list):
            self._obj_iterator = 0

        self._world._obj_name = obj_name
        print("it {} - obj_name {}".format(self._obj_iterator, obj_name))

        self._world.reset()

        # # Let the world run for a bit
        # for _ in range(600):
        #     p.stepSimulation(physicsClientId=self._physics_client_id)
        #     if self._renders:
        #         time.sleep(self._time_step)

        # --- reset base controller --- #
        self._base_controller.reset(obj_id=self._world.obj_id,
                                    starting_pose=self._robot._home_hand_pose,
                                    n_control_pt=self._n_control_pt)

        self._base_controller.set_robot_base_pose(p.getBasePositionAndOrientation(self._robot.robot_id,
                                                                                  physicsClientId=self._physics_client_id))

        # --- compute superquadric and grasp pose --- #
        ok = self.compute_grasp_pose()

        # move robot closer to the object, to reduce esploration space
        if ok:
            base_action, _ = self._base_controller.get_next_action()
            base_joint_conf = self._base_controller._approach_joint_path.pop()
            self._robot._use_simulation = False
            self._robot._use_IK = 0
            self._robot.apply_action(base_joint_conf[:self._robot.get_action_dim()])
            self._robot._use_simulation = True
            self._robot._use_IK = 1
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # --- draw some reference frames in the simulation for debugging --- #
        self._robot.debug_gui()
        self._world.debug_gui()
        self.debug_gui()
        p.stepSimulation(physicsClientId=self._physics_client_id)

        # ---  set target object height for a successful lift --- #
        world_obs, _ = self._world.get_observation()
        self._target_h_lift = world_obs[2] + 0.15

        self._t_grasp, self._t_lift = 0, 0
        self.last_approach_step = False

        # --- Define gym spaces if not done already --- #
        if not self.are_gym_spaces_set:
            self.observation_space, self.action_space = self.create_spaces()
            self.are_gym_spaces_set = True

        # --- Get observation and scale it --- #
        obs, _ = self.get_extended_observation()
        scaled_obs = obs
        if self._normalize_obs:
            scaled_obs = scale_gym_data(self.observation_space, obs)

        return scaled_obs

    def compute_grasp_pose(self):
        # USe the base controller to compute the grasp pose

        # Update object info
        self._base_controller.set_object_info(self._world.get_object_shape_info())

        world_obs, _ = self._world.get_observation()
        if self._control_eu_or_quat is 0:
            obj_pose = world_obs[:3] + list(p.getQuaternionFromEuler(world_obs[3:6]))
        else:
            obj_pose = world_obs[:3] + world_obs[3:7]

        # --- Compute the partial point cloud of the object --- #
        ok = self._base_controller.compute_object_pointcloud(obj_pose)
        if not ok:
            print("Can't get good point cloud of the object")
            self.reset()
            return False

        # --- Estimate the superquadric --- #
        ok = self._base_controller.estimate_superq()
        if not ok:
            print("can't compute good superquadrics")
            self.reset()
            return False

        # --- Compute the grasp pose --- #
        ok = self._base_controller.estimate_grasp()
        if not ok:
            print("can't compute any grasp pose")
            self.reset()
            return False

        # print computed quantities for debugging
        self._superqs = self._base_controller.get_superqs()
        self._grasp_pose = self._base_controller.get_grasp_pose()
        sq_ct, sq_ea, sq_dim, sq_exp = self._superqs[0].center, self._superqs[0].ea, self._superqs[0].dim, self._superqs[0].exp

        print("object pose: {},{}".format(np.round(world_obs[:3], 2), np.round(world_obs[3:6], 2)))
        print("superq pose: {}, {}".format(np.round([sq_ct[0][0], sq_ct[1][0], sq_ct[2][0]], 2),
                                           np.round([sq_ea[0][0], sq_ea[1][0], sq_ea[2][0]], 2)))
        print("superq parms: {}, {}".format(np.round([sq_dim[0][0], sq_dim[1][0], sq_dim[2][0]], 2),
                                            np.round([sq_exp[0][0], sq_exp[1][0]], 2)))
        print("grasp pose: {}, {}".format(np.round(self._grasp_pose[:3], 2), np.round(self._grasp_pose[3:6], 2)))

        # --- visualize superquadric and grasp pose in VTK --- #
        if self._renders:
            self._base_controller._visualizer.render()

        # --- Check if object shape/pose is valid --- #
        w_obs, _ = self._world.get_observation()
        w_ws = self._world.get_workspace()
        if w_obs[0] < w_ws[0][0] or w_obs[0] > w_ws[0][1] or w_obs[1] < w_ws[1][0] or w_obs[1] > w_ws[1][1]:
            print("object has fallen out of workspace")
            self.reset()
            return False
        print("visual dim {}".format(self._world.get_object_shape_info()[3]))
        if sq_dim[0] > 0.05 and sq_dim[1] > 0.05:
            print("the object shape is ungraspable with panda")
            self.reset()
            return False

        # --- Compute the trajectory to the grasp pose --- #
        # first move the robot to a top-table configuration, to have better trajectory
        self._robot._use_simulation = False
        self._robot_traj._use_simulation = False
        pose = (0.3, 0.0, 1.2, m.pi, -m.pi/4, 0)
        self._robot.apply_action(pose)
        self._robot_traj.apply_action(pose)
        self._robot._use_simulation = True
        self._robot_traj._use_simulation = True

        ok = self._base_controller.compute_approach_path()
        if not ok:
            print("can't compute the approach path")
            self.reset()
            return False

        return True

    def get_extended_observation(self):
        self._observation = []
        observation_lim = []

        # ------------------------- #
        # --- Robot observation --- #
        # ------------------------- #
        robot_observation, robot_obs_lim = self._robot.get_observation()

        self._observation.extend(list(robot_observation))
        observation_lim.extend(robot_obs_lim)

        # get quaternion of robot end-effector rotation
        if self._control_eu_or_quat is 0:
            r_quat = p.getQuaternionFromEuler(robot_observation[3:6])
        else:
            r_quat = robot_observation[3:7]

        # ------------------ #
        # --- Grasp pose --- #
        # ------------------ #
        gp = self._grasp_pose.copy()

        self._observation.extend(list(gp[:3]))
        observation_lim.extend(robot_obs_lim[:3])

        if self._control_eu_or_quat is 0:
            self._observation.extend(list(gp[3:6]))
            observation_lim.extend(robot_obs_lim[3:6])
        else:
            self._observation.extend(list(p.getQuaternionFromEuler(gp[3:6])))
            observation_lim.extend(robot_obs_lim[3:7])

        # ------------------- #
        # --- Object pose --- #
        # ------------------- #
        world_observation, world_obs_lim = self._world.get_observation()

        self._observation.extend(list(world_observation))
        observation_lim.extend(world_obs_lim)

        if self._use_superq:
            # -------------------------------- #
            # --- Superquadric related obs --- #
            # -------------------------------- #

            # get superquadric params
            sq_pos = [self._superqs[0].center[0][0], self._superqs[0].center[1][0], self._superqs[0].center[2][0]]
            sq_quat = axis_angle_to_quaternion((self._superqs[0].axisangle[0][0], self._superqs[0].axisangle[1][0],
                                                self._superqs[0].axisangle[2][0], self._superqs[0].axisangle[3][0]))
            sq_eu = p.getEulerFromQuaternion(sq_quat)
            sq_dim = self._superqs[0].dim
            sq_exp = self._superqs[0].exp

            # --- superquadric pose --- #
            self._observation.extend(list(sq_pos))
            observation_lim.extend(world_obs_lim[:3])

            if self._control_eu_or_quat is 0:
                self._observation.extend(list(sq_eu))
                observation_lim.extend(world_obs_lim[3:6])
            else:
                self._observation.extend(list(sq_quat))
                observation_lim.extend(world_obs_lim[3:7])

            # --- superquadric shape params --- #
            self._observation.extend([sq_dim[0][0], sq_dim[1][0], sq_dim[2][0],
                                      sq_exp[0][0], sq_exp[1][0]])

            # check dim limits of sq dim params
            observation_lim.extend([[0.0, 0.12], [0.0, 0.12], [0, 0.12], [0, 2], [0, 2]])

            # --- superq position wrt hand c.o.m. frame --- #
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
            # -------------------------------- #
            # --- Other object related obs --- #
            # -------------------------------- #
            if self._control_eu_or_quat is 0:
                w_quat = p.getQuaternionFromEuler(world_observation[3:6])
            else:
                w_quat = world_observation[3:7]

            # --- object position wrt hand c.o.m. frame --- #
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
        # --- read action from agent --- #
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
            self._robot.pre_grasp()
            p.stepSimulation(physicsClientId=self._physics_client_id)
            if self._renders:
                time.sleep(self._time_step)

            w_obs, _ = self._world.get_observation()
            # r_obs, _ = self._robot.get_observation()
            # self._cum_reward += self._compute_reward(w_obs, r_obs, action)

            if self._termination(w_obs):
                terminate = True

        # --- if it is the last step, try to grasp and lift the object --- #

        if self.last_approach_step and not terminate:

            # --> do grasp
            grasping_step = 10
            while grasping_step > 0:
                self._robot.grasp(self._world.obj_id)
                # move fingers
                p.stepSimulation(physicsClientId=self._physics_client_id)
                if self._renders:
                    time.sleep(self._time_step)

                grasping_step -= 1

            # --> do lift
            final_action_pos[2] += 0.2

            it_step = 0
            while it_step < self._action_repeat*3 and not terminate:

                it_step += 1

                self._robot.apply_action(final_action_pos.tolist() + final_action_quat.tolist(), max_vel=1)

                p.stepSimulation(physicsClientId=self._physics_client_id)
                if self._renders:
                    time.sleep(self._time_step)

                w_obs, _ = self._world.get_observation()
                # r_obs, _ = self._robot.get_observation()
                # self._cum_reward += self._compute_reward(w_obs, r_obs, action)

                if self._termination(w_obs):
                    terminate = True

            # set step counter to max in order to force end of episode
            self._env_step_counter = self._max_steps+1

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

        scaled_obs = obs.copy()
        if self._normalize_obs:
            scaled_obs = scale_gym_data(self.observation_space, obs)

        # print("reward")
        # print(self._cum_reward)

        return scaled_obs, np.array(reward), np.array(done), info

    def seed(self, seed=None):
        # seed everything for reproducibility
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
        # -------------------- #
        # --- object fall? --- #
        # -------------------- #
        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
            quat = w_obs[3:7]
        else:
            eu = w_obs[3:6]
            quat = p.getQuaternionFromEuler(w_obs[3:6])

        init_obj_pos, init_obj_quat = self._world.get_object_init_pose()
        rel_quat = quat_multiplication(quat_inverse(np.array(quat)), np.array(init_obj_quat))

        rel_eu = p.getEulerFromQuaternion(rel_quat)

        if self._object_fallen(rel_eu[0], rel_eu[1]):
            print("FALLEN")
            return np.float32(1.)

        # ---------------------- #
        # --- object lifted? --- #
        # ---------------------- #
        if self._object_lifted(w_obs[2], self._target_h_lift) and self.last_approach_step:
            print("SUCCESS")
            return np.float32(1.)

        # -------------------------------------- #
        # --- max num of iterations reached? --- #
        # -------------------------------------- #
        if self._env_step_counter > self._max_steps:
            print("MAX STEPS")
            return np.float32(1.)

        return np.float32(0.)

    def _is_success(self, w_obs):
        # ---------------------- #
        # --- object lifted? --- #
        # ---------------------- #
        if self._object_lifted(w_obs[2], self._target_h_lift) and self.last_approach_step:
            return np.float32(1.)
        else:
            return np.float32(0.)

    def _compute_reward(self, w_obs, r_obs, action):

        c1, c2, c3 = np.float32(0.0), np.float32(0.0), np.float32(0.0)
        r1, r2 = np.float32(0.0), np.float32(0.0)

        # ------------------------------ #
        # --- cost 1: object touched --- #
        # ------------------------------ #
        if self._robot.check_collision(self._world.obj_id):
            # print("<<----------->> 1. object collision <<----------------->>")
            c1 = -np.float32(5)

        # ------------------------------- #
        # --- cost 1.1: table touched --- #
        # ------------------------------- #
        if self._world.check_contact(self._robot.robot_id, self._world.table_id):
            # print("<<----------->> 2. table collision <<----------------->>")
            c1 -= np.float32(10)

        # --------------------------- #
        # --- cost 2: object fall --- #
        # --------------------------- #
        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
            quat = w_obs[3:7]
        else:
            eu = w_obs[3:6]
            quat = p.getQuaternionFromEuler(w_obs[3:6])

        init_obj_pos, init_obj_quat = self._world.get_object_init_pose()
        rel_quat = quat_multiplication(quat_inverse(np.array(quat)), np.array(init_obj_quat))

        rel_eu = p.getEulerFromQuaternion(rel_quat)

        if self._object_fallen(rel_eu[0], rel_eu[1]):
            c2 = -np.float32(30.0)

        if not self.last_approach_step:
            # ------------------------------------------------ #
            # --- cost 3: distance between hand and object --- #
            # ------------------------------------------------ #
            d = goal_distance(np.array(r_obs[:3]), np.array(w_obs[:3]))
            w_d = 1  # / self._action_repeat
            c3 = w_d * (-1 + self._compute_distance_reward(d, max_dist=self._distance_threshold))

            # ---------------------------------------------- #
            # --- cost 4: magnitude of action correction --- #
            # ---------------------------------------------- #
            d_a = np.linalg.norm(action)
            w_d_a = 1  # / self._action_repeat
            c3 += w_d_a * (-1 + self._compute_distance_reward(d_a, max_dist=0.04))

        # ------------------------------------------------------- #
        # --- reward 1: contact between object and fingertips --- #
        # ------------------------------------------------------- #
        w_r1 = 5
        ct, _ = self._robot.check_contact_fingertips(self._world.obj_id)

        r1 = np.float32(w_r1 * ct)

        # ------------------------------------------------------- #
        # --- reward 2: when object lifted of target_h_object --- #
        # ------------------------------------------------------- #
        if self.last_approach_step:
            r_sq = np.float32(0.0)

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
