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
from pybullet_robot_envs.envs.utils import goal_distance, axis_angle_to_quaternion

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)


class iCubGraspResidualGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50}

    def __init__(self,
                 action_repeat=30,
                 control_arm='l',
                 control_orientation=1,
                 obj_name=get_ycb_objects_list()[0],
                 obj_pose_rnd_std=0.05,
                 noise_pcl=0.00,
                 renders=False,
                 max_steps=1000,
                 use_superq=1):

        self._time_step = 1. / 240.  # 4 ms

        self._control_arm = control_arm
        self._control_orientation = control_orientation
        self._action_repeat = action_repeat
        self._observation = []

        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self._t_grasp, self._t_lift = 0, 0
        self._obj_pose_rnd_std = obj_pose_rnd_std
        self. _noise_pcl = noise_pcl
        self._last_frame_time = 0
        self._use_superq = use_superq

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
        self._robot = iCubHandsEnv(use_IK=1, control_arm=self._control_arm, control_orientation=self._control_orientation)

        # Load world environment
        self._world = YcbWorldFetchEnv(obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                       workspace_lim=self._robot._workspace_lim)

        # Load base controller
        self._base_controller = SuperqGraspPlanner(self._robot.robot_id, self._world.obj_id, render=self._renders,
                                                   robot_base_pose=self._robot._home_hand_pose,
                                                   grasping_hand=self._control_arm,
                                                   noise_pcl=self._noise_pcl)

        # limit iCub workspace to table plane
        self._robot._workspace_lim[2][0] = self._world.get_table_height()

        self._superqs = []
        self._grasp_pose = []

        # initialize simulation environment
        self.seed()
        self.reset()

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
        action_bound = 0.005
        action_high = np.array([action_bound] * action_dim)
        action_space = spaces.Box(-action_high, action_high, dtype='float32')

        return observation_space, action_space

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        self._env_step_counter = 0

        p.setGravity(0, 0, -9.8)

        self._robot.reset()
        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation()

        self._world.reset()
        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation()

        self._robot.debug_gui()
        self._world.debug_gui()
        robot_obs, _ = self._robot.get_observation()
        self._base_controller.reset(robot_id=self._robot.robot_id, obj_id=self._world.obj_id,
                                    starting_pose=np.array(robot_obs))

        self._base_controller.set_robot_base_pose(p.getBasePositionAndOrientation(self._robot.robot_id))

        self.compute_grasp_pose()

        self.debug_gui()
        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation()

        if self._renders:
            self._base_controller._visualizer.render()

        self._t_grasp, self._t_lift = 0, 0

        obs, _ = self.get_extended_observation()
        return obs

    def compute_grasp_pose(self):

        self._base_controller.set_object_info(self._world.get_object_shape_info())

        # TO DO: add check on outputs!
        world_obs, _ = self._world.get_observation()
        ok = self._base_controller.compute_object_pointcloud(world_obs)
        if not ok:
            print("Can't get good point cloud of the object")
            return

        ok = self._base_controller.estimate_superq()
        if not ok:
            print("can't compute good superquadrics")
            return

        ok = self._base_controller.estimate_grasp()
        if not ok:
            print("can't compute any grasp pose")
            return

        self._base_controller.compute_approach_path()

        self._superqs = self._base_controller.get_superqs()
        self._grasp_pose = self._base_controller.get_grasp_pose()

        print("object pose: {}".format(world_obs))
        print("superq pose: {} {}".format(self._superqs[0].center, self._superqs[0].ea))
        print("grasp pose: {}".format(self._grasp_pose))

        if self._renders:
            self._base_controller._visualizer.render()

    def get_extended_observation(self):
        self._observation = []
        observation_lim = []

        # get observation form robot and world
        robot_observation, robot_obs_lim = self._robot.get_observation()
        world_observation, world_obs_lim = self._world.get_observation()

        self._observation.extend(list(robot_observation))
        observation_lim.extend(robot_obs_lim)

        # get superquadric params of dimension and shape
        if self._use_superq:
            # get superquadric params
            sq_pos = [self._superqs[0].center[0][0], self._superqs[0].center[1][0], self._superqs[0].center[2][0]]
            sq_eu = [self._superqs[0].ea[0][0], self._superqs[0].ea[1][0], self._superqs[0].ea[2][0]]
            sq_dim = self._superqs[0].dim
            sq_exp = self._superqs[0].exp

            self._observation.extend(list(sq_pos))
            self._observation.extend(list(sq_eu))
            self._observation.extend([sq_dim[0][0], sq_dim[1][0], sq_dim[2][0],
                                      sq_exp[0][0], sq_exp[1][0]])

            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])
            observation_lim.extend([[0, 2 * m.pi], [0, 2 * m.pi], [0, 2 * m.pi]])
            # check dim limits of sq dim params
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1], [0, 2], [0, 2]])

            # relative superq position wrt hand c.o.m. frame
            inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3],
                                                           p.getQuaternionFromEuler(robot_observation[3:6]))
            sq_pos_in_hand, sq_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn,
                                                                  sq_pos, p.getQuaternionFromEuler(sq_eu))
            sq_euler_in_hand = p.getEulerFromQuaternion(sq_orn_in_hand)

            self._observation.extend(list(sq_pos_in_hand))
            self._observation.extend(list(sq_euler_in_hand))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])
            observation_lim.extend([[0, 2 * m.pi], [0, 2 * m.pi], [0, 2 * m.pi]])

        else:
            self._observation.extend(list(world_observation))
            observation_lim.extend(world_obs_lim)

            # relative object position wrt hand c.o.m. frame
            inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3],
                                                           p.getQuaternionFromEuler(robot_observation[3:6]))
            obj_pos_in_hand, obj_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn,
                                                                    world_observation[:3],
                                                                    p.getQuaternionFromEuler(world_observation[3:6]))
            obj_euler_in_hand = p.getEulerFromQuaternion(obj_orn_in_hand)

            self._observation.extend(list(obj_pos_in_hand))
            self._observation.extend(list(obj_euler_in_hand))
            observation_lim.extend([[-1, 1], [-1, 1], [-1, 1]])
            observation_lim.extend([[0, 2 * m.pi], [0, 2 * m.pi], [0, 2 * m.pi]])

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
        action = 1.5*np.clip(action, self.action_space.low, self.action_space.high)
        action[:3] = np.clip(action[:3], -0.1, 0.1)
        pos_action = action[:3]
        eu_action = action[3:6]
        quat_action = p.getQuaternionFromEuler(eu_action)

        # get action from base controller
        robot_obs, _ = self._robot.get_observation()
        world_obs, _ = self._world.get_observation()

        base_action = self._base_controller.get_next_action(robot_obs, world_obs)

        final_action_pos = np.add(base_action[0], pos_action)
        final_action_quat = np.quaternion(base_action[1][3], base_action[1][0], base_action[1][1], base_action[1][2]) * \
                          np.quaternion(quat_action[3], quat_action[0], quat_action[1], quat_action[2])
        final_action_quat = quaternion.as_float_array(final_action_quat)
        final_action_quat_1 = [final_action_quat[1], final_action_quat[2], final_action_quat[3], final_action_quat[0]]

        #final_action = np.add(base_action[0].tolist() + base_action[1].tolist(), action[:6])

        self._robot.apply_action(final_action_pos.tolist() + final_action_quat_1)

        for _ in range(self._action_repeat):
            p.stepSimulation()
            if self._termination():
                break

            self._env_step_counter += 1

    def step(self, action):

        # apply action on the robot
        self.apply_action(action)

        obs, _ = self.get_extended_observation()

        done = self._termination()
        reward = self._compute_reward()

        # print("reward")
        # print(reward)

        return obs, np.array(reward), np.array(done), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
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

    def _termination(self):

        obs, _ = self._world.get_observation()

        if self._env_step_counter > self._max_steps:
            return np.float32(1.)

        # early termination if object falls
        if self._object_fallen(obs[3], obs[4]):
            print("FALLEN")
            return np.float32(1.)

        # here check lift for termination
        # if self._object_lifted(world_obs[2], world_obs[-1]) and self._t_lift >= 2:
        #    print("SUCCESS")
        #    return np.float32(1.)

        return np.float32(0.)

    def _compute_reward(self):
        c1, c2, r = np.float32(0.0), np.float32(0.0), np.float32(0.0)
        w_obs, _ = self._world.get_observation()

        # cost 1: trajectory as short as possible
        #if not self._robot.isGrasping():
        c1 = 1/2000 * self._env_step_counter

        # cost 2: object falls
        if self._object_fallen(w_obs[3], w_obs[4]):
            c2 = np.float32(10.0)

        #if self._robot.isGrasping() or self._robot.checkContactPalm():
        #    r += 1

        # reward: when object lifted of target_h_object for > 3 secs
        if self._object_lifted(w_obs[2], w_obs[-1]):
            self._t_lift += self._time_step*self._action_repeat
            r += 1
        else:
            self._t_lift = 0

        if self._t_lift >= 2:  # secs
            r += 100

        reward = r - (c1+c2)

        return reward

    def _object_fallen(self, obj_roll, obj_pitch):
        return obj_roll <= -1 or obj_roll >= 1 or obj_pitch <= -1 or obj_pitch >= 1

    def _object_lifted(self, z_obj, h_target, atol=0.05):
        return z_obj >= h_target - atol

    def debug_gui(self):

        pose = self._superqs[0].center
        p.addUserDebugLine([pose[0][0], pose[1][0], pose[2][0]], [pose[0][0]+0.1, pose[1][0], pose[2][0]], [1, 0, 0])
        p.addUserDebugLine([pose[0][0], pose[1][0], pose[2][0]], [pose[0][0], pose[1][0]+0.1, pose[2][0]], [0, 1, 0])
        p.addUserDebugLine([pose[0][0], pose[1][0], pose[2][0]], [pose[0][0], pose[1][0], pose[2][0]+0.1], [0, 0, 1])

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
