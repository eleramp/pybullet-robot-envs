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


class iCubReachGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50}

    def __init__(self,
                 log_file=os.path.join(currentdir),
                 action_repeat=5,
                 control_arm='r',
                 control_orientation=1,
                 control_eu_or_quat=0,
                 obj_name=get_ycb_objects_list()[0],
                 obj_pose_rnd_std=0.05,
                 renders=False,
                 max_steps=5000,
                 use_superq=1):

        self._time_step = 1. / 240.  # 4 ms

        self._control_arm = control_arm
        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._action_repeat = action_repeat
        self._observation = []
        self._grasp_steps = [i / 20 for i in range(0, 21, 1)]
        self._grasp_idx = 0

        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self._t_grasp = 0
        self._obj_pose_rnd_std = obj_pose_rnd_std
        self._last_frame_time = 0
        self._use_superq = use_superq
        self._distance_threshold = 0.09

        self._log_file = []
        self._log_file_path = []
        self._log_file_path.append(os.path.join(log_file, 'learned.txt'))
        self._log_file.append(open(self._log_file_path[0], "w+"))

        self._log_file[0].close()

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
            self._physics_client_id = p.connect(p.SHARED_MEMORY)
            if self._physics_client_id < 0:
                self._physics_client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.0, -0.0, -0.0], physicsClientId=self._physics_client_id)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        # Load robot
        self._robot = iCubHandsEnv(self._physics_client_id,
                                   use_IK=1, control_arm=self._control_arm,
                                   control_orientation=self._control_orientation,
                                   control_eu_or_quat=self._control_eu_or_quat)

        # Load world environment
        self._world = YcbWorldFetchEnv(self._physics_client_id,
                                       obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                       workspace_lim=self._robot._workspace_lim,
                                       control_eu_or_quat=self._control_eu_or_quat)

        # limit iCub workspace to table plane
        self._robot._workspace_lim[2][0] = self._world.get_table_height()

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
        action_bound = 1
        if self._control_eu_or_quat is 0:
            action_high = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
            action_low = np.array([-0.01, -0.01, -0.01, -0.1, -0.1, -0.1])
        else:
            action_high = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1,  0.1])
            action_low = np.array([-0.01, -0.01, -0.01, -0.1, -0.1, -0.1, -0.1])

        action_space = spaces.Box(action_low, action_high, dtype='float32')

        return observation_space, action_space

    def reset(self):
        if not self._log_file[0].closed:
            self._log_file[0].close()
        self._log_file[0] = open(self._log_file_path[0], "a+")

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self._physics_client_id)
        p.setTimeStep(self._time_step, physicsClientId=self._physics_client_id)
        self._env_step_counter = 0

        p.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)

        self._robot.reset()

        # Let the world run for a bit
        for _ in range(50):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        self._robot.pre_grasp()

        obj_name = get_ycb_objects_list()[self.np_random.randint(0,3)]
        self._world._obj_name = obj_name
        print("obj_name {}".format(obj_name))
        
        self._world.reset()

        # Let the world run for a bit
        for _ in range(150):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        self._robot.debug_gui()
        self._world.debug_gui()
        robot_obs, _ = self._robot.get_observation()

        self.debug_gui()
        p.stepSimulation(physicsClientId=self._physics_client_id)

        robot_obs, _ = self._robot.get_observation()
        world_obs, _ = self._world.get_observation()

        self._t_grasp, self._t_lift = 0, 0
        self._grasp_idx = 0

        if self._use_IK:
            self._hand_pose = self._robot._home_hand_pose

        obs, _ = self.get_extended_observation()
        return obs

    def get_extended_observation(self):
        self._observation = []
        observation_lim = []

        # get observation form robot and world
        robot_observation, robot_obs_lim = self._robot.get_observation()
        world_observation, world_obs_lim = self._world.get_observation()

        if self._control_eu_or_quat is 0:
            r_quat = p.getQuaternionFromEuler(robot_observation[3:6])
            w_quat = p.getQuaternionFromEuler(world_observation[3:6])
        else:
            r_quat = robot_observation[3:7]
            w_quat = world_observation[3:7]

        self._observation.extend(list(robot_observation))
        observation_lim.extend(robot_obs_lim)

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
            observation_lim.extend([[-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi], [-2*m.pi, 2*m.pi]])

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

        # position
        pos_action = action[:3]

        # orientation
        if self._control_eu_or_quat is 0:
            quat_action = p.getQuaternionFromEuler(action[3:6])
        else:
            quat_action = action[3:7]
            if quat_action[0] == 0 and quat_action[1] == 0 and quat_action[2] == 0:
                quat_action[3] = 1

        # sum commanded pos/orn increment to current pose
        new_action_pos = np.add(self._hand_pose[:3], pos_action)

        new_action_quat = np.quaternion(self._hand_pose[6], self._hand_pose[3], self._hand_pose[4], self._hand_pose[5]) * \
                          np.quaternion(quat_action[3], quat_action[0], quat_action[1], quat_action[2])

        new_action_quat = quaternion.as_float_array(new_action_quat)
        new_action_quat_1 = [new_action_quat[1], new_action_quat[2], new_action_quat[3], new_action_quat[0]]

        self._hand_pose = new_action_pos.tolist() + new_action_quat_1

        self._robot.apply_action(new_action_pos.tolist() + new_action_quat_1)
        for _ in range(self._action_repeat):
            p.stepSimulation(physicsClientId=self._physics_client_id)
            time.sleep(self._time_step)
            if self._termination():
                break

            self._env_step_counter += 1

        # dump data
        self.dump_data(self._hand_pose)

    def step(self, action):
        # apply action on the robot
        self.apply_action(action)

        obs, _ = self.get_extended_observation()

        done = self._termination()
        reward = self._compute_reward()

        print("reward")
        print(reward)

        return obs, np.array(reward), np.array(done), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
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

    def _termination(self):
        if self._env_step_counter > self._max_steps:
            print("MAX STEPS")
            return np.float32(1.)

        # early termination if object falls
        w_obs, _ = self._world.get_observation()

        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7], physicsClientId=self._physics_client_id)
        else:
            eu = w_obs[3:6]

        # cost: object falls
        if self._object_fallen(eu[0], eu[1]):
            print("FALLEN")
            return np.float32(1.)

        # rew 1: distance between hand and grasp pose
        r_obs, _ = self._robot.get_observation()
        # Compute distance between goal and the achieved goal.
        d = goal_distance(np.array(r_obs[:3]), np.array(w_obs[:3]))
        if d <= self._distance_threshold and self._t_grasp >= 1:
            print("SUCCESS")
            return np.float32(1.)

        return np.float32(0.)

    def _compute_reward(self):
        c1, c2, r = np.float32(0.0), np.float32(0.0), np.float32(0.0)

        # cost 1: object touched
        if self._world.check_contact(self._robot.robot_id):
            c1 = np.float32(1.0)

        # cost 2: object falls
        w_obs, _ = self._world.get_observation()

        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
        else:
            eu = w_obs[3:6]

        if self._object_fallen(eu[0], eu[1]):
            c2 = np.float32(10.0)

        # rew 1: distance between hand and grasp pose
        r_obs, _ = self._robot.get_observation()
        # Compute distance between hand and obj.
        d = goal_distance(np.array(r_obs[:3]), np.array(w_obs[:3]))
        if d <= self._distance_threshold:
            r += np.float32(10.0)
            self._t_grasp += self._time_step*self._action_repeat
        else:
            self._t_grasp = 0

        if d <= self._distance_threshold and self._t_grasp >= 1:
            r = np.float32(100.0)

        reward = r - (c1+c2)

        return reward

    def _object_fallen(self, obj_roll, obj_pitch):
        return obj_roll <= -0.785 or obj_roll >= 0.785 or obj_pitch <= -0.785 or obj_pitch >= 0.785

    def _object_lifted(self, z_obj, h_target, atol=0.05):
        return z_obj >= h_target - atol

    def debug_gui(self):
        pass

    def dump_data(self, data):
        for ii in data:
            self._log_file[0].write(str(ii))
            self._log_file[0].write(" ")
        self._log_file[0].write("\n")
