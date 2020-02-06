import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import math as m
from pybullet_robot_envs.envs.icub_envs.icub_env_with_hands import iCubHandsEnv
from pybullet_robot_envs.envs.icub_envs.icub_grasp_residual_gym_env import iCubGraspResidualGymEnv
from pybullet_robot_envs.envs.world_envs.ycb_fetch_env import get_ycb_objects_list, YcbWorldFetchEnv
from pybullet_robot_envs.envs.icub_envs.superq_grasp_planner import SuperqGraspPlanner

from pybullet_robot_envs.envs.utils import goal_distance

class iCubGraspResidualGymGoalEnv(gym.GoalEnv, iCubGraspResidualGymEnv):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 log_file=currentdir,
                 action_repeat=30,
                 control_arm='l',
                 control_orientation=1,
                 control_eu_or_quat=0,
                 obj_name=get_ycb_objects_list()[0],
                 obj_pose_rnd_std=0.0,
                 noise_pcl=0.00,
                 renders=False,
                 max_steps=2000, use_superq=1):

        self._time_step = 1. / 240.  # 4 ms

        self._control_arm = control_arm
        self._control_orientation = control_orientation
        self._control_eu_or_quat = control_eu_or_quat
        self._action_repeat = action_repeat
        self._observation = []
        self.goal = np.zeros(6)

        self._env_step_counter = 0
        self._renders = renders
        self._max_steps = max_steps
        self.terminated = 0
        self._t_grasp, self._t_grasp_target = 0, 2
        self._grasp_done = False
        self._obj_pose_rnd_std = obj_pose_rnd_std
        self._noise_pcl = noise_pcl
        self._last_frame_time = 0
        self._distance_threshold = 0.09
        self._use_superq = use_superq

        self._log_file = []
        self._log_file_path = []
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
        self._world = YcbWorldFetchEnv(obj_name=obj_name, obj_pose_rnd_std=obj_pose_rnd_std,
                                       workspace_lim=self._robot._workspace_lim,
                                       control_eu_or_quat=self._control_eu_or_quat)

        # Load base controller
        self._base_controller = SuperqGraspPlanner(self._robot.robot_id, self._world.obj_id, render=self._renders,
                                                   robot_base_pose=p.getBasePositionAndOrientation(self._robot.robot_id),
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
        goal_obs = self.get_goal_observation()

        observation_low = []
        observation_high = []
        for el in obs_lim:
            observation_low.extend([el[0]])
            observation_high.extend([el[1]])

        # Configure the observation space
        obs_space = spaces.Box(np.array(observation_low), np.array(observation_high), dtype='float32')

        # Configure the observation space
        observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=goal_obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=goal_obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(np.array(observation_low), np.array(observation_high), dtype='float32'),
        ))

        # Configure action space
        action_dim = self._robot.get_action_dim()
        action_bound = 1
        action_high = np.array([0.05, 0.05, 0.05, 0.785, 0.2, 1])
        action_low = np.array([-0.05, -0.05, -0.05, -0.785, -0.2, -1])
        action_space = spaces.Box(action_low, action_high, dtype='float32')

        return observation_space, action_space

    def reset(self):
        self.terminated = 0

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
        for _ in range(100):
            p.stepSimulation()

        self._robot.pre_grasp()

        self._world.reset()

        # Let the world run for a bit
        for _ in range(300):
            p.stepSimulation()

        self._robot.debug_gui()
        self._world.debug_gui()

        self._base_controller.reset(robot_id=self._robot.robot_id, obj_id=self._world.obj_id,
                                    starting_pose=self._robot._home_hand_pose)

        self._base_controller.set_robot_base_pose(p.getBasePositionAndOrientation(self._robot.robot_id))

        self.compute_grasp_pose()
        self._base_controller.compute_approach_path()

        self.debug_gui()
        p.stepSimulation()

        robot_obs, _ = self._robot.get_observation()
        world_obs, _ = self._world.get_observation()

        # move hand to the first way point on approach trajectory
        base_action = self._base_controller.get_next_action(robot_obs, world_obs)
        self._robot.apply_action(base_action[0].tolist() + base_action[1].tolist())

        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation()

        # compute goal
        self.goal = self._compute_goal()

        self._t_grasp = 0
        self._grasp_done = False

        obs = self.get_goal_observation()
        return obs

    def _compute_goal(self):
        # hand_pos, obj_pos, finger joints
        gp = self._grasp_pose.copy()
        world_obs, _ = self._world.get_observation()

        if self._control_eu_or_quat is 0:
            return np.array(gp + world_obs[:6] + self._robot._grasp_pos)
        else:
            return np.array(gp + world_obs[:7] + self._robot._grasp_pos)

    def get_goal_observation(self):
        obs, _ = self.get_extended_observation()
        w_obs, _ = self._world.get_observation()
        finger_joints = self._robot.get_finger_joints_poses()

        if self._control_eu_or_quat is 0:
            hand_pos = obs[:6]
            obj_pos = w_obs[:6]
        else:
            hand_pos = obs[:7]
            obj_pos = w_obs[:7]

        return {
            'observation': obs.copy(),
            'achieved_goal': np.concatenate((hand_pos, np.array(obj_pos), np.array(finger_joints))),
            'desired_goal': self.goal.copy(),
        }

    def step(self, action):
        # apply action on the robot
        applied_action = self.apply_action(action)

        # grasp object
        if not self._termination() and applied_action[2] < 0 and not self._grasp_done:
            steps = [i / 20 for i in range(0, 21, 1)]
            for i in steps:
                self._robot.grasp(i)
                for _ in range(5):
                    p.stepSimulation()
                    time.sleep(self._time_step)

                obs = self.get_goal_observation()
                if self._termination() or self._is_success(obs['achieved_goal'], self.goal):
                    break

            self._grasp_done = True
            # if self._robot.checkContacts(self._world.obj_id) and d <= 0.09:
            #    print("grasp success??")

        obs = self.get_goal_observation()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }

        done = self._termination() or info['is_success']

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        print("reward")
        print(reward)

        return obs, np.array(reward), np.array(done), info

    def _termination(self):
        if self._env_step_counter > self._max_steps:
            print("MAX STEPS")
            return np.float32(1.)

        # early termination if object falls
        w_obs, _ = self._world.get_observation()

        if self._control_eu_or_quat is 1:
            eu = p.getEulerFromQuaternion(w_obs[3:7])
        else:
            eu = w_obs[3:6]

        # cost: object falls
        if self._object_fallen(eu[0], eu[1]):
            print("FALLEN")
            return np.float32(1.)

        # here check lift for termination
        # if self._object_lifted(world_obs[2], world_obs[-1]) and self._t_lift >= 2:
        #    print("SUCCESS")
        #    return np.float32(1.)

        return np.float32(0.)

    def _is_success(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d_hands = goal_distance(achieved_goal[:3], goal[:3])
        d_objs = goal_distance(achieved_goal[6:9], goal[6:9])
        d_hand_obj = goal_distance(achieved_goal[:3], achieved_goal[6:9])
        d_fingers = goal_distance(achieved_goal[-20:], goal[-20:])

        ok = self._robot.checkContacts(self._world.obj_id) and \
                d_hand_obj <= self._distance_threshold and \
                d_hands <= self._distance_threshold and \
                d_objs <= 0.08 and d_fingers <= 0.7

        if ok:
            print("grasp success??")

        return ok

    def compute_reward(self, achieved_goal, goal, info):
        r = np.float32(-1.0)

        if self._control_eu_or_quat is 1:
            obj_eu = p.getEulerFromQuaternion(achieved_goal[9:13])
        else:
            obj_eu = achieved_goal[9:12]

        # cost: object falls
        if self._object_fallen(obj_eu[0], obj_eu[1]):
            return np.float32(-100.0)

        # Compute distance between goal and the achieved goal.
        d_hands = goal_distance(achieved_goal[:3], goal[:3])
        d_objs = goal_distance(achieved_goal[6:9], goal[6:9])
        d_hand_obj = goal_distance(achieved_goal[:3], achieved_goal[6:9])
        d_fingers = goal_distance(achieved_goal[-20:], goal[-20:])
        # cost: object-hand collision
        if d_hand_obj > self._distance_threshold and self._world.check_contact(self._robot.robot_id):
            return np.float32(-10.0)

        r = -np.float32(d_hand_obj > self._distance_threshold or
              d_hands > self._distance_threshold or
              d_objs > 0.08 or
              not self._robot.checkContacts(self._world.obj_id) or
              d_fingers > 0.7)

        return r
