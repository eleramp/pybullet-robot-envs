import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from icub_env import iCubEnv
from world_fetch_env import WorldFetchEnv
from superq_grasp_planner import SuperqGraspPlanner
import pybullet_data
import robot_data
import pybullet_robot_envs.envs.utils
from pybullet_robot_envs.envs.utils import goal_distance, axis_angle_to_quaternion, quaternion_to_axis_angle, sph_coord

RENDER_HEIGHT = 240
RENDER_WIDTH = 320


class iCubGraspResidualGymGoalEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=robot_data.getDataPath(),
                 actionRepeat=30,
                 control_arm='l',
                 useOrientation=0,
                 rnd_obj_pose=0.05,
                 noise_pcl=0.00,
                 renders=False,
                 maxSteps = 2000,
                 terminal_failure = True):

        self._control_arm = control_arm
        self._time_step = 1./240.  # 4 ms
        self._useOrientation = useOrientation
        self._urdfRoot = urdfRoot
        self._action_repeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = -90
        self._cam_pitch = -40
        self._t_grasp, self._t_lift = 0, 0
        self._rnd_obj_pose = rnd_obj_pose
        self. _noise_pcl = noise_pcl
        self._last_frame_time = 0
        self._terminal_failure = terminal_failure
        self.distance_threshold = 0.05
        self._obj0_T_sq =[]

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
        self._robot = iCubEnv(urdfRootPath=self._urdfRoot, useInverseKinematics=1,
                              arm=self._control_arm, useOrientation=self._useOrientation)

        # Load world environment
        self._world = WorldFetchEnv(rnd_obj_pose=self._rnd_obj_pose, workspace_lim=self._robot.workspace_lim)

        # Load base controller
        self._base_controller = SuperqGraspPlanner(self._robot.icubId, self._world.obj_id, render = self._renders,
                                                   robot_base_pose=p.getBasePositionAndOrientation(self._robot.icubId),
                                                   grasping_hand=self._control_arm,
                                                   noise_pcl=self._noise_pcl)
        self._superqs = []
        self._grasp_pose = []

        # initialize simulation environment
        self.seed()
        obs = self.reset()

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        action_dim = self._robot.getActionDimension()
        action_bound = 1.
        self.action_space = spaces.Box(-action_bound, action_bound, shape=(action_dim,), dtype='float32')

        self.viewer = None

    def reset(self):

        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        self._envStepCounter = 0

        self._robot.reset()
        self._world.reset()

        # limit iCub workspace to table plane
        self._robot.workspace_lim[2][0] = self._world.get_table_height()

        p.setGravity(0, 0, -9.8)

        # Let the world run for a bit
        for _ in range(500):
            p.stepSimulation()

        self._robot.debug_gui()
        self._world.debug_gui()



        self._base_controller.reset(robot_id=self._robot.icubId, obj_id=self._world.obj_id,
                                    starting_pose=np.array(self._robot.getObservation()))

        self.compute_grasp_pose()
        self.debug_gui()

        p.stepSimulation()

        self.goal = self._compute_goal()

        self._t_grasp, self._t_lift = 0, 0

        return self.get_extended_observation()

    def _compute_goal(self):
        sq_pos = self._superqs[0].center.copy()
        sq_eu = self._superqs[0].ea.copy()

        gp = self._grasp_pose.copy()

        # relative sq position wrt grasping pose
        inv_gp_pose = p.invertTransform(gp[:3], p.getQuaternionFromEuler(gp[3:6]))
        sq_pos_in_gp, sq_orn_in_gp = p.multiplyTransforms(inv_gp_pose[0], inv_gp_pose[1],
                                                          sq_pos, p.getQuaternionFromEuler(sq_eu))
        sq_eu_in_gp = p.getEulerFromQuaternion(sq_orn_in_gp)

        return np.array(sq_pos_in_gp + sq_eu_in_gp)

    def compute_grasp_pose(self):

        self._base_controller.set_object_info(self._world.get_object_shape_info())

        # TO DO: add check on outputs!
        ok = self._base_controller.compute_object_pointcloud(self._world.get_observation())
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

        print("object pose: {}".format(self._world.get_observation()))
        print("superq pose: {} {}".format(self._superqs[0].center, self._superqs[0].ea))
        print("grasp pose: {}".format(self._grasp_pose))

        if self._renders:
            self._base_controller._visualizer.render()

        return self._grasp_pose

    def get_extended_observation(self):
        # get observation from robot and world
        robot_observation = self._robot.getObservation()
        world_observation = self._world.get_observation()

        # get superquadric params of dimension and shape
        sq_dim = self._superqs[0].dim
        sq_exp = self._superqs[0].exp
        sq_arr = np.array([sq_dim[0][0], sq_dim[1][0], sq_dim[2][0], sq_exp[0][0], sq_exp[1][0]])

        sq_pos = [self._superqs[0].center[0][0], self._superqs[0].center[1][0], self._superqs[0].center[2][0]]
        sq_eu = [self._superqs[0].ea[0][0], self._superqs[0].ea[1][0], self._superqs[0].ea[2][0]]

        # relative object pose wrt superq pose
        inv_sq_pos, inv_sq_orn = p.invertTransform(sq_pos, p.getQuaternionFromEuler(sq_eu))
        obj_pos_in_sq, obj_orn_in_sq = p.multiplyTransforms(inv_sq_pos, inv_sq_orn,
                                                            world_observation[:3], p.getQuaternionFromEuler(world_observation[3:6]))
        obj_eu_in_sq = p.getEulerFromQuaternion(obj_orn_in_sq)

        # relative superq position wrt hand c.o.m. frame
        inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3], p.getQuaternionFromEuler(robot_observation[3:6]))
        sq_pos_in_hand, sq_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn,
                                                              sq_pos, p.getQuaternionFromEuler(sq_eu))
        sq_euler_in_hand = p.getEulerFromQuaternion(sq_orn_in_hand)

        observation = np.concatenate([robot_observation, sq_pos, sq_eu, sq_arr, obj_pos_in_sq, obj_eu_in_sq])

        return {
            'observation': observation.copy(),
            'achieved_goal': np.array(sq_pos_in_hand + sq_euler_in_hand),
            'desired_goal': self.goal.copy(),
        }

    def step(self, action):
        # scale action
        real_pos = [a*0.1 for a in action[:3]]
        real_orn = []
        if self.action_space.shape[-1] >= 6:
            real_orn = [a*0.8 for a in action[3:6]]
        if self.action_space.shape[-1] == 7:
            fingers = [action[-1]]  # +1 open, -1 close, 0 nothing
        sc_action = [real_pos, real_orn, fingers]
        #np.clip(sc_action, self.action_space.low, self.action_space.high)
        return self.step2(sc_action)

    def step2(self, action):

        if self._renders:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._action_repeat * self._time_step - time_spent

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        # get action from base controller
        robot_observation = self._robot.getObservation()
        world_observation = self._world.get_observation()
        base_action = self._base_controller.get_next_action(robot_observation,
                                                            world_observation)

        final_action = np.add(base_action, action)
        # final_action[0] = np.clip(final_action[0], self.action_space.low, self.action_space.high)
        # final_action[1] = np.clip(final_action[1], self.action_space.low, self.action_space.high)
        # final_action[2] = np.clip(final_action[2], self.action_space.low, self.action_space.high)

        self._robot.applyAction(final_action[0].tolist() + final_action[1].tolist())

        # grasp object
        #if final_action[2] < 0:
        #    if not self._robot.isGrasping() and self._robot.checkContactPalm():
        #        self._t_grasp += self._time_step * self._action_repeat
        #    else:
        #        self._t_grasp = 0

        #   if self._t_grasp >= 0.5:
        #        self._robot.graspObject(self._world.obj_id, world_observation)

        #elif final_action[2] > 0:
        #    self._robot.releaseObject()

        for _ in range(self._action_repeat):
            p.stepSimulation()
            if self._termination():
                break

            self._envStepCounter += 1

        obs = self.get_extended_observation()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }

        done = self._termination() or info['is_success']

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        #print("reward")
        #print(reward)

        return obs, np.array(reward), np.array(done), info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
          return np.array([])

        base_pos, _ = self._p.getBasePositionAndOrientation(self._robot.icubId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
            #renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def __del__(self):
        self._p.disconnect(self._cid)

    def _termination(self):

        obs = self._world.get_observation()

        if self._envStepCounter > self._maxSteps:
            return np.float32(1.)

        # early termination if object falls
        if self._terminal_failure and self._object_fallen(obs[3], obs[4]):
            print("FALLEN")
            return np.float32(1.)

        return np.float32(0.)

    def _is_success(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal[:3], goal[:3])
        if self._robot.checkContactPalm() and d < self.distance_threshold:
            return np.float32(1.0)
        else:
            return np.float32(0.0)

    def compute_reward(self, achieved_goal, goal, info):
        r = np.float32(-1.0)
        w_obs = self._world.get_observation()

        # cost 2: object falls
        if self._object_fallen(w_obs[3], w_obs[4]):
            return np.float32(-1.0)

        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal[:3], goal[:3])
        if self._robot.checkContactPalm() and d < self.distance_threshold:
            r = np.float32(0.0)
        return r

    def _object_fallen(self, obj_roll, obj_pitch):
        return obj_roll <= -0.5 or obj_roll >= 0.5 or obj_pitch <= -0.5 or obj_pitch >= 0.5

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
