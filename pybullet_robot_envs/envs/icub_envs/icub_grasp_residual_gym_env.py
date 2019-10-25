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

largeValObservation = 100

RENDER_HEIGHT = 240
RENDER_WIDTH = 320


class iCubGraspResidualGymEnv(gym.Env):
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
                 maxSteps = 3000,
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
        self.reset()

        observationDim = len(self.get_extended_observation())
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype='float32')

        action_dim = self._robot.getActionDimension()
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

        self.viewer = None

    def reset(self):

        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)
        self._envStepCounter = 0

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

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

        self._base_controller.set_object_info(self._world.get_object_shape_info())

        # TO DO: add check on outputs!
        if self._base_controller.compute_object_pointcloud(self._world.get_observation()):
            self._superqs = self._base_controller.estimate_superq()

        else:
            print("Can't get good point cloud of the object")

        self._grasp_pose = self._base_controller.estimate_grasp()
        self._base_controller.compute_approach_path()

        print("object pose: {}".format(self._world.get_observation()))
        print("superq pose: {} {}".format(self._superqs[0].center, self._superqs[0].ea))
        print("grasp pose: {}".format(self._grasp_pose))

        self.get_extended_observation()

        self.debug_gui()
        p.stepSimulation()

        if self._renders:
            self._base_controller._visualizer.render()

        self._t_grasp, self._t_lift = 0, 0

        return np.array(self._observation)

    def get_extended_observation(self):
        self._observation = []

        # get observation form robot and world
        robot_observation = self._robot.getObservation()
        world_observation = self._world.get_observation()

        # relative object position wrt hand c.o.m. frame
        inv_hand_pos, inv_hand_orn = p.invertTransform(robot_observation[:3], p.getQuaternionFromEuler(robot_observation[3:6]))
        obj_pos_in_hand, obj_orn_in_hand = p.multiplyTransforms(inv_hand_pos, inv_hand_orn, world_observation[:3],
                                                                p.getQuaternionFromEuler(world_observation[3:6]))
        obj_euler_in_hand = p.getEulerFromQuaternion(obj_orn_in_hand)

        # get superquadric params of dimension and shape
        sq_dim = self._superqs[0].dim
        sq_exp = self._superqs[0].exp

        self._observation.extend(list(robot_observation))
        self._observation.extend(list(world_observation))
        self._observation.extend(list(obj_pos_in_hand))
        self._observation.extend(list(obj_euler_in_hand))
        self._observation.extend([sq_dim[0][0], sq_dim[1][0], sq_dim[2][0], sq_exp[0][0], sq_exp[1][0]])

        target_h_obj = self._world.get_table_height() + 0.2
        self._observation.extend([target_h_obj])

        return np.array(self._observation)

    def step(self, action):
        # scale action
        real_pos = [a*0.05 for a in action[:3]]
        real_orn = []
        if self.action_space.shape[-1] >= 6:
            real_orn = [a*0.08 for a in action[3:6]]
        if self.action_space.shape[-1] == 7:
            fingers = [action[-1]]  # +1 open, -1 close, 0 nothing

        return self.step2([real_pos, real_orn, fingers])

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
        self.get_extended_observation()
        base_action = self._base_controller.get_next_action(self._observation[:self._robot.getObservationDimension()],
                                                            self._observation[self._robot.getObservationDimension():])

        final_action = np.add(base_action, action)
        # final_action[0] = np.clip(final_action[0], -1, 1)
        # final_action[1] = np.clip(final_action[1], -1, 1)
        final_action[2] = np.clip(final_action[2], -1, 1)

        self._robot.applyAction(final_action[0].tolist() + final_action[1].tolist())

        # grasp object
        if final_action[2] < 0:
            if not self._robot.isGrasping() and self._robot.checkContactPalm():
                self._t_grasp += self._time_step * self._action_repeat
            else:
                self._t_grasp = 0

            if self._t_grasp >= 0.5:
                obj_pose = self._observation[self._robot.getObservationDimension():self._robot.getObservationDimension()+6]
                self._robot.graspObject(self._world.obj_id, obj_pose)

        elif final_action[2] > 0:
            self._robot.releaseObject()

        for _ in range(self._action_repeat):
            p.stepSimulation()
            if self._termination():
                break

            self._envStepCounter += 1

        obs = self.get_extended_observation()

        done = self._termination()
        reward = self._compute_reward()

        print("reward")
        print(reward)

        return obs, np.array(reward), np.array(done), {}

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

        if self.terminated or self._envStepCounter > self._maxSteps:
            self.get_extended_observation()
            return np.float32(1.0)

        self.terminated = np.float32(0.0)
        obs = self.get_extended_observation()[self._robot.getObservationDimension():]

        # early termination if object falls
        if self._terminal_failure and self._object_fallen(obs[3], obs[4]):
            print("FALLEN")
            self.terminated = np.float32(1.0)

        # here check lift for termination
        if self._object_lifted(obs[2], obs[-1]) and self._t_lift >= 2:
            print("SUCCESS")
            self.terminated = np.float32(1.0)

        return self.terminated

    def _compute_reward(self):
        c1, c2, r = np.float32(0.0), np.float32(0.0), np.float32(0.0)
        w_obs = self.get_extended_observation()[self._robot.getObservationDimension():]

        # cost 1: trajectory as short as possible
        if not self._robot.isGrasping():
            c1 = 1/2000 * self._envStepCounter

        # cost 2: object falls
        if self._object_fallen(w_obs[3], w_obs[4]):
            c2 = np.float32(10.0)

        if self._robot.isGrasping() or self._robot.checkContactPalm():
            r += 1

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
