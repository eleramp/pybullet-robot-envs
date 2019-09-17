import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,currentdir)

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
from . import goal_distance

largeValObservation = 100

RENDER_HEIGHT = 240
RENDER_WIDTH = 320


class iCubGraspGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50 }

    def __init__(self,
                 urdfRoot=robot_data.getDataPath(),
                 actionRepeat=1,
                 isDiscrete=0,
                 control_arm='l',
                 useOrientation=0,
                 rnd_obj_pose=1,
                 renders=False,
                 maxSteps = 1000):

        self._control_arm = control_arm
        self._timeStep = 1./240.
        self._useOrientation = useOrientation
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._robot = []
        self._world = []
        self._target_dist_min = 0.05
        self._rnd_obj_pose = rnd_obj_pose

        self._base_controller = []
        self._superq = []
        self._grasp_pose = []

        # Initialize PyBullet simulator
        self._p = p
        if self._renders:
            self._cid = p.connect(p.SHARED_MEMORY)
            if self._cid < 0:
                self._cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5,90,-60,[0.0,-0.0,-0.0])
        else:
            self._cid = p.connect(p.DIRECT)

        # initialize simulation environment
        self.seed()
        self.reset()

        observationDim = len(self._observation)
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
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])

        # Load robot
        self._robot = iCubEnv(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, useInverseKinematics=1,
                              arm=self._control_arm, useOrientation=self._useOrientation)

        # Load world environment
        self._world = WorldFetchEnv(rnd_obj_pose=self._rnd_obj_pose, workspace_lim=self._robot.workspace_lim)

        # limit iCub workspace to table plane
        self._robot.workspace_lim[2][0] = self._world.get_table_height()

        p.setGravity(0, 0, -9.8)

        # Let the world run for a bit
        for _ in range(500):
            p.stepSimulation()

        self._observation = self.get_extended_observation()

        self._robot.debug_gui()
        self._world.debug_gui()

        # Load base controller
        self._base_controller = SuperqGraspPlanner(robot_base_pose=p.getBasePositionAndOrientation(self._robot.icubId))
        self._base_controller.reset(self._observation[:6])

        self._base_controller.set_object_info(self._world.get_object_shape_info())
        if self._base_controller.compute_object_pointcloud(self._world.get_observation()):
            self._superq = self._base_controller.estimate_superq()
        else:
            print("Can't get good point cloud of the object")

        self._grasp_pose = self._base_controller.estimate_grasp()

        print("object pose: {}".format(self._world.get_observation()))
        print("superq pose: {} {}".format(self._superq[0].center, self._superq[0].ea))
        print("grasp pose: {}".format(self._grasp_pose))

        pose = self._superq[0].center
        p.addUserDebugLine([pose[0][0], pose[1][0], pose[2][0]], [pose[0][0]+0.1, pose[1][0], pose[2][0]], [1, 0, 0], parentObjectUniqueId=self._robot.icubId, parentLinkIndex=-1)
        p.addUserDebugLine([pose[0][0], pose[1][0], pose[2][0]], [pose[0][0], pose[1][0]+0.1, pose[2][0]], [0, 1, 0], parentObjectUniqueId=self._robot.icubId, parentLinkIndex=-1)
        p.addUserDebugLine([pose[0][0], pose[1][0], pose[2][0]], [pose[0][0], pose[1][0], pose[2][0]+0.1], [0, 0, 1], parentObjectUniqueId=self._robot.icubId, parentLinkIndex=-1)

        pose = self._grasp_pose[:3]
        p.addUserDebugLine(pose, [pose[0]+0.2,pose[1],pose[2]], [1, 0, 0], parentObjectUniqueId=self._robot.icubId, parentLinkIndex=-1)
        p.addUserDebugLine(pose, [pose[0],pose[1]+0.2,pose[2]], [0, 1, 0], parentObjectUniqueId=self._robot.icubId, parentLinkIndex=-1)
        p.addUserDebugLine(pose, [pose[0],pose[1],pose[2]+0.2], [0, 0, 1], parentObjectUniqueId=self._robot.icubId, parentLinkIndex=-1)

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

        self._observation.extend(list(robot_observation))
        self._observation.extend(list(world_observation))
        self._observation.extend(list(obj_pos_in_hand))
        self._observation.extend(list(obj_euler_in_hand))

        return np.array(self._observation)

    def step(self, action):

        dv = 0.01
        real_pos = [a*0.003 for a in action[:3]]
        real_orn = []
        if self.action_space.shape[-1] is 6:
            real_orn = [a*0.01 for a in action[3:]]

        return self.step2(real_pos+real_orn)


    def step2(self, action):

        # tracker --> check how to put it in separate thread
        if self._base_controller.check_object_moved(self._world.get_observation()):
            ok = self._base_controller.compute_object_pointcloud(self._world.get_observation())
            if ok:
                self._superq = self._base_controller.estimate_superq()

        # get action from base controller
        base_action = self._base_controller.get_next_action(self._robot.hand_pose)

        final_action = np.array(list(base_action)) + np.array(list(action))

        for _ in range(self._actionRepeat):
            self._robot.applyAction(final_action.tolist())
            p.stepSimulation()

            if self._termination():
                break

            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.get_extended_observation()

        done = self._termination()
        reward = self._compute_reward()

        #print("reward")
        #print(reward)

        return self._observation, np.array(reward), np.array(done), {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array", close=False):
        ## TODO Check the behavior of this function
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
            self._observation = self.get_extended_observation()
            return np.float32(1.0)

        hand_pos = self._robot.getObservation()[:3]
        obj_pos = self._world.get_observation()[:3]

        d = goal_distance(np.array(hand_pos), np.array(obj_pos))

        if d <= self._target_dist_min:
            self.terminated = 1
        return (d <= self._target_dist_min)

    def _compute_reward(self):

        reward = np.float32(0.0)

        hand_pos = self._robot.getObservation()[:3]
        obj_pos = self._world.get_observation()[:3]

        d1 = goal_distance(np.array(hand_pos), np.array(obj_pos))
        #print("distance")
        #print(d1)

        reward = -d1
        if d1 <= self._target_dist_min:
            reward = np.float32(100.0)

        return reward
