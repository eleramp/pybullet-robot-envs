import logging
import gym
from gym.envs.registration import register

#register(
#        id='iCubReach-v0',
#        entry_point='pybullet_robot_envs.envs:iCubReachGymEnv',
#        max_episode_steps=1000,
#        kwargs={ 'useIK':1, 'isDiscrete':0, 'control_arm':'l', 'useOrientation':0, 'rnd_obj_pose':1, 'maxSteps':1000},
#)

#register(
#        id='iCubPush-v0',
#        entry_point='pybullet_robot_envs.envs:iCubPushGymEnv',
#        max_episode_steps=1000,
#        kwargs={ 'useIK':1, 'isDiscrete':0, 'control_arm':'l', 'useOrientation':0, 'rnd_obj_pose':1, 'maxSteps':1000, 'reward_type':0},
#)

register(
        id='iCubGraspResidual-v0',
        entry_point='pybullet_robot_envs.envs:iCubGraspResidualGymEnv',
        max_episode_steps=1000,
        kwargs={'control_arm': 'l',
                'useOrientation': 1,
                'rnd_obj_pose': 0.05,
                'noise_pcl': 0.01,
                'maxSteps': 3000,
                'renders': False,
                'terminal_failure': True},
)


register(
        id='pandaReach-v0',
        entry_point='pybullet_robot_envs.envs:pandaReachGymEnv',
        max_episode_steps=1000,
        kwargs={
                 'useIK':0,
                 'isDiscrete':0,
                 'actionRepeat':1,
                 'renders':False,
                 'numControlledJoints':7, 'fixedPositionObj':False, 'includeVelObs':True},
)

register(
        id='pandaPush-v0',
        entry_point='pybullet_robot_envs.envs:pandaPushGymEnv',
        max_episode_steps=1000,
        kwargs={
                 'useIK':0,
                 'isDiscrete':0,
                 'actionRepeat':1,
                 'renders':False,
                 'numControlledJoints':7, 'fixedPositionObj':False, 'includeVelObs':True},
)

# --------------------------- #
def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
    return btenvs

getList()
