import logging
import gym
from gym.envs.registration import register

register(
        id='iCubReach-v0',
        entry_point='pybullet_robot_envs.envs:iCubReachGymEnv',
        max_episode_steps=1000,
        kwargs={ 'useIK': 1,
                 'isDiscrete': 0,
                 'control_arm': 'l',
                 'control_orientation': 0,
                 'rnd_obj_pose': 1,
                 'max_steps': 1000},
)

register(
        id='iCubPush-v0',
        entry_point='pybullet_robot_envs.envs:iCubPushGymEnv',
        max_episode_steps=1000,
        kwargs={'useIK': 1,
                'discrete_action': 0,
                'control_arm': 'l',
                'control_orientation': 1,
                'obj_pose_rnd_std': 0.05,
                'max_steps': 1000,
                'reward_type': 0},
)

register(
        id='iCubPushGoal-v0',
        entry_point='pybullet_robot_envs.envs:iCubPushGymGoalEnv',
        max_episode_steps=1000,
        kwargs={ 'use_IK': 1,
                 'discrete_action': 0,
                 'control_arm': 'r',
                 'control_orientation': 1,
                 'obj_pose_rnd_std': 0.0,
                 'max_steps': 1000},
)

register(
        id='iCubGrasp-v0',
        entry_point='pybullet_robot_envs.envs:iCubGraspGymEnv',
        max_episode_steps=1000,
        kwargs={'control_arm': 'r',
                'control_orientation': 1,
                'obj_pose_rnd_std': 0.05,
                'max_steps': 1000,
                'renders': False},
)

register(
        id='iCubGraspResidual-v0',
        entry_point='pybullet_robot_envs.envs:iCubGraspResidualGymEnv',
        max_episode_steps=1000,
        kwargs={'control_arm': 'r',
                'control_orientation': 1,
                'obj_pose_rnd_std': 0.05,
                'noise_pcl': 0.0,
                'use_superq': 1,
                'max_steps': 1000,
                'renders': False},
)

register(
        id='iCubGraspResidualGoal-v0',
        entry_point='pybullet_robot_envs.envs:iCubGraspResidualGymGoalEnv',
        max_episode_steps=1000,
        kwargs={'control_arm': 'r',
                'control_orientation': 1,
                'control_eu_or_quat': 0,
                'obj_pose_rnd_std': 0.05,
                'noise_pcl': 0.0,
                'max_steps': 1000,
                'renders': False,
                'log_file': '/home/erampone/workspace/phd'},
)

register(
        id='iCubReachResidualGoal-v0',
        entry_point='pybullet_robot_envs.envs:iCubReachResidualGymGoalEnv',
        max_episode_steps=1000,
        kwargs={'control_arm': 'r',
                'control_orientation': 1,
                'control_eu_or_quat': 0,
                'obj_pose_rnd_std': 0.05,
                'noise_pcl': 0.0,
                'max_steps': 1000,
                'renders': False,
                'log_file': '/home/erampone/workspace/phd'},
)


register(
        id='pandaReach-v0',
        entry_point='pybullet_robot_envs.envs:pandaReachGymEnv',
        max_episode_steps=1000,
        kwargs={
                 'useIK': 0,
                 'isDiscrete': 0,
                 'actionRepeat': 1,
                 'renders': False,
                 'numControlledJoints': 7,
                  'fixedPositionObj': False,
                  'includeVelObs': True},
)

register(
        id='pandaPush-v0',
        entry_point='pybullet_robot_envs.envs:pandaPushGymEnv',
        max_episode_steps=1000,
        kwargs={
                 'useIK': 0,
                 'isDiscrete': 0,
                 'actionRepeat': 1,
                 'renders': False,
                 'numControlledJoints': 7,
                 'fixedPositionObj': False,
                 'includeVelObs': True},
)

# --------------------------- #
def getList():
    print("getlist:")
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
    return btenvs

getList()
