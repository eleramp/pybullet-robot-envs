import logging
import gym
from gym.envs.registration import register

register(
        id='iCubReach-v0',
        entry_point='pybullet_robot_envs.envs:iCubReachGymEnv',
        max_episode_steps=500,
        kwargs={ 'control_arm': 'r',
                 'control_orientation': 1,
                 'obj_pose_rnd_std': 0.05,
                 'max_steps': 500,
                 'renders': False},
)

register(
        id='iCubPush-v0',
        entry_point='pybullet_robot_envs.envs:iCubPushGymEnv',
        max_episode_steps=1000,
        kwargs={'use_IK': 1,
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
                 'obj_pose_rnd_std': 0.05,
                 'max_steps': 1000},
)

register(
        id='iCubReachResidual-v0',
        entry_point='pybullet_robot_envs.envs:iCubReachResidualGymEnv',
        max_episode_steps=1000,
        kwargs={'control_arm': 'r',
                'control_orientation': 1,
                'obj_pose_rnd_std': 0.05,
                'noise_pcl': 0.0,
                'use_superq': 1,
                'max_steps': 500,
                'n_control_pt': 4,
                'r_weights': [-5, -10, 10],
                'obj_name': 1,
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
                'n_control_pt': 4,
                'obj_name': 0,
                'renders': False},
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

register(
        id='pandaGraspResidual-v0',
        entry_point='pybullet_robot_envs.envs:PandaGraspResidualGymEnv',
        max_episode_steps=1000,
        kwargs={'control_orientation': 1,
                'obj_pose_rnd_std': 0.05,
                'noise_pcl': 0.0,
                'use_superq': 1,
                'max_steps': 1000,
                'n_control_pt': 2,
                'obj_name': None,
                'renders': False},
)

register(
        id='PandaGraspResidualGymEnvSqObj-v0',
        entry_point='pybullet_robot_envs.envs:PandaGraspResidualGymEnvSqObj',
        max_episode_steps=1000,
        kwargs={'dset': 'train',
                'control_orientation': 1,
                'control_eu_or_quat': 0,
                'obj_pose_rnd_std': 0.0,
                'obj_orn_rnd': 1.0,
                'noise_pcl': 0.00,
                'use_superq': 1,
                'max_steps': 1000,
                'n_control_pt': 2,
                'renders': False},
)

# --------------------------- #
def getList():
    print("getlist:")
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('iCub')>=0]
    return btenvs

getList()
