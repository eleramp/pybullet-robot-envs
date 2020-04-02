#!/usr/bin/env python
import math

robots = [
        'icub',
        'icub_hands',
        'panda',
    ]

objects = {
        'YcbMustardBottle': 'default',
        'YcbTomatoSoupCan': 'cylinder',
        'YcbCrackerBox': 'box',
        'YcbFoamBrick': 'box',
        'YcbGelatinBox': 'box',
        'YcbMasterChefCan': 'box',
        'YcbPear': 'sphere',
        'YcbPottedMeatCan': 'cylinder',
        'YcbTennisBall': 'sphere',
        'YcbChipsCan': 'cylinder',
    }


def get_robot_name_list():
    return robots


mode = {
    'control_arms': 'right',
}

sq_model = {
    'object_class': 'default',  # cylinder, box, sphere, default
    'tol': 1e-5,
    'optimizer_points': 50,
    'random_sampling': True,
    'merge_model': True,
    'minimum_points': 150,
}

sq_grasp = {
    robots[0]: {
        'tol': 1e-5,
        'constr_tol': 1e-4,
        'max_superq': 1,
        'plane_table': [0.0, 0.0, 1.0, 0.09],  # check this
        'displacement': [0.00, 0.00, 0.00],  # check this
        'hand_sq': [0.03, 0.06, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # how to set the following bounds?
        'bounds_right': [[-0.5, 0.0], [-0.25, 0.25], [-0.3, 0.3], [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]],
        'bounds_left': [[-0.5, 0.0], [-0.25, 0.25], [-0.3, 0.3], [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]],
        # 'bounds_constr_right': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.001, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001, 10.0, 0.00001, 10.0],
        # 'bounds_constr_left': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.01, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001, 10.0, 0.00001, 10.0],
        },

    robots[1]: {
        'tol': 1e-5,
        'constr_tol': 1e-4,
        'max_superq': 1,
        'plane_table': [0.0, 0.0, 1.0, 0.09],  # check this
        'displacement': [0.00, 0.00, 0.00],  # check this
        'hand_sq': [0.03, 0.06, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # how to set the following bounds?
        'bounds_right': [[-0.5, 0.0], [-0.25, 0.25], [-0.3, 0.3], [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]],
        'bounds_left': [[-0.5, 0.0], [-0.25, 0.25], [-0.3, 0.3], [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]],
        # 'bounds_constr_right': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.001, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001,
        #                         10.0, 0.00001, 10.0],
        # 'bounds_constr_left': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.01, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001,
        #                        10.0, 0.00001, 10.0],
    },

    robots[2]: {
        'tol': 1e-5,
        'constr_tol': 1e-4,
        'max_superq': 1,
        'plane_table': [0.0, 0.0, 1.0, 0.05],  # check this
        'displacement': [0.00, 0.00, 0.003],  # check this
        'hand_sq': [0.03, 0.04, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # how to set the following bounds?
        'bounds_right': [[-0.7, -0.3], [-0.3, 0.3], [-0.2, 0.4], [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]],
        'bounds_left': [[-0.7, -0.3], [-0.3, 0.3], [-0.2, 0.4], [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]],
        # 'bounds_constr_right': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.001, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001,
        #                         10.0, 0.00001, 10.0],
        # 'bounds_constr_left': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.01, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001,
        #                        10.0, 0.00001, 10.0],
    }
}

visualizer = {
    'x': 0,
    'y': 0,
    'width': 600,
    'height': 600,
}
