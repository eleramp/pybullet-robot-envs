#!/usr/bin/env python
import math

mode = {
    'control_arms': 'right',
}

sq_model = {
    'object_class': 'box',
    'tol': 1e-5,
    'optimizer_points': 50,
    'random_sampling': True,
    'merge_model': True,
    'minimum_points': 150,
    'fraction_pc': 8,
    'threshold_axis': 0.7,
    'threshold_section1': 0.6,
    'threshold_section2': 0.03,
}
sq_grasp = {
    'tol': 1e-5,
    'constr_tol': 1e-4,
    'max_superq': 4,
    'plane_table': [0.0, 0.0, 1.0, 0.65],  # check this
    'displacement': [0.0, 0.0, 0.0],  # check this
    'hand_sq': [0.03, 0.06, 0.03, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # how to set the following bounds?
    'bounds_right': [-0.5, 0.0, -0.2, 0.2, -0.3, 0.3, -math.pi, math.pi, -math.pi, math.pi, -math.pi, math.pi],
    'bounds_left': [-0.5, 0.0, -0.2, 0.2, -0.3, 0.3, -math.pi, math.pi, -math.pi, math.pi, -math.pi, math.pi],
    'bounds_constr_right': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.001, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001, 10.0, 0.00001, 10.0],
    'bounds_constr_left': [-10000, 0.0, -10000, 0.0, -10000, 0.0, 0.01, 10.0, 0.0, 1.0, 0.00001, 10.0, 0.00001, 10.0, 0.00001, 10.0],
}

visualizer = {
    'x': 0,
    'y': 0,
    'width': 600,
    'height': 600,
}

