import numpy as np
import math as m
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import struct

sns.set()


def goal_distance(a: np.ndarray, b: np.ndarray):
    if not a.shape == b.shape:
        raise AssertionError("goal_distance(): shape of points mismatch")
    return np.linalg.norm(a - b, axis=-1)


def quat_distance(a: np.ndarray, b: np.ndarray):
    if not a.shape == b.shape and a.shape == 4:
        raise AssertionError("quat_distance(): wrong shape of points")
    elif not (np.linalg.norm(a) == 1.0 and np.linalg.norm(b) == 1.0):
        warnings.warn("quat_distance(): vector(s) without unitary norm {} , {}".format(np.linalg.norm(a), np.linalg.norm(b)))

    inner_quat_prod = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    dist = 1 - inner_quat_prod*inner_quat_prod
    return dist


def quat_multiplication(a: np.ndarray, b: np.ndarray):

    if not a.shape == b.shape and a.shape == 4:
        raise AssertionError("quat_distance(): wrong shape of points")
    elif not (np.linalg.norm(a) == 1.0 and np.linalg.norm(b) == 1.0):
        warnings.warn("quat_distance(): vector(s) without unitary norm {} , {}".format(np.linalg.norm(a), np.linalg.norm(b)))

    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]

    x12 = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y12 = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z12 = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w12 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x12, y12, z12, w12])


def axis_angle_to_quaternion(vec_aa: tuple):
    qx = vec_aa[0] * m.sin(vec_aa[3] / 2)
    qy = vec_aa[1] * m.sin(vec_aa[3] / 2)
    qz = vec_aa[2] * m.sin(vec_aa[3] / 2)
    qw = m.cos(vec_aa[3] / 2)
    quat = [qx, qy, qz, qw]
    return quat


def quaternion_to_axis_angle(quat: tuple):
    angle = 2 * m.acos(quat[3])
    x = quat[0] / m.sqrt(1 - quat[3] * quat[3])
    y = quat[1] / m.sqrt(1 - quat[3] * quat[3])
    z = quat[2] / m.sqrt(1 - quat[3] * quat[3])
    vec_aa = [x, y, z, angle]
    return vec_aa


def floor_vec(vec: tuple):
    r_vec = [0]*len(vec)
    for i, v in enumerate(vec):
        r_vec[i] = np.sign(v) * m.floor(m.fabs(v) * 100) / 100
    return r_vec


def sph_coord(x: float, y: float, z: float):
    ro = m.sqrt(x*x + y*y + z*z)
    theta = m.acos(z/ro)
    phi = m.atan2(y,x)
    return [ro, theta, phi]


def plot_trajectories_from_file(file_1, file_2):

    f1 = open(file_1, 'rb')
    f2 = open(file_2, 'rb')
    lines_list_1 = f1.readlines()
    lines_list_2 = f2.readlines()

    action_list_1 = []
    for ln in range(0, len(lines_list_1)):
        tmp_list = []
        for val in lines_list_1[ln].split():
            tmp_list.extend([float(val)])

        action_list_1.append(tmp_list[:3])

    action_list_2 = []
    for ln in range(0, len(lines_list_2)):
        tmp_list = []
        for val in lines_list_2[ln].split():
            tmp_list.extend([float(val)])

        action_list_2.append(tmp_list[:3])

    df_1 = pd.DataFrame(action_list_1, columns=['x', 'y', 'z'])
    df_2 = pd.DataFrame(action_list_2, columns=['x', 'y', 'z'])

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Add title and axis names
    ax.plot(df_1['x'], df_1['y'], df_1['z'], marker='', color='orange', linewidth=4, alpha=0.7)
    ax.plot(df_2['x'], df_2['y'], df_2['z'], marker='', color='grey', linewidth=1, alpha=0.4)

    # Rotate it
    ax.view_init(30, 45)

    plt.style.use('seaborn-darkgrid')
    plt.title('My title')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def read_pybullet_log_file(filename, verbose=True):
    f = open(filename, 'rb')

    print('Opened'),
    print(filename)

    keys = f.readline().decode('utf8').rstrip('\n').split(',')
    fmt = f.readline().decode('utf8').rstrip('\n')

    # The byte number of one record
    sz = struct.calcsize(fmt)
    # The type number of one record
    ncols = len(fmt)

    if verbose:
        print('Keys:'),
        print(keys)
        print('Format:'),
        print(fmt)
        print('Size:'),
        print(sz)
        print('Columns:'),
        print(ncols)

    # Read data
    wholeFile = f.read()
    # split by alignment word
    chunks = wholeFile.split(b'\xaa\xbb')
    log = list()
    for chunk in chunks:
        if len(chunk) == sz:
            values = struct.unpack(fmt, chunk)
            record = list()
            for i in range(ncols):
                record.append(values[i])
            log.append(record)

    return keys, log


def plot_contact_normal_forces(filename):
    keys, log = read_pybullet_log_file(filename, verbose=True)

    # divido per link e per ogni link salvo [timestamp, normalforce]
    link_idx = keys.index('linkIndexB')
    timestamp_idx = keys.index('timeStamp')
    n_force_idx = keys.index('normalForce')

    data ={}
    timestamp_max = 0
    force_max = 0
    for ln in log:
        # if is a new link add it to the data keys
        if not ln[link_idx] in data:
            data[ln[link_idx]] = np.zeros([1, 2])

        data[ln[link_idx]] = np.append(data[ln[link_idx]], [[ln[timestamp_idx], ln[n_force_idx]]], axis=0)
        timestamp_max = max(timestamp_max, ln[timestamp_idx])
        force_max = max(force_max, ln[n_force_idx])

    print('data keys: {}'.format(data.keys()))

    # Make a data frame
    a = []
    for k in sorted(data):
        values = np.transpose(data[k])
        a.append(pd.DataFrame({'x': values[0], str(k): values[1]}))

    #df = pd.DataFrame({'x': np.linspace(0, timestamp_max, num=10, endpoint=True), 'y1': np.random.randn(10)})

    # Initialize the figure
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set3')

    # multiple line plot
    num = 0
    for df in a:
        num += 1
        df_keys = df.keys()

        # Find the right spot on the plot
        plt.subplot(3, 4, num)

        # plot every groups, but discreet
        #for v in df.drop('x', axis=1):
        #    plt.plot(df['x'], df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)

        # Plot the lineplot
        plt.plot(df[df_keys[0]], df[df_keys[1]], marker='o', color=palette(num), linewidth=2, alpha=0.9, label=df_keys[1])

        # Same limits for everybody!
        plt.xlim(0, timestamp_max)
        plt.ylim(-10, 75)

        # Not ticks everywhere
        if num in range(9):
            plt.tick_params(labelbottom='off')
        if num not in [1, 5, 9]:
            plt.tick_params(labelleft='off')

        # Add title
        plt.title(df_keys[1], loc='left', fontsize=12, fontweight=0, color=palette(num))

    # general title
    plt.suptitle("normal force on contacts for each link in contact", fontsize=13, fontweight=0, color='black',
                 style='italic', y=0.95)

    # Axis title
    plt.text(2, -20, 'Time (s)', ha='center', va='center')
    plt.text(-3.5, 150, 'Force (N)', ha='center', va='center', rotation='vertical')

    plt.show()


#file_1 = '/home/erampone/workspace/phd/robot-agents/log_successful_grasp.bin'
file_2 = '/home/erampone/workspace/phd/robot-agents/log_successful_grasp_ct.bin'

plot_contact_normal_forces(file_2)
