import numpy as np

def center_deviation_behavior(traj, env=None):
    if traj is None:
        return None, False
    ee_xyz = traj[:, 7:10]
    u = np.array([np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3])
    d = np.dot(ee_xyz, u)
    projs = np.stack((d, d, d), axis=1) * u
    avg_dist = (np.linalg.norm(projs - ee_xyz, axis=1)).mean()
    return avg_dist, True


def ee_avg_jerk(traj, env=None):
    if traj is None:
        return None, False
    ee_xyz = traj[:, 7:10]
    first_derivative = np.gradient(ee_xyz, axis=0)
    second_derivative = np.gradient(first_derivative, axis=0)
    third_derivative = np.absolute(np.gradient(second_derivative, axis=0)).sum(axis=1)
    return np.mean(third_derivative), True


def ee_distance_behavior(traj, env=None):
    if traj is None:
        return None, False
    ee_xyz = traj[:, 7:10]
    ee_dxyz = ee_xyz[1:] - ee_xyz[:-1]
    dist = np.linalg.norm(ee_dxyz, axis=1).sum()
    return dist, True


def illegibility_behavior(traj, env):
    if traj is None:
        return None, False
    traj_x = traj[:, 7]
    left_violation = np.maximum(-traj_x, 0)
    right_violation = np.maximum(traj_x, 0)
    if env[0] < 0:  # target on the left
        return np.mean(right_violation), True
    else:  # target on the right
        return np.mean(left_violation), True