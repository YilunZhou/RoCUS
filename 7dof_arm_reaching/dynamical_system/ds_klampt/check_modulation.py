
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from modulation import Modulator
import sys
epsilon = sys.float_info.epsilon

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.3*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_box(ax, center, size, color='C0', alpha=0.2):
    xc, yc, zc = center
    xs, ys, zs = size
    xl, xh = xc - xs / 2, xc + xs / 2
    yl, yh = yc - ys / 2, yc + ys / 2
    zl, zh = zc - zs / 2, zc + zs / 2

    xx, yy = np.meshgrid([xl, xh], [yl, yh])
    ax.plot_surface(xx, yy, np.array([zl, zl, zl, zl]).reshape(2, 2), color=color, alpha=alpha)
    ax.plot_surface(xx, yy, np.array([zh, zh, zh, zh]).reshape(2, 2), color=color, alpha=alpha)

    xx, zz = np.meshgrid([xl, xh], [zl, zh])
    ax.plot_surface(xx, np.array([yl, yl, yl, yl]).reshape(2, 2), zz, color=color, alpha=alpha)
    ax.plot_surface(xx, np.array([yh, yh, yh, yh]).reshape(2, 2), zz, color=color, alpha=alpha)

    yy, zz = np.meshgrid([yl, yh], [zl, zh])
    ax.plot_surface(np.array([xl, xl, xl, xl]).reshape(2, 2), yy, zz, color=color, alpha=alpha)
    ax.plot_surface(np.array([xh, xh, xh, xh]).reshape(2, 2), yy, zz, color=color, alpha=alpha)


def rand_target_loc():
    '''
    generate random target location
    '''
    x = np.random.uniform(low=0.05, high=0.5)
    if np.random.randint(0, 2) == 0:
        x = -x
    y = np.random.uniform(low=-0.3, high=0.2)
    z = np.random.uniform(low=0.65, high=1.0)
    # z = np.random.uniform(low=0.65, high=0.95)
    return x, y, z

# For a pre-learned model
model_filename = "../dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS_bounded_pyb.pkl"
reference_points  = np.array([[0, 0, 0.975], [0.0, 0.5, 0.80], [0.0, 0, 0.61]])
reference_points  = np.array([[0, 0.15, 0.975], [0.0, 0.5, 0.80], [0.0, 0, 0.61]])
m = Modulator(model_filename, reference_points)

initial_ee_loc = [0, -0.137, 1.173]  # this is the end effector position at the initial pose

plt.figure()
plt.subplot(1, 1, 1, projection='3d')
ax = plt.gca()
vertical_wall = [0, 0.1, 0.825], [0.01, 0.8, 0.4]
horizontal_wall = [0, 0.1, 1.025], [0.7, 0.8, 0.01]
table_top = [0, 0, 0.6], [1.5, 1, 0.05]
plot_box(ax, *vertical_wall, **{'color': 'C1'})
plot_box(ax, *horizontal_wall, **{'color': 'C1'})
plot_box(ax, *table_top, **{'color': 'C1'})

reference_points = m.get_reference_points()
for oo in range(len(reference_points)):
    ax.scatter(reference_points[oo][0], reference_points[oo][1], reference_points[oo][2], c='C7')


for _ in range(10):
    target_loc = rand_target_loc()
    x = initial_ee_loc    
    # Doing integration outside of function for "direct" robot control
    traj = [x]
    dt   = 0.05
    for _ in trange(1500):
        x_dot = m.get_modulated_direction(x, target_loc)
        x_dot = x_dot/np.linalg.norm(x_dot + epsilon) * 0.10
        x     = x + x_dot * dt
        traj.append(x)
    xs, ys, zs = zip(*traj)
    
    ax.plot(xs, ys, zs, 'C0')
    ax.scatter(target_loc[0], target_loc[1], target_loc[2], c='C3')
set_axes_equal(ax)
plt.show()


check_openloop_function = 0
if check_openloop_function:
    plt.figure()
    plt.subplot(1, 1, 1, projection='3d')
    ax = plt.gca()
    vertical_wall = [0, 0.1, 0.825], [0.01, 0.8, 0.4]
    horizontal_wall = [0, 0.1, 1.025], [0.7, 0.8, 0.01]
    table_top = [0, 0, 0.6], [1.5, 1, 0.05]
    plot_box(ax, *vertical_wall, **{'color': 'C1'})
    plot_box(ax, *horizontal_wall, **{'color': 'C1'})
    plot_box(ax, *table_top, **{'color': 'C1'})

    reference_points = m.get_reference_points()
    for oo in range(len(reference_points)):
        ax.scatter(reference_points[oo][0], reference_points[oo][1], reference_points[oo][2], c='C7')

    for _ in range(10):
        target_loc = rand_target_loc()
        x = initial_ee_loc

        # Using the get_openloop_trajectory function
        traj = m.get_openloop_trajectory(x, target_loc, dt = 0.03, eps = 0.03, max_N = 2000)    
        
        ax.plot(traj[0,:], traj[1,:], traj[2,:], 'C0')
        ax.scatter(target_loc[0], target_loc[1], target_loc[2], c='C3')
    set_axes_equal(ax)
    plt.show()