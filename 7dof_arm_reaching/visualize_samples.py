import warnings
warnings.filterwarnings('ignore')
import sys, pickle, argparse
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from environment import PandaEnv


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

def plot_environment(ax):
    vertical_wall = [0, 0.1, 0.825], [0.01, 0.8, 0.4]
    horizontal_wall = [0, 0.1, 1.025], [0.7, 0.8, 0.01]
    table_top = [0, 0, 0.6], [1.5, 1, 0.05]
    plot_box(ax, *vertical_wall, **{'color': 'C2'})
    plot_box(ax, *horizontal_wall, **{'color': 'C2'})
    plot_box(ax, *table_top, **{'color': 'C2'})
    plot_box(ax, [0.55/2, -0.05, 1.65/2], [0.45, 0.5, 0.35], **{'color': 'C1'})
    plot_box(ax, [-0.55/2, -0.05, 1.65/2], [0.45, 0.5, 0.35], **{'color': 'C1'})

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.3 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def visualize_trajectories(ax, prior_fn=None, posterior_fn=None, prior_num=100, posterior_num=100):
    if prior_fn is not None:
        trajectories = [d[2] for d in pickle.load(open(prior_fn, 'rb'))]
        for traj in trajectories[::int(len(trajectories) / prior_num)]:
            if traj is None:
                continue
            ax.plot(xs=traj[:, 7], ys=traj[:, 8], zs=traj[:, 9], c='C0', alpha=0.3)
    if posterior_fn is not None:
        trajectories = pickle.load(open(posterior_fn, 'rb'))['trajectory']
        for traj in trajectories[::int(len(trajectories) / posterior_num)]:
            if traj is None:
                continue
            ax.plot(xs=traj[:, 7], ys=traj[:, 8], zs=traj[:, 9], c='C3', alpha=0.8)
    plot_environment(ax)
    set_axes_equal(ax)
    plt.axis('off')

def visualize_target_locs(ax, prior_fn=None, posterior_fn=None, prior_num=500, posterior_num=500):
    if prior_fn is not None:
        data = pickle.load(open(prior_fn, 'rb'))
        target_locs = [d[0] for d in data[:prior_num]]
        xs, ys, zs = zip(*target_locs)
        sc = ax.scatter(xs=xs, ys=ys, zs=zs, alpha=0.3)
    if posterior_fn is not None:
        data = pickle.load(open(posterior_fn, 'rb'))
        target_locs = data['env'][1000:]
        freq = int(len(target_locs) / posterior_num)
        target_locs = target_locs[::freq]
        behaviors = data['behavior'][1000:][::freq]
        xs, ys, zs = zip(*target_locs)
        sc = ax.scatter(xs=xs, ys=ys, zs=zs, c=behaviors, cmap='Reds_r')
    plot_environment(ax)
    set_axes_equal(ax)
    plt.axis('off')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', help='.pkl file containing the prior samples')
    parser.add_argument('--posterior', help='.pkl file containing the posterior samples')
    parser.add_argument('--save-fn', help='file name to save the image')
    args = parser.parse_args()
    assert args.prior is not None or args.posterior is not None, 'At least one of prior and posterior needs to be specified'

    plt.figure(figsize=[8, 4])

    plt.subplot(1, 2, 1, projection='3d')
    visualize_target_locs(plt.gca(), prior_fn=args.prior, posterior_fn=args.posterior)

    plt.subplot(1, 2, 2, projection='3d')
    visualize_trajectories(plt.gca(), prior_fn=args.prior, posterior_fn=args.posterior, prior_num=100, posterior_num=100)

    plt.tight_layout()
    if args.save_fn is not None:
        plt.savefig(args.save_fn, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
