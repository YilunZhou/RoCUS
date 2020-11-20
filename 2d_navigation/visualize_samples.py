import sys, pickle
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from environment import RBF2dGymEnv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds_prior_loc', default="samples/ds_prior.pkl", help=".pkl file containing DS prior samples")
parser.add_argument('--ds_behavior_loc', default=None, help=".pkl file containing sampled DS behavior")
parser.add_argument('--il_prior_loc', default="samples/il_prior.pkl", help=".pkl file containing IL prior samples")
parser.add_argument('--il_behavior_loc', default=None, help=".pkl file containing sampled IL behavior")
parser.add_argument('--rrt_prior_loc', default="samples/rrt_prior.pkl", help=".pkl file containing RRT prior samples")
parser.add_argument('--rrt_behavior_loc', default=None, help=".pkl file containing sampled RRT behavior")
parser.add_argument('--output_loc', default=None, help="where to save the image")


def plot_trajs(trajs, line_style, alpha):
    for traj in trajs:
        plt.plot(traj[:, 0], traj[:, 1], line_style, alpha=alpha)


def aggregate_envs(envs, res=500):
    env = RBF2dGymEnv(use_lidar=False, grid_res=res)
    xs, ys = np.meshgrid(np.linspace(-1.2, 1.2, res), np.linspace(-1.2, 1.2, res))
    xys = np.stack([xs.flatten(), ys.flatten()], axis=1)
    aggregate_env_img = np.zeros((res, res))
    for e in tqdm(envs):
        env.reset(obs_override=e)
        env_img = env.arena.occ_grid
        aggregate_env_img += env_img
    aggregate_env_img = aggregate_env_img / len(envs)
    return aggregate_env_img


def plot_prior_trajs(prior_fn, only_success, line_style, alpha, num=500):
    if only_success:
        keep = lambda t: np.linalg.norm(t[-1] - [1, 1]) <= 0.03
    else:
        keep = lambda t: True
    prior_data = pickle.load(open(prior_fn, 'rb'))
    trajs = [t[2] for t in prior_data if t is not None and keep(t[2])]
    trajs = trajs[:num]
    plot_trajs(trajs, line_style, alpha)


def plot_posterior_trajs(posterior_fn, line_style, alpha, num=500, warmup_discard=2000):
    posterior_data = pickle.load(open(posterior_fn, 'rb'))
    trajs = posterior_data['trajectory'][warmup_discard:]
    trajs = trajs[::int(len(trajs) / num)]
    plot_trajs(trajs, line_style, alpha)


def aggregate_prior_envs(prior_fn, only_success, num=500):
    if only_success:
        keep = lambda t: np.linalg.norm(t[-1] - [1, 1]) <= 0.03
    else:
        keep = lambda t: True
    prior_data = pickle.load(open(prior_fn, 'rb'))
    envs_trajs = [(t[0], t[2]) for t in prior_data if t is not None and keep(t[2])]
    envs, _ = zip(*envs_trajs)
    envs = envs[:num]
    return aggregate_envs(envs)


def aggregate_posterior_envs(posterior_fn, num=500, warmup_discard=2000):
    posterior_data = pickle.load(open(posterior_fn, 'rb'))
    envs = posterior_data['env'][warmup_discard:]
    envs = envs[::int(len(envs) / num)]
    return aggregate_envs(envs)


def main():
    args = parser.parse_args()
    column_count = sum(1 for e in [args.ds_behavior_loc, args.il_behavior_loc, args.rrt_behavior_loc] if e)

    img_agg_num = 1000
    plt.figure(figsize=[5, 4])
    gs = GridSpec(ncols=column_count, nrows=2)
    col_idx = 0

    def plot_traj(col_idx, prior, behavior):
        plt.subplot(gs[0, col_idx])
        plot_prior_trajs(prior, True, 'C0', 0.1, num=500)
        plot_posterior_trajs(behavior, 'C1', 0.6, num=100)
        plt.plot([1], [1], 'C3*', markersize=10)
        plt.plot([-1], [-1], 'C2o', markersize=7)
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.gca().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])

    def plot_obs(col_idx, prior, behavior):
        plt.subplot(gs[1, col_idx])
        prior_env = aggregate_prior_envs(prior, only_success=True, num=img_agg_num)
        posterior_env = aggregate_posterior_envs(behavior, num=img_agg_num)
        diff = prior_env - posterior_env
        plt.imshow(diff, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', vmin=-abs(diff).max(), vmax=abs(diff).max(),
                   cmap='coolwarm')
        plt.plot([1], [1], 'C3*', markersize=10)
        plt.plot([-1], [-1], 'C2o', markersize=7)
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.gca().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])

    if args.ds_behavior_loc is not None:
        plot_traj(col_idx, args.ds_prior_loc, args.ds_behavior_loc)
        plot_obs(col_idx, args.ds_prior_loc, args.ds_behavior_loc)
        plt.xlabel('Dynamical System')
        col_idx += 1

    if args.rrt_behavior_loc is not None:
        plot_traj(col_idx, args.rrt_prior_loc, args.rrt_behavior_loc)
        plot_obs(col_idx, args.rrt_prior_loc, args.rrt_behavior_loc)
        plt.xlabel('Rapidly-Exploring\nRandom Tree')
        col_idx += 1

    if args.il_behavior_loc is not None:
        plot_traj(col_idx, args.il_prior_loc, args.il_behavior_loc)
        plot_obs(col_idx, args.il_prior_loc, args.il_behavior_loc)
        plt.xlabel('Imitation Learning')
        col_idx += 1

    plt.tight_layout()
    if args.output_loc is not None:
        plt.savefig(args.output_loc, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
