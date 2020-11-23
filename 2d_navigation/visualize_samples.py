import warnings
warnings.filterwarnings('ignore')
import sys, pickle, argparse
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from environment import RBF2dGymEnv

def plot_trajs(trajs, line_style, alpha):
    for traj in trajs:
        plt.plot(traj[:, 0], traj[:, 1], line_style, alpha=alpha)

def plot_prior_trajs(prior_fn, success_only=True, line_style='C0', alpha=0.3, num=500, legend=None):
    if success_only:
        keep = lambda t: np.linalg.norm(t[-1] - [1, 1]) <= 0.03
    else:
        keep = lambda t: True
    prior_data = pickle.load(open(prior_fn, 'rb'))
    trajs = [t[2] for t in prior_data if t is not None and keep(t[2])]
    trajs = trajs[:num]
    plot_trajs(trajs, line_style, alpha)
    if legend is not None:
        legend.extend(['prior'] + ['_'] * (len(trajs) - 1))

def plot_posterior_trajs(posterior_fn, line_style='C1', alpha=0.6, num=100, warmup_discard=2000, legend=None):
    posterior_data = pickle.load(open(posterior_fn, 'rb'))
    trajs = posterior_data['trajectory'][warmup_discard:]
    trajs = trajs[::int(len(trajs) / num)]
    plot_trajs(trajs, line_style, alpha)
    if legend is not None:
        legend.extend(['posterior'] + ['_'] * (len(trajs) - 1))

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

def aggregate_prior_envs(prior_fn, success_only=True, num=500):
    if success_only:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', help='.pkl file containing the prior samples')
    parser.add_argument('--posterior', help='.pkl file containing the posterior samples')
    parser.add_argument('--save-fn', help='file name to save the image')
    args = parser.parse_args()
    assert args.prior is not None or args.posterior is not None, 'At least one of prior and posterior needs to be specified'

    plt.figure(figsize=[8, 4])

    plt.subplot(1, 2, 1)
    legend = []
    if args.prior is not None:
        plot_prior_trajs(args.prior, legend=legend)
    if args.posterior is not None:
        plot_posterior_trajs(args.posterior, legend=legend)
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.gca().set_aspect('equal')
    plt.legend(legend)

    plt.subplot(1, 2, 2)
    if args.prior is not None:
        prior_env = aggregate_prior_envs(args.prior)
    if args.posterior is not None:
        posterior_env = aggregate_posterior_envs(args.posterior)
    if args.prior is not None and args.posterior is not None:
        diff = posterior_env - prior_env
        plt.imshow(diff, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', vmin=-abs(diff).max(), vmax=abs(diff).max(), cmap='coolwarm')
    elif args.prior is not None:
        plt.imshow(prior_env, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='gray_r')
    elif args.posterior is not None:
        plt.imshow(posterior_env, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='gray_r')
    plt.plot([1], [1], 'C3*', markersize=10)
    plt.plot([-1], [-1], 'C2o', markersize=7)
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    if args.save_fn is not None:
        plt.savefig(args.save_fn, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
