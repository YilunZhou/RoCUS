import math as m
import pickle, os, time
from copy import deepcopy as copy
from multiprocessing import Pool
from tqdm import tqdm, trange

import numpy as np
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys

import pybullet
import gym, pybulletgym

from controller import *
from ik import IK
from rapidly_exploring_random_tree.rrt import *
from behavior import *
from kernel import *

from pybulletgym.envs.roboschool.envs.manipulation.panda_reacher_env import PandaReacherEnv


class PandaReacherEarlyTerminateEnv(PandaReacherEnv):
    '''
	a modified environment that runs for a maximum of 500 time steps, but terminates when the target is reached. 
	'''

    def __init__(self, shelf=True, timelimit=500, target_threshold=0.03):
        super(PandaReacherEarlyTerminateEnv, self).__init__(shelf=shelf)
        self.timelimit = timelimit
        self.target_threshold = target_threshold

    def reset(self, **kwargs):
        self.cur_time = 0
        super().reset(**kwargs)

    def step(self, a):
        s, r, done, info = super().step(a)
        self.cur_time += 1
        # print(self.cur_time)
        done = done or self.cur_time == self.timelimit or np.linalg.norm(s[19:22]) <= self.target_threshold
        return s, r, done, info

    def render(self, mode='human', **kwargs):
        super().render(mode=mode, **kwargs)

    def s(self):
        return self.robot.calc_state()

    def log_prior(self, target_loc):
        assert -0.5 <= target_loc[0] <= -0.05 or 0.05 <= target_loc[0] <= 0.5
        assert -0.3 <= target_loc[1] <= 0.2 and 0.65 <= target_loc[2] <= 1
        return 0



def get_sigma(alpha, prior_file, behavior_func, target_type, target_behavior=None, min_N=1000):
    assert target_type in ['match', 'maximal']
    data = pickle.load(open(prior_file, 'rb'))
    behaviors = []
    for ek_value, ck_value, traj in tqdm(data, ncols=78):
        behavior, acceptable = behavior_func(traj, ek_value)
        if not acceptable:
            continue
        behaviors.append(behavior)
    behaviors = np.array(behaviors)
    assert len(behaviors) > min_N, f'Insufficient number of acceptable trajectories: {len(behaviors)}/{min_N}'

    if target_type == 'match':
        assert target_behavior is not None
        dist = abs(behaviors - target_behavior)
        dist.sort()
        return dist[int(alpha * len(behaviors))] / np.sqrt(3), 0, 1
    else:
        mean = behaviors.mean()
        std = behaviors.std()
        behaviors = (behaviors - mean) / std
        betas = 1 / (1 + np.exp(-behaviors))
        dists = 1 - betas
        dists.sort()
        return dists[int(alpha * len(behaviors))] / np.sqrt(3), mean, std


def sample(N, alpha, prior_file, env, controller, behavior_func, env_kernel, controller_kernel,
           target_type, target_behavior=None, N_sigma=1000, sigma_override=None):
    def get_behavior(ek, ck):
        env.reset(target_loc=ek.value)
        traj = controller.get_trajectory(env, ck)
        behav, accep = behavior_func(traj, ek.value)
        return behav, accep, traj

    sigma, b_mean, b_std = get_sigma(alpha, prior_file, behavior_func, target_type, target_behavior, N_sigma)
    if sigma_override is not None:
        sigma = sigma_override
    if target_type == 'match':
        likelihood = norm(loc=target_behavior, scale=sigma)
    elif target_type == 'maximal':
        likelihood = norm(loc=1, scale=sigma)

    def log_posterior(ekv, ckv, b):
        if target_type == 'match':
            assert b_mean == 0 and b_std == 1
            return env.log_prior(ekv) + controller.log_prior(ckv) + likelihood.logpdf(b)
        else:
            b_scaled = (b - b_mean) / b_std
            beta = 1 / (1 + np.exp(-b_scaled))
            return env.log_prior(ekv) + controller.log_prior(ckv) + likelihood.logpdf(beta)

    behavior, acceptable, trajectory = get_behavior(env_kernel, controller_kernel)
    while not acceptable:  # just get to an environment with "acceptable" behavior
        env_kernel.sample_prior()
        controller_kernel.sample_prior()
        behavior, acceptable, trajectory = get_behavior(env_kernel, controller_kernel)
    log_post = log_posterior(env_kernel.value, controller_kernel.value, behavior)

    env_samples = [env_kernel.value]
    controller_samples = [controller_kernel.value]
    trajectories = [trajectory]
    behaviors = [behavior]
    tot_acc = 0
    bar = trange(N, ncols=78)
    for i in bar:
        e_f, e_b = env_kernel.propose()
        c_f, c_b = controller_kernel.propose()
        behavior, acceptable, trajectory = get_behavior(env_kernel, controller_kernel)
        if not acceptable:  # directly reject the proposal if the behavior is not "acceptable"
            trajectories.append(trajectories[-1])
            behaviors.append(behaviors[-1])
            env_kernel.revert()
            controller_kernel.revert()
        else:
            proposed_log_post = log_posterior(env_kernel.value, controller_kernel.value, behavior)
            accept_log_ratio = proposed_log_post - log_post + e_b + c_b - e_f - c_f
            accept_ratio = np.exp(accept_log_ratio)
            if np.random.random() < accept_ratio:  # accept
                trajectories.append(trajectory)
                behaviors.append(behavior)
                log_post = proposed_log_post
                tot_acc += 1
            else:  # reject
                trajectories.append(trajectories[-1])
                behaviors.append(behaviors[-1])
                env_kernel.revert()
                controller_kernel.revert()
        env_samples.append(env_kernel.value)
        controller_samples.append(controller_kernel.value)
        bar.set_description(f'%Acc: {tot_acc / (i + 1):0.2f}')
        bar.refresh()
    return env_samples, controller_samples, trajectories, behaviors


def env_traj_prior_samples(N, save_fn, env, controller, ek, ck):
    if os.path.isfile(save_fn):
        input('File already exists. Press Enter to append to the file. Press Ctrl-C to abort...')
        data = pickle.load(open(save_fn, 'rb'))
    else:
        data = []
    for _ in trange(N):
        ek.sample_prior()
        ck.sample_prior()
        env.reset(target_loc=ek.value)
        traj = controller.get_trajectory(env, ck)
        data.append((ek.value, ck.value, traj))
    pickle.dump(data, open(save_fn, 'wb'))


def sample_rrt_prior(N=2000, save_fn='rrt_prior.pkl'):
    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelNormal()
    rrt_kernel = RRTKernelNormal()
    env_traj_prior_samples(N, save_fn, env, rrt_controller, env_kernel, rrt_kernel)


def sample_rl_prior(N=2000, save_fn='rl_prior.pkl'):
    env = PandaReacherEarlyTerminateEnv()
    rl_controller = RLController()
    env_kernel = ReachingEnvKernelNormal()
    rl_kernel = TransitionKernel()
    env_traj_prior_samples(N, save_fn, env, rl_controller, env_kernel, rl_kernel)




def visualize_rrt():
    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelNormal()
    rrt_kernel = RRTKernelNormal()
    while True:
        env_kernel.sample_prior()
        rrt_kernel.sample_prior()
        env.reset(target_loc=env_kernel.value)
        traj = rrt_controller.get_trajectory(env, rrt_kernel)
        print(traj.shape)


def visualize_il():
    env = PandaReacherEarlyTerminateEnv()
    rl_controller = RLController()
    env_kernel = RBF2dEnvKernelNormal()
    rl_kernel = TransitionKernel()
    while True:
        env_kernel.sample_prior()
        rl_kernel.sample_prior()
        env.reset(target_loc=env_kernel.value)
        traj = rl_controller.get_trajectory(env, rl_kernel)
        print(traj.shape)


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
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_trajectories(prior_fn=None, posterior_fn=None, prior_num=100, posterior_num=100):
    if prior_fn is not None:
        trajectories = [d[2] for d in pickle.load(open(prior_fn, 'rb'))]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj in trajectories[::int(len(trajectories) / prior_num)]:
            if traj is None:
                continue
            ax.plot(xs=traj[:, 7], ys=traj[:, 8], zs=traj[:, 9], c='C0', alpha=0.5)

    if posterior_fn is not None:
        trajectories = pickle.load(open(posterior_fn, 'rb'))['trajectory']
        for traj in trajectories[::int(len(trajectories) / posterior_num)]:
            if traj is None:
                continue
            ax.plot(xs=traj[:, 7], ys=traj[:, 8], zs=traj[:, 9], c='C3', alpha=0.5)

    # vertical_wall = [0, 0.1, 0.825], [0.01, 0.8, 0.4]
    # horizontal_wall = [0, 0.1, 1.025], [0.7, 0.8, 0.01]
    # table_top = [0, 0, 0.6], [1.5, 1, 0.05]
    # plot_box(ax, *vertical_wall, **{'color': 'C2'})
    # plot_box(ax, *horizontal_wall, **{'color': 'C2'})
    # plot_box(ax, *table_top, **{'color': 'C2'})
    plot_box(ax, [0.55 / 2, -0.05, 1.65 / 2], [0.45, 0.5, 0.35], **{'color': 'C1'})
    plot_box(ax, [-0.55 / 2, -0.05, 1.65 / 2], [0.45, 0.5, 0.35], **{'color': 'C1'})
    set_axes_equal(ax)
    plt.show()


def visualize_target_locs(prior_fn=None, posterior_fn=None, prior_num=500, posterior_num=500):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # if prior_fn is not None:
    # 	data = pickle.load(open(prior_fn, 'rb'))
    # 	target_locs = [d[0] for d in data]
    # 	freq = int(len(target_locs) / prior_num)
    # 	print(len(target_locs))
    # 	target_locs = target_locs[::freq]
    # 	behaviors = [ee_distance_behavior(d[2])[0] for d in data]
    # 	print(len(behaviors))
    # 	behaviors = behaviors[::freq]
    # 	xs, ys, zs = zip(*target_locs)
    # 	sc = ax.scatter(xs=xs, ys=ys, zs=zs, c=behaviors)
    # 	print('scattered')

    if posterior_fn is not None:
        data = pickle.load(open(posterior_fn, 'rb'))
        target_locs = data['env']
        freq = int(len(target_locs) / posterior_num)
        target_locs = target_locs[::freq]
        behaviors = data['behavior'][::freq]
        xs, ys, zs = zip(*target_locs)
        sc = ax.scatter(xs=xs, ys=ys, zs=zs, c=behaviors, cmap='Reds_r')

    # vertical_wall = [0, 0.1, 0.825], [0.01, 0.8, 0.4]
    # horizontal_wall = [0, 0.1, 1.025], [0.7, 0.8, 0.01]
    # table_top = [0, 0, 0.6], [1.5, 1, 0.05]
    # plot_box(ax, *vertical_wall, **{'color': 'C2'})
    # plot_box(ax, *horizontal_wall, **{'color': 'C2'})
    # plot_box(ax, *table_top, **{'color': 'C2'})
    # plot_box(ax, [0.55/2, -0.05, 1.65/2], [0.45, 0.5, 0.35], **{'color': 'C1'})
    # plot_box(ax, [-0.55/2, -0.05, 1.65/2], [0.45, 0.5, 0.35], **{'color': 'C1'})
    plt.colorbar(sc)
    set_axes_equal(ax)
    plt.show()


# def main():
# 	# visualize_target_locs(prior_fn='rl_prior.pkl', posterior_fn='rl_min_ee_distance.samples.pkl')
# 	# visualize_target_locs(prior_fn='rrt_prior.pkl', posterior_fn='rrt_min_ee_distance_uniform.samples.pkl')

# 	# visualize_il()
# 	# visualize_rrt()
# 	# sample_rl_prior()
# 	# sample_rrt_prior()
# 	# visualize_trajectories(prior_fn='rl_prior.pkl', posterior_fn='rl_min_ee_distance.samples.pkl')
# 	# visualize_trajectories(prior_fn='rrt_prior.pkl', posterior_fn='rrt_min_ee_distance.samples.pkl')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] not in all_instances:
            raise Exception('The specified sampling instance is not found. \nAvailable instances are ' +
                            f'[{", ".join(all_instances.keys())}]')
        else:
            all_instances[sys.argv[1]]()
    elif len(sys.argv) == 1:
        print('Available sampling instances are: ')
        keys = list(all_instances.keys())
        for i, k in enumerate(keys):
            print(f'{i + 1}: {k}')
        idx = int(input('Please enter the index: ')) - 1
        if idx < 0 or idx >= len(all_instances):
            raise Exception('Invalid index. ')
        all_instances[keys[idx]]()
    else:
        raise Exception('At most one argument can be provided to specify the sampling instance name. ')
