import os, pickle
from tqdm import tqdm, trange
import numpy as np
from scipy.stats import norm, truncnorm


def get_sigma(alpha, prior_file, behavior_func, target_type, target_behavior=None, min_N=1000):
    assert target_type in ['match', 'maximal']
    data = pickle.load(open(prior_file, 'rb'))
    behaviors = []
    for ek_value, ck_value, traj in tqdm(data):
        behavior, acceptable = behavior_func(traj, ek_value)
        if not acceptable:
            continue
        behaviors.append(behavior)
    behaviors = np.array(behaviors)
    if len(behaviors) < min_N:
        raise Exception(f'Insufficient number of acceptable trajectories: {len(behaviors)}/{min_N}')
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


def sample(N, alpha, prior_file, N_sigma, behavior_func,
           env, env_kernel, controller, controller_kernel,
           target_type, target_behavior=None, save=None):
    if save is not None:
        assert isinstance(save, str), 'Parameter "save" needs to be a string if not None'
        if os.path.isfile(save):
            input(f'{save} already exists. Press Enter to overwrite it or press Ctrl-C to abort... ')

    def get_behavior(ek, ck):
        env.reset(ek.value)
        traj = controller.get_trajectory(env, ck)
        behav, accep = behavior_func(traj, env)
        return behav, accep, traj

    sigma, b_mean, b_std = get_sigma(alpha, prior_file, behavior_func, target_type, target_behavior, N_sigma)
    print(f'sigma: {sigma}, b_mean: {b_mean}, b_std: {b_std}')
    if target_type == 'match':
        likelihood = norm(loc=target_behavior, scale=sigma)
    elif target_type == 'maximal':
        likelihood = norm(loc=1, scale=sigma)

    def log_posterior(ekv, ckv, b):
        if target_type == 'match':
            assert b_mean == 0 and b_std == 1
            return env.log_prior(ekv) + controller.log_prior(ckv) + likelihood.logpdf(b)
        else:
            beta = 1 / (1 + np.exp(- (b - b_mean) / b_std))
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
    bar = trange(N)
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
    if save is not None:
        samples = {'env': env_samples, 'controller': controller_samples,
                   'trajectory': trajectories, 'behavior': behaviors}
        pickle.dump(samples, open(save, 'wb'))
    return env_samples, controller_samples, trajectories, behaviors


def visualize_prior_trajectories(fn):
    data = pickle.load(open(fn, 'rb'))
    plt.figure()
    for _, _, traj in data:
        if traj is None:
            continue
        plt.plot(traj[:, 0], traj[:, 1], 'C0', alpha=0.1)
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.gca().set_aspect('equal')
    plt.show()


def visualize_ds():
    env = RBF2dGym(time_limit=500, oob_termination=False)
    env.turn_off_lidar()
    ds_controller = DSController()
    env_kernel = RBF2dEnvKernelNormal()
    ds_kernel = TransitionKernel()
    while True:
        env_kernel.sample_prior()
        ds_kernel.sample_prior()
        env.reset(obs_override=env_kernel.value)
        traj = ds_controller.get_trajectory(env, ds_kernel)
        plt.imshow(env.env.env_img, origin='lower', extent=[-1.2, 1.2, -1.2, 1.2], cmap='coolwarm')
        plt.plot(traj[:, 0], traj[:, 1])
        plt.show()


def visualize_rrt():
    env = RBF2dGym()
    env.turn_off_lidar()
    rrt_controller = RRTController()
    env_kernel = RBF2dEnvKernelNormal()
    rrt_kernel = RRTKernelNormal([-1, -1], [1, 1])
    while True:
        env_kernel.sample_prior()
        rrt_kernel.sample_prior()
        env.reset(obs_override=env_kernel.value)
        traj = rrt_controller.get_trajectory(env, rrt_kernel)
        plt.imshow(env.env.env_img, origin='lower', extent=[-1.2, 1.2, -1.2, 1.2], cmap='coolwarm')
        plt.plot(traj[:, 0], traj[:, 1])
        plt.show()


def visualize_il():
    env = RBF2dGym()
    il_controller = ILController('imitation_learning/best.pt')
    env_kernel = RBF2dEnvKernelNormal()
    il_kernel = TransitionKernel()
    while True:
        env_kernel.sample_prior()
        il_kernel.sample_prior()
        env.reset(obs_override=env_kernel.value)
        traj = il_controller.get_trajectory(env, il_kernel)
        plt.imshow(env.env.env_img, origin='lower', extent=[-1.2, 1.2, -1.2, 1.2], cmap='coolwarm')
        plt.plot(traj[:, 0], traj[:, 1])
        plt.show()