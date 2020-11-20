import sys, inspect
from sampler import *
from behavior import *
from kernel import *

if __name__ == '__main__':
    all_instances = dict()


def register(instance):
    if __name__ == '__main__':
        all_instances[instance.__name__] = instance


@register
def rl_min_avg_jerk(prior_file='samples/rl_prior.pkl'):
    print('rl_min_avg_jerk')
    env = PandaReacherEarlyTerminateEnv()
    rl_controller = RLController()
    env_kernel = ReachingEnvKernelNormal()
    rl_kernel = TransitionKernel()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rl_controller,
                     behavior_func=ee_avg_jerk, env_kernel=env_kernel, controller_kernel=rl_kernel,
                     target_type='match', target_behavior=0, N_sigma=1000, sigma_override=None)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rl_min_avg_jerk.samples.pkl', 'wb'))


@register
def rl_min_center_deviation(prior_file='samples/rl_prior.pkl'):
    print('rl_min_center_deviation')
    env = PandaReacherEarlyTerminateEnv()
    rl_controller = RLController()
    env_kernel = ReachingEnvKernelNormal()
    rl_kernel = TransitionKernel()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rl_controller,
                     behavior_func=center_deviation_behavior, env_kernel=env_kernel, controller_kernel=rl_kernel,
                     target_type='match', target_behavior=0, N_sigma=1000, sigma_override=None)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rl_min_center_deviation.samples.pkl', 'wb'))


@register
def rrt_min_center_deviation(prior_file='samples/rrt_prior.pkl'):
    print('rrt_min_center_deviation')

    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelNormal()
    rrt_kernel = RRTKernelNormal()

    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rrt_controller,
                     behavior_func=center_deviation_behavior, env_kernel=env_kernel, controller_kernel=rrt_kernel,
                     target_type='match', target_behavior=0, N_sigma=1000)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rrt_min_center_deviation.samples.pkl', 'wb'))


@register
def rrt_min_avg_jerk(prior_file='samples/rrt_prior.pkl'):
    print('rrt_min_avg_jerk')
    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelNormal()
    rrt_kernel = RRTKernelNormal()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rrt_controller,
                     behavior_func=ee_avg_jerk, env_kernel=env_kernel, controller_kernel=rrt_kernel,
                     target_type='match', target_behavior=0, N_sigma=1000)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rrt_min_avg_jerk.samples.pkl', 'wb'))


@register
def rl_min_ee_distance(prior_file='samples/rl_prior.pkl'):
    print('rrt_min_ee_distance')
    env = PandaReacherEarlyTerminateEnv()
    rl_controller = RLController()
    env_kernel = ReachingEnvKernelNormal()
    rl_kernel = TransitionKernel()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rl_controller,
                     behavior_func=ee_distance_behavior, env_kernel=env_kernel, controller_kernel=rl_kernel,
                     target_type='match', target_behavior=0, N_sigma=1000, sigma_override=None)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rl_min_ee_distance.samples.pkl', 'wb'))


@register
def rrt_min_ee_distance(prior_file='samples/rrt_prior.pkl'):
    print('rrt_min_ee_distance')
    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelNormal()
    rrt_kernel = RRTKernelNormal()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rrt_controller,
                     behavior_func=ee_distance_behavior, env_kernel=env_kernel, controller_kernel=rrt_kernel,
                     target_type='match', target_behavior=0, N_sigma=1000)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rrt_min_ee_distance.samples.pkl', 'wb'))


@register
def rrt_min_ee_distance_uniform(prior_file='samples/rrt_prior.pkl'):
    print('rrt_min_ee_distance_uniform')
    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelUniform()
    rrt_kernel = RRTKernelNormal()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rrt_controller,
                     behavior_func=ee_distance_behavior, env_kernel=env_kernel, controller_kernel=rrt_kernel,
                     target_type='match', target_behavior=0, N_sigma=1000)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rrt_min_ee_distance_uniform.samples.pkl', 'wb'))


@register
def rl_max_illegibility(prior_file='samples/rl_prior.pkl'):
    print('rl_max_illegibility')
    env = PandaReacherEarlyTerminateEnv()
    rl_controller = RLController()
    env_kernel = ReachingEnvKernelNormal()
    rl_kernel = TransitionKernel()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rl_controller,
                     behavior_func=illegibility_behavior, env_kernel=env_kernel, controller_kernel=rl_kernel,
                     target_type='maximal', N_sigma=1000, sigma_override=None)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rl_max_illegibility.samples.pkl', 'wb'))


@register
def rrt_max_illegibility(prior_file='samples/rrt_prior.pkl'):
    print('rrt_max_illegibility')
    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelNormal()
    rrt_kernel = RRTKernelNormal()
    samples = sample(N=10000, alpha=0.99, prior_file=prior_file, env=env, controller=rrt_controller,
                     behavior_func=illegibility_behavior, env_kernel=env_kernel, controller_kernel=rrt_kernel,
                     target_type='maximal', N_sigma=1000, sigma_override=None)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rrt_max_illegibility.samples.pkl', 'wb'))


@register
def rrt_max_illegibility_uniform(prior_file='samples/rrt_prior.pkl'):
    print('rrt_max_illegibility')
    env = PandaReacherEarlyTerminateEnv()
    rrt_controller = RRTController()
    env_kernel = ReachingEnvKernelUniform()
    rrt_kernel = RRTKernelNormal()
    samples = sample(N=10000, alpha=0.1, prior_file=prior_file, env=env, controller=rrt_controller,
                     behavior_func=illegibility_behavior, env_kernel=env_kernel, controller_kernel=rrt_kernel,
                     target_type='maximal', N_sigma=1000, sigma_override=None)
    env_samples, controller_samples, trajectories, behaviors = samples
    pickle.dump(
        {'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors},
        open('samples/rrt_max_illegibility_uniform.samples.pkl', 'wb'))

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
