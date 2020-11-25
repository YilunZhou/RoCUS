
import sys, inspect
import sampler, behavior, environment, kernel, controller

if __name__ == '__main__':
    all_instances = dict()

def register(instance):
    if __name__ == '__main__':
        all_instances[instance.__name__] = instance

@register
def ds_min_legibility():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.neg_behavior(behavior.legibility),
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def il_min_legibility():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.neg_behavior(behavior.legibility),
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def rrt_min_legibility():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.neg_behavior(behavior.legibility),
                   env=environment.RBF2dGymEnv(use_lidar=False), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.RRTController(), controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='maximal', save=name)

@register
def ds_min_avg_jerkiness():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.avg_jerkiness,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def il_min_avg_jerkiness():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.avg_jerkiness,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def rrt_min_avg_jerkiness():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   env=environment.RBF2dGymEnv(use_lidar=False), controller=controller.RRTController(),
                   behavior_func=behavior.avg_jerkiness, env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='match', target_behavior=0, save=name)

@register
def ds_min_abs_jerkiness():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.min_jerkiness,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def il_min_abs_jerkiness():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.min_jerkiness,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def rrt_min_abs_jerkiness():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   env=environment.RBF2dGymEnv(use_lidar=False), controller=controller.RRTController(),
                   behavior_func=behavior.min_jerkiness, env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='match', target_behavior=0, save=name)

@register
def ds_min_avg_obstacle_clearance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.obstacle_clearance,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def il_min_avg_obstacle_clearance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.obstacle_clearance,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def rrt_min_avg_obstacle_clearance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   env=environment.RBF2dGymEnv(use_lidar=False), controller=controller.RRTController(),
                   behavior_func=behavior.obstacle_clearance, env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='match', target_behavior=0, save=name)

@register
def ds_max_avg_obstacle_clearance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.obstacle_clearance,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def il_max_avg_obstacle_clearance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.obstacle_clearance,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def rrt_max_avg_obstacle_clearance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.obstacle_clearance,
                   env=environment.RBF2dGymEnv(use_lidar=False), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.RRTController(), controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='maximal', save=name)

@register
def ds_min_straightline_deviation():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.straightline_deviation,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def il_min_straightline_deviation():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.straightline_deviation,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def rrt_min_straightline_deviation():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   env=environment.RBF2dGymEnv(use_lidar=False), controller=controller.RRTController(),
                   behavior_func=behavior.straightline_deviation, env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='match', target_behavior=0, save=name)

@register
def ds_max_end_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.end_distance,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def il_max_end_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.end_distance,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def rrt_max_end_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.end_distance,
                   env=environment.RBF2dGymEnv(use_lidar=False), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.RRTController(), controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='maximal', save=name)

@register
def ds_min_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.distance,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def il_min_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.distance,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='match', target_behavior=0, save=name)

@register
def rrt_min_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   env=environment.RBF2dGymEnv(use_lidar=False), controller=controller.RRTController(),
                   behavior_func=behavior.distance, env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='match', target_behavior=0, save=name)

@register
def ds_max_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/ds_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.distance,
                   env=environment.RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False),
                   env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.DSController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def il_max_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/il_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.distance,
                   env=environment.RBF2dGymEnv(use_lidar=True), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.ILController(), controller_kernel=kernel.TransitionKernel(),
                   target_type='maximal', save=name)

@register
def rrt_max_dist():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    sampler.sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000,
                   behavior_func=behavior.distance,
                   env=environment.RBF2dGymEnv(use_lidar=False), env_kernel=kernel.RBF2dEnvKernelNormal(),
                   controller=controller.RRTController(), controller_kernel=kernel.RRTKernelNormal([-1, -1], [1, 1]),
                   target_type='maximal', save=name)

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
