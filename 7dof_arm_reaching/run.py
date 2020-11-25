import sys, inspect

import behavior
from environment import PandaEnv
from sampler import sample
from controller import DSController, RLController, RRTController
from kernel import TransitionKernel, ReachingEnvKernelNormal, RRTKernelNormal

if __name__ == '__main__':
    all_instances = dict()

def register(instance):
    if __name__ == '__main__':
        all_instances[instance.__name__] = instance

@register
def ds_original_min_end_distance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    samples = sample(N=10000, alpha=0.1, prior_file='samples/ds_original_prior.pkl', N_sigma=1000, 
                     behavior_func=behavior.ee_distance_behavior, 
                     env=PandaEnv(), env_kernel=ReachingEnvKernelNormal(),
                     controller=DSController(typ='original'), controller_kernel=TransitionKernel(),
                     target_type='match', target_behavior=0, save=name)

@register
def ds_improved_min_end_distance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    samples = sample(N=10000, alpha=0.1, prior_file='samples/ds_improved_prior.pkl', N_sigma=1000, 
                     behavior_func=behavior.ee_distance_behavior, 
                     env=PandaEnv(), env_kernel=ReachingEnvKernelNormal(),
                     controller=DSController(typ='improved'), controller_kernel=TransitionKernel(),
                     target_type='match', target_behavior=0, save=name)

@register
def rl_min_ee_distance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    samples = sample(N=10000, alpha=0.1, prior_file='samples/rl_prior.pkl', N_sigma=1000, 
                     behavior_func=behavior.ee_distance_behavior, 
                     env=PandaEnv(), env_kernel=ReachingEnvKernelNormal(),
                     controller=RLController(), controller_kernel=TransitionKernel(),
                     target_type='match', target_behavior=0, save=name)

@register
def rl_min_ee_distance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    samples = sample(N=10000, alpha=0.1, prior_file='samples/rl_prior.pkl', N_sigma=1000, 
                     behavior_func=behavior.ee_distance_behavior, 
                     env=PandaEnv(), env_kernel=ReachingEnvKernelNormal(),
                     controller=RLController(), controller_kernel=TransitionKernel(),
                     target_type='match', target_behavior=0, save=name)

@register
def rrt_min_ee_distance():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    samples = sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000, 
                     behavior_func=behavior.ee_distance_behavior, 
                     env=PandaEnv(), env_kernel=ReachingEnvKernelNormal(),
                     controller=RRTController(), controller_kernel=RRTKernelNormal(),
                     target_type='match', target_behavior=0, save=name)

@register
def rrt_max_illegibility():
    name = f'samples/{inspect.currentframe().f_code.co_name}.pkl'
    samples = sample(N=10000, alpha=0.1, prior_file='samples/rrt_prior.pkl', N_sigma=1000, 
                     behavior_func=behavior.illegibility_behavior, 
                     env=PandaEnv(), env_kernel=ReachingEnvKernelNormal(),
                     controller=RRTController(), controller_kernel=RRTKernelNormal(),
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
