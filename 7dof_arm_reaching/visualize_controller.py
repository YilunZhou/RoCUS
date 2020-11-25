import numpy as np
import matplotlib.pyplot as plt

from environment import PandaEnv
from controller import DSController, RLController, RRTController
from kernel import TransitionKernel, RRTKernelNormal, ReachingEnvKernelNormal

def main():
    idx = input('Please select the controller -- \n1a. Original Dynamical System\n1b. Improved Dynamical System\n' + 
                '2.  Reinforcement Learning\n3.  Rapidly-Exploring Random Tree\nEnter your choice: ')
    assert idx in ['1a', '1b', '2', '3'], 'Invalid input! The input needs to be 1a, 1b, 2, or 3. '
    env = PandaEnv()
    env.render()
    env_kernel = ReachingEnvKernelNormal()
    if idx == '1a':
        controller = DSController(typ='original', visualize=True)
        controller_kernel = TransitionKernel()
    elif idx == '1b':
        controller = DSController(typ='improved', visualize=True)
        controller_kernel = TransitionKernel()
    elif idx == '2':
        controller = RLController(visualize=True)
        controller_kernel = TransitionKernel()
    else:
        controller = RRTController(visualize=True)
        controller_kernel = RRTKernelNormal()
    while True:
        env_kernel.sample_prior()
        controller_kernel.sample_prior()
        env.reset(target_loc=env_kernel.value)
        controller.get_trajectory(env, controller_kernel)

if __name__ == '__main__':
    main()
