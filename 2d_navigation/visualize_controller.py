import numpy as np
import matplotlib.pyplot as plt

from environment import RBF2dGymEnv
from controller import DSController, ILController, RRTController
from kernel import TransitionKernel, RRTKernelNormal, RBF2dEnvKernelNormal


def main():
    idx = input('Please select the controller type: \n1. Dynamical System\n2. Imitation Learning\n3. Rapidly-Exploring Random Tree\n' + 
                'Enter a number between 1 and 3: ')
    try:
        idx = int(idx)
        assert 1 <= idx <= 3
    except:
        print('Invalid input! The input needs to be a number between 1 and 3. ')
        quit()
    if idx == 1:
        env = RBF2dGymEnv(time_limit=500, oob_termination=False, use_lidar=False)
        controller = DSController()
        controller_kernel = TransitionKernel()
    elif idx == 2:
        env = RBF2dGymEnv(use_lidar=True)
        controller = ILController()
        controller_kernel = TransitionKernel()
    else:
        env = RBF2dGymEnv(use_lidar=False)
        controller = RRTController()
        controller_kernel = RRTKernelNormal([-1, -1], [1, 1])
    env_kernel = RBF2dEnvKernelNormal()
    while True:
        env_kernel.sample_prior()
        controller_kernel.sample_prior()
        env.reset(env_kernel.value)
        traj = controller.get_trajectory(env, controller_kernel)
        plt.imshow(env.arena.occ_grid, origin='lower', extent=[-1.2, 1.2, -1.2, 1.2],
                   cmap='gray_r', vmin=0, vmax=2)
        plt.plot(traj[:, 0], traj[:, 1])
        plt.title('Close this window to view next obstacle configuration. ')
        plt.show()

if __name__ == '__main__':
    main()
