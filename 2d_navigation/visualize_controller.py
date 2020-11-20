import numpy as np
import matplotlib.pyplot as plt

from environment import RBF2dGymEnv
from controller import DSController, ILController, RRTController
from kernel import TransitionKernel, RRTKernelNormal


def main():
    idx = input('Please choose the controller that you want to sample -- 1. DS, 2. IL, 3. RRT: ')
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

    while True:
        env.reset(np.random.uniform(low=-0.7, high=0.7, size=(15, 2)))
        traj = controller.get_trajectory(env, controller_kernel)
        plt.imshow(env.arena.occ_grid, origin='lower', extent=[-1.2, 1.2, -1.2, 1.2],
                   cmap='gray_r', vmin=0, vmax=2)
        plt.plot(traj[:, 0], traj[:, 1])
        plt.title('Close this window to view next obstacle configuration. ')
        plt.show()


if __name__ == '__main__':
    main()
