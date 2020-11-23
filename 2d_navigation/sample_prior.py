import os
import pickle
from tqdm import tqdm, trange
from environment import RBF2dGymEnv
from controller import DSController, ILController, RRTController
from kernel import TransitionKernel, RBF2dEnvKernelNormal, RRTKernelNormal


def sample_prior(N, save_fn, env, controller, ek, ck):
    if os.path.isfile(save_fn):
        input('File already exists. Press Enter to append to the file. Press Ctrl-C to abort... ')
        data = pickle.load(open(save_fn, 'rb'))
    else:
        data = []
    for _ in trange(N):
        ek.sample_prior()
        ck.sample_prior()
        env.reset(ek.value)
        traj = controller.get_trajectory(env, ck)
        data.append((ek.value, ck.value, traj))
    pickle.dump(data, open(save_fn, 'wb'))


def main():
    idx = input('Please select the controller -- \n1. Dynamical System\n2. Imitation Learning\n3. Rapidly-Exploring Random Tree\n' + 
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
        save_fn = 'samples/ds_prior.pkl'
    elif idx == 2:
        env = RBF2dGymEnv(use_lidar=True)
        controller = ILController()
        controller_kernel = TransitionKernel()
        save_fn = 'samples/il_prior.pkl'
    else:
        env = RBF2dGymEnv(use_lidar=False)
        controller = RRTController()
        controller_kernel = RRTKernelNormal([-1, -1], [1, 1])
        save_fn = 'samples/rrt_prior.pkl'
    env_kernel = RBF2dEnvKernelNormal()
    sample_prior(2000, save_fn, env, controller, env_kernel, controller_kernel)

if __name__ == '__main__':
    main()
