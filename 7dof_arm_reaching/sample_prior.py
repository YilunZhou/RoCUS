import os, pickle
from tqdm import tqdm, trange
from environment import PandaEnv
from controller import DSController, RLController, RRTController
from kernel import TransitionKernel, RRTKernelNormal, ReachingEnvKernelNormal


def sample_prior(N, save_fn, env, controller, ek, ck):
	if os.path.isfile(save_fn):
		input('File already exists. Press Enter to append to the file. Press Ctrl-C to abort... ')
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


def main():
	idx = input('Please select the controller -- \n1a. Original Dynamical System\n1b. Improved Dynamical System\n' + 
				'2. Reinforcement Learning\n3. Rapidly-Exploring Random Tree\nEnter your choice: ')
	assert idx in ['1a', '1b', '2', '3'], 'Invalid input! The input needs to be 1a, 1b, 2, or 3. '
	env = PandaEnv()
	env_kernel = ReachingEnvKernelNormal()
	if idx == '1a':
		controller = DSController(typ='original')
		controller_kernel = TransitionKernel()
		save_fn = 'samples/ds_original_prior.pkl'
	elif idx == '1b':
		controller = DSController(typ='improved')
		controller_kernel = TransitionKernel()
		save_fn = 'samples/ds_improved_prior.pkl'
	elif idx == '2':
		controller = RLController()
		controller_kernel = TransitionKernel()
		save_fn = 'samples/rl_prior.pkl'
	else:
		controller = RRTController()
		controller_kernel = RRTKernelNormal()
		save_fn = 'samples/rrt_prior.pkl'
	sample_prior(2000, save_fn, env, controller, env_kernel, controller_kernel)

if __name__ == '__main__':
	main()
