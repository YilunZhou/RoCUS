
import pickle, os, time
from copy import deepcopy as copy
from tqdm import tqdm, trange

import numpy as np
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from klampt_controller import PandaReacherEarlyTerminateEnv, DSController

class TransitionKernel():
	'''
	A transition kernel on a random variable (or a set of RVs) stores the current value of the RV, 
	propose() will propose a new RV by setting the value attribute, and return forward and backward transition log probability. 
	revert() will revert the proposed value. revert can only be done once after a proposal. 
	sample_prior will reset the current value to one sampled from the prior, and erase prev_value to None since the chain is broken. 
	'''
	def __init__(self):
		self.sample_prior()
	def propose(self):
		self.prev_value = self.value
		self.value = 0
		return 0, 0
	def revert(self):
		assert self.prev_value is not None, 'no previous value available'
		self.value = self.prev_value
		self.prev_value = None
	def sample_prior(self):
		self.value = 0
		self.prev_value = None

class ReachingEnvKernelUniform(TransitionKernel):
	def __init__(self, left_lower=[-0.5, -0.3, 0.65], left_upper=[-0.05, 0.2, 1.0], 
					   right_lower=[0.05, -0.3, 0.65], right_upper=[0.5, 0.2, 1.0]):
		self.left_lower = left_lower
		self.left_upper = left_upper
		self.right_lower = right_lower
		self.right_upper = right_upper
		super(ReachingEnvKernelUniform, self).__init__()

	def propose(self):
		self.prev_value = self.value
		use_left = np.random.random() < 0.5
		if use_left:
			self.value = np.random.uniform(low=self.left_lower, high=self.left_upper)
		else:
			self.value = np.random.uniform(low=self.right_lower, high=self.right_upper)
		return 0, 0

	def sample_prior(self):
		use_left = np.random.random() < 0.5
		if use_left:
			self.value = np.random.uniform(low=self.left_lower, high=self.left_upper)
		else:
			self.value = np.random.uniform(low=self.right_lower, high=self.right_upper)
		self.prev_value = None

def truncnorm_rvs(a, b, mean, std):
	a_use = (a - mean) / std
	b_use = (b - mean) / std
	return truncnorm.rvs(a_use, b_use, mean, std)

def truncnorm_logpdf(x, a, b, mean, std):
	a_use = (a - mean) / std
	b_use = (b - mean) / std
	return truncnorm.logpdf(x, a_use, b_use, mean, std)

class ReachingEnvKernelNormal(TransitionKernel):
	def __init__(self, left_lower=[-0.5, -0.3, 0.65], left_upper=[-0.05, 0.2, 1.0], 
					   right_lower=[0.05, -0.3, 0.65], right_upper=[0.5, 0.2, 1.0], 
					   sigma_x=0.1, sigma_y=0.05, sigma_z=0.035):
		assert left_lower[1] == right_lower[1] and left_lower[2] == right_lower[2]
		assert left_upper[1] == right_upper[1] and left_upper[2] == right_upper[2]
		self.left_lower = left_lower
		self.left_upper = left_upper
		self.right_lower = right_lower
		self.right_upper = right_upper
		self.sigma_x = sigma_x
		self.sigma_y = sigma_y
		self.sigma_z = sigma_z
		super(ReachingEnvKernelNormal, self).__init__()

	def propose(self):
		self.prev_value = self.value
		cur_x, cur_y, cur_z = self.value
		total_forward_log_prob = 0
		total_backward_log_prob = 0

		left_total_prob = norm.cdf(self.left_upper[0], loc=cur_x, scale=self.sigma_x) - norm.cdf(self.left_lower[0], loc=cur_x, scale=self.sigma_x)
		right_total_prob = norm.cdf(self.right_upper[0], loc=cur_x, scale=self.sigma_x) - norm.cdf(self.right_lower[0], loc=cur_x, scale=self.sigma_x)
		left_ratio = left_total_prob / (left_total_prob + right_total_prob)
		if np.random.random() < left_ratio:
			prop_x = truncnorm_rvs(self.left_lower[0], self.left_upper[0], cur_x, self.sigma_x)
			total_forward_log_prob += np.log(left_ratio) + truncnorm_logpdf(prop_x, self.left_lower[0], self.left_upper[0], cur_x, self.sigma_x)
		else:
			prop_x = truncnorm_rvs(self.right_lower[0], self.right_upper[0], cur_x, self.sigma_x)
			total_forward_log_prob += np.log(1 - left_ratio) + truncnorm_logpdf(prop_x, self.right_lower[0], self.right_upper[0], cur_x, self.sigma_x)

		back_left_total_prob = norm.cdf(self.left_upper[0], loc=prop_x, scale=self.sigma_x) - norm.cdf(self.left_lower[0], loc=prop_x, scale=self.sigma_x)
		back_right_total_prob = norm.cdf(self.right_upper[0], loc=prop_x, scale=self.sigma_x) - norm.cdf(self.right_lower[0], loc=prop_x, scale=self.sigma_x)
		back_left_ratio = back_left_total_prob / (back_left_total_prob + back_right_total_prob)

		assert self.left_lower[0] <= cur_x <= self.left_upper[0] or self.right_lower[0] <= cur_x <= self.right_upper[0]
		if self.left_lower[0] <= cur_x <= self.left_upper[0]:
			total_backward_log_prob += np.log(back_left_ratio) + truncnorm_logpdf(cur_x, self.left_lower[0], self.left_upper[0], prop_x, self.sigma_x)
		else:
			total_backward_log_prob += np.log(1 - back_left_ratio) + truncnorm_logpdf(cur_x, self.right_lower[0], self.right_upper[0], prop_x, self.sigma_x)

		prop_y = truncnorm_rvs(self.left_lower[1], self.left_upper[1], cur_y, self.sigma_y)
		total_forward_log_prob += truncnorm_logpdf(prop_y, self.left_lower[1], self.left_upper[1], cur_y, self.sigma_y)
		total_backward_log_prob += truncnorm_logpdf(cur_y, self.left_lower[1], self.left_upper[1], prop_y, self.sigma_y)
		
		prop_z = truncnorm_rvs(self.left_lower[2], self.left_upper[2], cur_z, self.sigma_z)
		total_forward_log_prob += truncnorm_logpdf(prop_z, self.left_lower[2], self.left_upper[2], cur_z, self.sigma_z)
		total_backward_log_prob += truncnorm_logpdf(cur_z, self.left_lower[2], self.left_upper[2], prop_z, self.sigma_z)
		
		self.value = [prop_x, prop_y, prop_z]
		return total_forward_log_prob, total_backward_log_prob

	def sample_prior(self):
		use_left = np.random.random() < 0.5
		if use_left:
			self.value = np.random.uniform(low=self.left_lower, high=self.left_upper)
		else:
			self.value = np.random.uniform(low=self.right_lower, high=self.right_upper)
		self.prev_value = None

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
		bar.set_description(f'%Acc: {tot_acc/(i+1):0.2f}')
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

def sample_ds_prior(N=2000, save_fn='ds_prior.pkl'):
	env = PandaReacherEarlyTerminateEnv()
	ds_controller = DSController()
	env_kernel = ReachingEnvKernelNormal()
	ds_kernel = TransitionKernel()
	env_traj_prior_samples(N, save_fn, env, ds_controller, env_kernel, ds_kernel)

def ee_distance_behavior(traj, env=None):
	if traj is None:
		return None, False
	ee_xyz = traj[:, 7:10]
	ee_dxyz = ee_xyz[1:] - ee_xyz[:-1]
	dist = np.linalg.norm(ee_dxyz, axis=1).sum()
	return dist, True

def ee_avg_jerk(traj, env=None):
	if traj is None:
		return None, False
	ee_xyz = traj[:, 7:10]
	first_derivative = np.gradient(ee_xyz, axis=0)
	second_derivative = np.gradient(first_derivative, axis=0)
	third_derivative = np.absolute(np.gradient(second_derivative, axis=0)).sum(axis=1)
	return np.mean(third_derivative), True


def center_deviation_behavior(traj, env=None):
	if traj is None:
		return None, False
	ee_xyz = traj[:, 7:10]
	u = np.array([np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3)/3])
	d = np.dot(ee_xyz, u)
	projs = np.stack((d, d, d), axis=1) * u
	avg_dist = (np.linalg.norm(projs - ee_xyz, axis=1)).mean()
	return avg_dist, True


def ds_min_ee_distance():
	print('ds_min_ee_distance')
	env = PandaReacherEarlyTerminateEnv()
	ds_controller = DSController()
	env_kernel = ReachingEnvKernelNormal()
	ds_kernel = TransitionKernel()
	samples = sample(N=10000, alpha=0.1, prior_file='ds_prior.pkl', env=env, controller=ds_controller, 
					 behavior_func=ee_distance_behavior, env_kernel=env_kernel, controller_kernel=ds_kernel, 
					 target_type='match', target_behavior=0, N_sigma=1000, sigma_override=None)
	env_samples, controller_samples, trajectories, behaviors = samples
	pickle.dump({'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors}, 
				open('ds_min_ee_distance.samples.pkl', 'wb'))

def ds_min_avg_jerk():
	print('ds_min_avg_jerk')
	env = PandaReacherEarlyTerminateEnv()
	ds_controller = DSController()
	env_kernel = ReachingEnvKernelNormal()
	ds_kernel = TransitionKernel()
	samples = sample(N=10000, alpha=0.1, prior_file='ds_prior.pkl', env=env, controller=ds_controller, 
					 behavior_func=ee_avg_jerk, env_kernel=env_kernel, controller_kernel=ds_kernel, 
					 target_type='match', target_behavior=0, N_sigma=1000, sigma_override=None)
	env_samples, controller_samples, trajectories, behaviors = samples
	pickle.dump({'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors}, 
				open('ds_min_ee_avg_jerk.samples.pkl', 'wb'))


def ds_min_center_deviation():
	print('ds_min_center_deviation')
	env = PandaReacherEarlyTerminateEnv()
	ds_controller = DSController()
	env_kernel = ReachingEnvKernelNormal()
	ds_kernel = TransitionKernel()
	samples = sample(N=10000, alpha=0.1, prior_file='ds_prior.pkl', env=env, controller=ds_controller, 
					 behavior_func=center_deviation_behavior, env_kernel=env_kernel, controller_kernel=ds_kernel, 
					 target_type='match', target_behavior=0, N_sigma=1000, sigma_override=None)
	env_samples, controller_samples, trajectories, behaviors = samples
	pickle.dump({'env': env_samples, 'controller': controller_samples, 'trajectory': trajectories, 'behavior': behaviors}, 
				open('ds_min_ee_center_deviation.samples.pkl', 'wb'))

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
	plot_radius = 0.5*max([x_range, y_range, z_range])

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
	plot_box(ax, [0.55/2, -0.05, 1.65/2], [0.45, 0.5, 0.35], **{'color': 'C1'})
	plot_box(ax, [-0.55/2, -0.05, 1.65/2], [0.45, 0.5, 0.35], **{'color': 'C1'})
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

def main():
	# sample_ds_prior()
	# time.sleep(1)
	# ds_min_ee_distance()
	# time.sleep(1)
	# ds_min_avg_jerk()
	# time.sleep(1)
	ds_min_center_deviation()

if __name__ == '__main__':
	main()

