
import numpy as np
from scipy.spatial.distance import cdist

def neg_behavior(b_func):
	'''A second order function that inverts a given behavior function'''
	def neg_b_func(arg1, arg2):
		val, acceptable = b_func(arg1, arg2)
		if acceptable:
			val = - val
		return val, acceptable
	return neg_b_func

def obstacle_clearance(traj, env):
	grid = env.arena.occ_grid
	res = env.arena.grid_res
	bd = env.arena.grid_bd
	xs, ys = np.meshgrid(np.linspace(-bd, bd, res), np.linspace(-bd, bd, res))
	xys = np.stack([xs.flatten(), ys.flatten()], axis=1)
	obs_masks = (grid.flatten() == 1)
	obs_xys = xys[obs_masks]
	pairwise_dist = cdist(obs_xys, traj)
	min_dists = pairwise_dist.min(axis=0)
	assert len(min_dists) == len(traj)
	return min_dists.mean()

def smoothness(traj, env=None):
	if traj is None or len(traj) < 2 or np.linalg.norm(traj[-1] - [1, 1]) > 0.03:
		return None, False
	angles = []
	cos_sims = [np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2) for v1, v2 in zip(traj[1:], traj[:-1])]
	return np.mean(cos_sims), True

def avg_jerkiness(traj, env=None):
	if traj is None or len(traj) < 10 or np.linalg.norm(traj[-1] - [1, 1]) > 0.03:
		return None, False
	velocity = np.gradient(traj, axis=0)
	acceleration = np.gradient(velocity, axis=0)
	jerk = np.absolute(np.gradient(acceleration, axis=0)).sum(axis=1)
	return np.mean(jerk), True

def min_jerkiness(traj, env=None):
	if traj is None or len(traj) < 10 or np.linalg.norm(traj[-1] - [1, 1]) > 0.03:
		return None, False
	velocity = np.gradient(traj, axis=0)
	acceleration = np.gradient(velocity, axis=0)
	jerk = np.absolute(np.gradient(acceleration, axis=0)).sum(axis=1)
	return np.min(jerk), True

def distance(traj, env=None):
	if traj is None or len(traj) < 3 or np.linalg.norm(traj[-1] - [1, 1]) > 0.03:
		return None, False
	traj = np.array(traj)
	return np.linalg.norm(traj[1:] - traj[:-1], axis=1).sum(), True
	
def end_distance(traj, env=None):
	if traj is None:
		return None, False
	return np.linalg.norm(traj[-1] - [1, 1]), True

def straightline_deviation(traj, env=None):
	if traj is None or np.linalg.norm(traj[-1] - [1, 1]) > 0.03:
		return None, False
	u = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2])
	d = np.dot(traj, u)
	projs = np.stack((d, d), axis=1) * u
	avg_dist = (np.linalg.norm(projs - traj, axis=1)).mean()
	return avg_dist, True

def legibility(traj, env=None):
	if traj is None or np.linalg.norm(traj[-1] - [1, 1]) > 0.03:
		return None, False
	target_dir = [1, 1] - traj[:-1]
	control_dir = traj[1:] - traj[:-1]
	cos_sim = [np.dot(t, c) / (np.linalg.norm(t) * np.linalg.norm(c)) for t, c in zip(target_dir, control_dir)]
	avg_sim = np.mean(cos_sim)
	return avg_sim, True
