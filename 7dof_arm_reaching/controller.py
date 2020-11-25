
import sys, time, traceback

import numpy as np
import torch

from dynamical_system.modulation import Modulator
from dynamical_system.ik import IK
from reinforcement_learning.ppo import ActorCritic
from rapidly_exploring_random_tree.rrt import RRT

class Controller():
	'''
	This is an abstract super-class for all controllers. At the very least, 
	get_trajectory() and log_prior() need to be implemented. 
	'''
	def __init__(self):
		pass
	def get_trajectory(self, env, kernel):
		'''
		given an environment (i.e. a specific task instantiation), roll out the 
		controller with randomness specified in the kernel and return the 
		trajectory of shape T x 2. 
		'''
		raise NotImplementedError
	def log_prior(self, kernel):
		'''
		return the prior probability of the controller kernel, for stochastic controllers. 
		return 0 for deterministic controllers. 
		'''
		raise NotImplementedError


class DSController():
	def __init__(self, typ, svm_fn='dynamical_system/svm_model.pkl', visualize=False):
		self.modulator = Modulator(typ, svm_fn)
		self.ik = IK()
		self.visualize = visualize

	def get_trajectory(self, env, kernel=None):
		try:
			s = env.s()
			traj = [s[:10]]
			done = False
			while not done:
				ee_loc = s[7:10]
				target_loc = s[16:19]
				x_dot = self.modulator.get_modulated_direction(ee_loc, target_loc)
				x_dot = x_dot / np.linalg.norm(x_dot + sys.float_info.epsilon) * 0.10
				next_ee_loc = ee_loc + x_dot * 0.05
				cur_cfg = s[:7]
				next_cfg = self.ik(cur_cfg, next_ee_loc)
				cfg_diff = np.array(next_cfg) - cur_cfg
				cfg_diff = cfg_diff / max(cfg_diff.max(), 0.05)
				s, _, done, _ = env.step(cfg_diff)
				traj.append(s[:10])
				if self.visualize:
					time.sleep(0.03)
			return np.array(traj).astype('float32')
		except:
			raise
			traceback.print_exc()
			return None

	def log_prior(self, kernel):
		return 0

class RLController(Controller):
	def __init__(self, model_fn='reinforcement_learning/best.pt', device=None, visualize=False):
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		ckpt = torch.load(model_fn, map_location=device)
		activation = ckpt['config'].activation
		tanh_end = ckpt['config'].tanh_end
		self.actor_critic = ActorCritic(22, 7, activation=activation, tanh_end=tanh_end).to(device)
		self.actor_critic.load_state_dict(ckpt['actor-critic'])
		self.device = device
		self.visualize = visualize

	def get_trajectory(self, env, kernel=None):
		try:
			s = env.s()
			done = False
			traj = [s[:10]]
			while not done:
				a = self.actor_critic.act(s, train=False)
				s, _, done, _ = env.step(a)
				traj.append(s[:10])
				if self.visualize:
					time.sleep(0.03)
			return np.array(traj).astype('float32')
		except KeyboardInterrupt:
			raise
		except:
			traceback.print_exc()

	def to(self, device):
		self.device = device
		self.actor_critic.to(device)

	def log_prior(self, kernel):
		return 0

class RRTController(Controller):
	def __init__(self, control_res=0.05, collision_res=0.01, visualize=False):
		self.rrt = RRT(control_res, collision_res)
		self.visualize = visualize

	def get_trajectory(self, env, kernel):
		try:
			s = env.s()
			target_loc = s[16:19]
			planned_traj = self.rrt.get_path(target_loc, kernel)
			if planned_traj is None:
				return None
			planned_traj = self.rrt.increase_resolution(planned_traj)
			actual_traj = [s[:10]]
			done = False
			t = 1
			while not done:
				a = (planned_traj[t] - s[:7]) / 0.05
				s, _, done, _ = env.step(a)
				actual_traj.append(s[:10])
				if t < len(planned_traj) - 1:
					t += 1
				if self.visualize:
					time.sleep(0.03)
			return np.array(actual_traj).astype('float32')
		except KeyboardInterrupt:
			raise
		except:
			traceback.print_exc()
			return None

	def log_prior(self, kernel):
		return 0