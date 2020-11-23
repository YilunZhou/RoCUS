
import sys, traceback
import numpy as np
import torch

from dynamical_system.modulation import Modulator
from imitation_learning.model import BinClassifier
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

class DSController(Controller):
	def __init__(self):
		super().__init__()
		self.modulator = Modulator()
		self.warned = False

	def get_trajectory(self, env, kernel=None):
		assert env.oob_termination is False, 'early termination needs to be disabled'
		assert env.time_limit >= 500, 'time limit should be greater than or equal to 500'
		if env.enable_lidar is True and not self.warned:
			print('WARNING: Not turning off lidar on env could be significantly slower')
			self.warned = True
		try:
			self.modulator.set_arena(env.arena)
			epsilon = sys.float_info.epsilon
			done = False
			s = env.s()[:2]
			traj = [s]
			while not done:
				d = self.modulator.modulate(s)
				d = d / max([0.3, d[0] + epsilon, d[1] + epsilon])
				s, _, done, _ = env.step(d)
				s = s[:2]
				traj.append(s)
			traj = np.array(traj)
			return traj
		except KeyboardInterrupt:
			raise
		except:
			traceback.print_exc()
			return None

	def log_prior(self, kernel):
		return 0

class ILController(Controller):
	def __init__(self, model_fn='imitation_learning/data_and_model/best.pt', device=None):
		super().__init__()
		if device is None:
			device = ['cpu', 'cuda'][torch.cuda.is_available()]
		model_data = torch.load(model_fn)
		self.MEAN = model_data['MEAN']
		self.STD = model_data['STD']
		self.N_BINS = model_data['N_BINS']
		self.policy = BinClassifier(18, self.N_BINS)
		model_data = torch.load(model_fn, map_location='cpu')
		self.policy.load_state_dict(model_data['model'])
		self.BIN_RES = 2 * np.pi / self.N_BINS
		self.BIN_LOW = - np.pi
		self.device = device
		self.policy.to(self.device)

	def get_trajectory(self, env, kernel=None):
		try:
			traj = []
			with torch.no_grad():
				done = False
				s = env.s()
				traj.append(s[:2])
				while not done:
					s_scaled = torch.tensor((s - self.MEAN) / self.STD).float().to(self.device)
					bin_idx = self.policy(s_scaled).cpu().numpy().argmax()
					angle = bin_idx * self.BIN_RES - self.BIN_LOW
					dx = np.cos(angle)
					dy = np.sin(angle)
					s, _, done, _ = env.step([dx, dy])
					traj.append(s[:2])
			return np.array(traj)
		except KeyboardInterrupt:
			raise
		except:
			traceback.print_exc()
			return None

	def to(self, device):
		self.device = device
		self.policy.to(device)

	def log_prior(self, kernel):
		return 0

class RRTController(Controller):
	def __init__(self, control_res=0.03, collision_res=0.01):
		super().__init__()
		self.control_res = control_res
		self.rrt = RRT(control_res, collision_res)
		self.warned = False

	def get_trajectory(self, env, kernel):
		try:
			if env.enable_lidar is True and not self.warned:
				print('WARNING: Not turning off lidar on env could be significantly slower')
				self.warned = True
			xys = self.rrt.get_path(env.arena, kernel)
			if xys is None:
				return None
			xys = self.rrt.increase_resolution(xys)
			s = env.s()
			traj = [s[:2]]
			idx = 1
			done = False
			while not done:
				dx, dy = xys[idx] - s[:2]
				assert np.linalg.norm([dx, dy]) < self.control_res
				s, _, done, _ = env.step(np.array([dx, dy]) / self.control_res)
				traj.append(s[:2])
				idx += 1
				idx = min(idx, len(xys) - 1)
			return np.array(traj)
		except KeyboardInterrupt:
			raise
		except:
			traceback.print_exc()
			return None

	def log_prior(self, kernel):
		return 0
