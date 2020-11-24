
import numpy as np
from pybulletgym_rocus.envs.roboschool.envs.manipulation.panda_reacher_env import PandaReacherEnv

class PandaEnv(PandaReacherEnv):
	'''
	a modified environment that runs for a maximum of 500 time steps, but terminates as soon as the target is reached. 
	also supports calculating the log probability of a particular environment configuration (i.e. target location) under the prior. 
	'''
	def __init__(self, shelf=True, timelimit=500, target_threshold=0.03):
		super(PandaEnv, self).__init__(shelf=shelf)
		self.timelimit = timelimit
		self.target_threshold = target_threshold

	def reset(self, **kwargs):
		self.cur_time = 0
		super().reset(**kwargs)

	def step(self, a):
		s, r, done, info = super().step(a)
		self.cur_time += 1
		done = done or self.cur_time == self.timelimit or np.linalg.norm(s[19:22]) <= self.target_threshold
		return s, r, done, info

	def render(self, mode='human', **kwargs):
		super().render(mode=mode, **kwargs)

	def s(self):
		return self.robot.calc_state()

	def log_prior(self, target_loc):
		assert -0.5 <= target_loc[0] <= -0.05 or 0.05 <= target_loc[0] <= 0.5
		assert -0.3 <= target_loc[1] <= 0.2 and 0.65 <= target_loc[2] <= 1
		return 0
