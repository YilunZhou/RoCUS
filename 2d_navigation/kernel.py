
import numpy as np
from scipy.stats import norm, truncnorm

class TransitionKernel():
	'''
	A transition kernel on a random variable (or a set of RVs) stores the current value of the RV,
	propose() will propose a new RV by setting the value attribute, and return forward and backward 
	transition log probability.
	revert() will revert the proposed value. revert can only be done once after a proposal.
	sample_prior() will reset the current value to one sampled from the prior, and erase prev_value 
	to None since the chain is broken.
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

class RBF2dEnvKernelUniform(TransitionKernel):
	def __init__(self, N_points=15, obs_low=-0.7, obs_high=0.7):
		self.N_points = N_points
		self.obs_low = obs_low
		self.obs_high = obs_high
		super(RBF2dEnvKernelUniform, self).__init__()
	def propose(self):
		self.prev_value = self.value
		self.value = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.N_points, 2))
		return 0, 0
	def sample_prior(self):
		self.value = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.N_points, 2))
		self.prev_value = None

def truncnorm_rvs(a, b, mean, std):
	a_use = (a - mean) / std
	b_use = (b - mean) / std
	return truncnorm.rvs(a_use, b_use, mean, std)

def truncnorm_logpdf(x, a, b, mean, std):
	a_use = (a - mean) / std
	b_use = (b - mean) / std
	return truncnorm.logpdf(x, a_use, b_use, mean, std)

class RBF2dEnvKernelNormal(TransitionKernel):
	def __init__(self, sigma=0.1, N_points=15, obs_low=-0.7, obs_high=0.7):
		self.sigma = sigma
		self.N_points = N_points
		self.obs_low = obs_low
		self.obs_high = obs_high
		super(RBF2dEnvKernelNormal, self).__init__()
	def propose(self):
		self.prev_value = self.value
		total_forward_log_prob = 0
		total_backward_log_prob = 0
		self.value = np.zeros(self.prev_value.shape)
		for i in range(self.value.shape[0]):
			for j in range(self.value.shape[1]):
				self.value[i, j] = truncnorm_rvs(a=self.obs_low, b=self.obs_high, mean=self.prev_value[i, j], std=self.sigma)
				total_forward_log_prob += truncnorm_logpdf(self.value[i, j], a=self.obs_low, b=self.obs_high,
														   mean=self.prev_value[i, j], std=self.sigma)
				total_backward_log_prob += truncnorm_logpdf(self.prev_value[i, j], a=self.obs_low, b=self.obs_high,
															mean=self.value[i, j], std=self.sigma)
		return total_forward_log_prob, total_backward_log_prob
	def sample_prior(self):
		self.value = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.N_points, 2))
		self.prev_value = None

class RRTKernelNormal(TransitionKernel):
	def __init__(self, cspace_low, cspace_high, sigma_ratio=0.1):
		self.cspace_low = np.array(cspace_low)
		self.cspace_high = np.array(cspace_high)
		self.sigma = (np.array(cspace_high) - cspace_low) * sigma_ratio
		super(RRTKernelNormal, self).__init__()
	def propose(self):
		self.prev_value = self.value
		total_forward_log_prob = 0
		total_backward_log_prob = 0
		self.value = []
		for pv in self.prev_value:
			v = np.zeros(pv.shape)
			for i, p_val in enumerate(pv):
				v[i] = truncnorm_rvs(a=self.cspace_low[i], b=self.cspace_high[i], mean=p_val, std=self.sigma[i])
				total_forward_log_prob += truncnorm_logpdf(v[i], a=self.cspace_low[i], b=self.cspace_high[i],
														   mean=p_val, std=self.sigma[i])
				total_backward_log_prob += truncnorm_logpdf(p_val, a=self.cspace_low[i], b=self.cspace_high[i],
															mean=v[i], std=self.sigma[i])
			self.value.append(v)
		return total_forward_log_prob, total_backward_log_prob
	def sample_prior(self):
		self.value = []
		self.prev_value = None
	def __getitem__(self, idx):
		if idx >= len(self.value):
			if idx > len(self.value):
				print('accessing non-consecutive entries? ')
			for _ in range(len(self.value), idx + 1):
				new = np.random.uniform(low=self.cspace_low, high=self.cspace_high)
				self.value.append(new)
		return self.value[idx]
	def __setitem__(self, idx, val):
		raise Exception('You should not mannually set kernel entries. ')
