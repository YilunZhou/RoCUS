import sys
sys.path.append("../dynamical_system_modulation_svm/")
import learn_gamma_fn
from modulation_utils import *
import modulation_svm
import pickle
import numpy as np
epsilon = sys.float_info.epsilon

# class Modulator():
# 	def __init__(self):
# 		pass
# 	def get_modulated_direction(self, x, x_target):
# 		return x_target - x

class Modulator():
	def __init__(self, filename="", reference_points = []):
		self.learned_gamma, self.gamma_svm, self.c_svm = pickle.load(open(filename, 'rb'))
		self.classifier         = self.learned_gamma['classifier']
		self.max_dist           = self.learned_gamma['max_dist']
		if reference_points == []:
			self.reference_points   = self.learned_gamma['reference_points']
		else: 
			self.reference_points = reference_points
		self.max_lin_vel        = 1.0

	def gamma(self, x):
		return learn_gamma_fn.get_gamma(np.array(x), self.classifier, self.max_dist, self.reference_points, dimension=3)

	def gamma_grad(self, x, dim=3):
		return learn_gamma_fn.get_normal_direction(np.array(x), self.classifier, self.reference_points, self.max_dist, dimension=3)

	def linear_controller(self, x, x_target):
		return x_target - x

	def get_reference_points(self):
		return	self.reference_points

	def get_modulated_direction(self, x, x_target, repulsive_gammaMargin = 0.01):
		x = np.array(x)
		d = len(x)
		assert d == 3
		x_target = np.array(x_target)
		gamma_val  = self.gamma(x)
		normal_vec = self.gamma_grad(x)
		orig_ds    = self.linear_controller(x, x_target)
		x_dot      = modulation_svm.modulation_singleGamma_HBS_multiRef(query_pt=x, orig_ds=orig_ds, gamma_query=gamma_val,
        	normal_vec_query=normal_vec.reshape(d), obstacle_reference_points=self.reference_points, repulsive_gammaMargin = repulsive_gammaMargin)
		if np.linalg.norm(x_dot) > self.max_lin_vel:
			x_dot = x_dot/np.linalg.norm(x_dot) * self.max_lin_vel

		return x_dot

	def get_openloop_trajectory(self, x_initial, x_target, dt = 0.03, eps = 0.03, max_N = 1000):
		x_traj, _ = modulation_svm.forward_integrate_singleGamma_HBS(np.array(x_initial), np.array(x_target), self.learned_gamma, dt, eps, max_N)
		return x_traj.T

def rand_target_loc():
    '''
    generate random target location
    '''
    x = np.random.uniform(low=0.05, high=0.5)
    if np.random.randint(0, 2) == 0:
        x = -x
    y = np.random.uniform(low=-0.3, high=0.2)
    z = np.random.uniform(low=0.65, high=1.0)
    return x, y, z

if __name__ == '__main__':
	model_filename = "../dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS_bounded_pyb.pkl"
	x_init    = [0, -0.137, 1.173]  
	x_target = rand_target_loc()
	m = Modulator(filename=model_filename)
	gamma     = m.gamma(x_init)
	normal    = m.gamma_grad(x_init)
	x_dot_mod = m.get_modulated_direction(x_init, x_target)
	x_traj_openloop = m.get_openloop_trajectory(x_init, x_target)
	print(x_traj_openloop.shape)