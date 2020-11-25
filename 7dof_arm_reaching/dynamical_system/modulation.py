import pickle
import numpy as np

class Modulator():
	def __init__(self, typ, svm_fn):
		self.model, self.svm_gamma, self.svm_C = pickle.load(open(svm_fn, 'rb'))
		self.classifier = self.model['classifier']
		self.max_dist = self.model['max_dist']
		if typ == 'original':
			self.ref_pts = self.model['reference_points']
		elif typ == 'improved':
			self.ref_pts = np.array([[0, 0.15, 0.975], [0.0, 0.5, 0.80], [0.0, 0, 0.61]])
		self.typ = typ
		self.max_lin_vel = 1.0
		self.margin = 0.01
		self.tangent_scale_max = 7.5
		self.max_norm = 0.1

	def gamma(self, x):
		x = x.flatten()
		svm_bias = self.classifier.intercept_
		score = self.classifier.decision_function(x.reshape(1, 3))
		ref_pt = self.find_closest_ref_pt(x)
		dist = [np.linalg.norm(x - ref_pt)]
		score_adapt = score
		if self.typ == 'original':
			bias_scale = 1.0
		elif self.typ == 'improved':
			bias_scale = 2.0
		ind_highvals = score < bias_scale * svm_bias
		score_adapt  = score
		score_adapt[ind_highvals]  = score[ind_highvals] + bias_scale * svm_bias
		score_adapt[ind_highvals]  = bias_scale * svm_bias
		outer_ref_dist = self.max_dist * 2
		dist = np.clip(dist, self.max_dist, outer_ref_dist)
		ind_noninf = outer_ref_dist > dist
		distance_score = (outer_ref_dist - self.max_dist) / (outer_ref_dist - dist[ind_noninf])
		max_float = 1e12
		gamma = np.zeros(dist.shape)
		gamma[ind_noninf] = (-score_adapt[ind_noninf] + 1) * distance_score
		gamma[~ind_noninf] = max_float
		gamma = gamma[0]
		return gamma

	def gamma_grad(self, x):
		x = x.flatten()
		grad = np.zeros(len(x))
		for i in range(3):
			x_low, x_high = x.copy(), x.copy()
			x_high[i] = x_high[i] + 1e-5
			x_low[i] = x_low[i] - 1e-5
			gm_high = self.gamma(x_high)
			gm_low = self.gamma(x_low)
			grad[i] = (gm_high - gm_low) / (2 * 1e-5)
		norm = np.linalg.norm(grad, axis=0)
		if norm > 0:
			grad = grad / norm
		return grad

	def linear_controller(self, x, x_target):
		return x_target - x

	def get_modulated_direction(self, x, x_target, repulsive_gammaMargin = 0.01):
		x = np.array(x)
		d = len(x)
		assert d == 3
		x_target = np.array(x_target)
		gamma_val  = self.gamma(x)
		gamma_grad_val = self.gamma_grad(x).flatten()
		orig_ds = self.linear_controller(x, x_target)
		x_dot = self.modulation_single_obs_multi_ref(x, orig_ds, gamma_val, gamma_grad_val)
		if np.linalg.norm(x_dot) > self.max_lin_vel:
			x_dot = x_dot / np.linalg.norm(x_dot) * self.max_lin_vel
		return x_dot

	def null_space_bases(n):
		'''construct a set of d-1 basis vectors orthogonal to the given d-dim vector n'''
		x = np.random.rand(d)
		x -= x.dot(n) * n / np.linalg.norm(n) ** 2
		x /= np.linalg.norm(x)
		y = np.cross(x, n)
		assert max(np.dot(n,x), np.dot(n,y), np.dot(x, y)) <= 1e-10, 'not orthogonal?'
		return [x, y]

	def linear_controller(self, x, x_target):
		x_dot = x_target - x
		n = np.linalg.norm(x_dot)
		if n < self.max_norm:
			return x_dot
		else:
			return x_dot / n * self.max_norm

	def modulation_single_obs_multi_ref(self, x, orig_ds, gamma_val, gamma_grad_val):
		'''
			Computes modulated velocity for an environment described by a single gamma function
			and multiple reference points (describing multiple obstacles)
		'''
		if gamma_val > 1e9:
			return orig_ds
		ref_pt = self.find_closest_ref_pt(x)
		if gamma_val < ( 1 + self.margin):
			repu_dir = x - ref_pt
			x_dot_mod = 5 * (repu_dir / np.linalg.norm(repu_dir) + orig_ds / np.linalg.norm(orig_ds))
		else:
			# Calculate real modulated dynamical system
			M = self.modulation_single_obs(x, orig_ds, gamma_grad_val, gamma_val, ref_pt)
			x_dot_mod = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()

		return x_dot_mod

	def modulation_single_obs(self, x, orig_ds, gamma_grad_val, gamma_val, ref_pt):
		'''
		Compute modulation matrix for a single obstacle described with a gamma function
		and unique reference point
		'''
		x = np.array(x)
		assert len(x.shape) == 1, 'x is not 1-dimensional?'
		d = x.shape[0]
		n = gamma_grad_val
		E, E_orth = compute_decomposition_matrix(x, gamma_grad_val, ref_pt)
		invE = np.linalg.inv(E)
		tangent_scaling = 1
		if gamma_val <= 1:
			inv_gamma = 1
		else:
			inv_gamma = 1 / gamma_val        
			# tail effect
			tail_angle = np.dot(gamma_grad_val, orig_ds)
			if (tail_angle) < 0:
				# robot moving towards obstacle
				tangent_scaling = max(1, self.tangent_scale_max - (1 - inv_gamma))
			else:
				# robot moving away from  obstacle
				tangent_scaling   = 1.0
				inv_gamma = 0
		lambdas = np.stack([1 - inv_gamma] + [tangent_scaling * (1 + inv_gamma)] * (d - 1))
		D = np.diag(lambdas)
		M = np.matmul(np.matmul(E, D), invE)
		if gamma_val > 1e9:
			M =  np.identity(d)
		return M

	def find_closest_ref_pt(self, x):
		dists = np.linalg.norm(x - self.ref_pts, axis=1)
		return self.ref_pts[dists.argmin()]

def compute_decomposition_matrix(x, normal_vec, ref_pt):
	dot_margin = 0.05
	ref_dir = - (x - ref_pt)
	ref_norm = np.linalg.norm(ref_dir)
	if ref_norm > 0:
		ref_dir = ref_dir / ref_norm 
	dot_prod = np.dot(normal_vec, ref_dir)
	if np.abs(dot_prod) < dot_margin:
		if np.linalg.norm(normal_vec) == 0:
			normal_vec = -ref_dir
		else:
			weight = np.abs(dot_prod) / dot_margin
			dir_norm = np.copysign(1, dot_prod)
			ref_dir = combine(ref_dir=normal_vec, dirs=np.vstack((ref_dir, dir_norm * normal_vec)).T, weights=np.array([weight, (1 - weight)]))
	E_orth = get_orthogonal_basis(normal_vec)
	E = E_orth.copy()
	E[:, 0] = -ref_dir
	return E, E_orth

def combine(ref_dir, dirs, weights, total_weight=1):
	ref_dir = np.copy(ref_dir)
	dirs = dirs[:, weights > 0] 
	weights = weights[weights > 0]
	if total_weight < 1:
		weights = weights / np.sum(weights) * total_weight
	n_directions = weights.shape[0]
	if (n_directions == 1) and total_weight >= 1:
		return dirs[:, 0]
	dim = np.array(ref_dir).shape[0]
	assert np.linalg.norm(ref_dir) != 0, 'Zero norm direction as input'
	ref_dir /= np.linalg.norm(ref_dir)
	norm_dir = np.linalg.norm(dirs, axis=0)
	dirs[:, norm_dir > 0] = dirs[:, norm_dir > 0] / np.tile(norm_dir[norm_dir > 0], (dim, 1))
	basis = get_orthogonal_basis(ref_dir)
	dirs_ref = np.zeros(np.shape(dirs))
	for ii in range(np.array(dirs).shape[1]):
		dirs_ref[:,ii] = basis.T.dot( dirs[:,ii])
	dirs_kappa = dirs_ref[1:, :]
	norm_kappa = np.linalg.norm(dirs_kappa, axis=0)
	dirs_kappa[:,norm_kappa > 0] = (dirs_kappa[:, norm_kappa > 0] /  np.tile(norm_kappa[norm_kappa > 0], (dim-1, 1)))
	cos_dirs = dirs_ref[0,:]
	if np.sum(cos_dirs > 1) or np.sum(cos_dirs < -1):
		cos_dirs = np.min(np.vstack((cos_dirs, np.ones(n_directions))), axis=0)
		cos_dirs = np.max(np.vstack((cos_dirs, -np.ones(n_directions))), axis=0)
	dirs_kappa *= np.tile(np.arccos(cos_dirs), (dim-1, 1))
	dirs_kappa_sum = np.sum(dirs_kappa * np.tile(weights, (dim-1, 1)), axis=1)
	norm_kappa_sum = np.linalg.norm(dirs_kappa_sum)
	if norm_kappa_sum:
		dirs_sum = (basis.dot(np.hstack((np.cos(norm_kappa_sum), np.sin(norm_kappa_sum) / norm_kappa_sum * dirs_kappa_sum))))
	else:
		dirs_sum = basis[:, 0]
	return dirs_sum

def get_orthogonal_basis(v):
	if isinstance(v, list):
		v = np.array(v)
	v_norm = np.linalg.norm(v)
	assert v_norm != 0, 'Orthogonal basis matrix not defined for 0 vector.'
	v = v / v_norm
	dim = v.shape[0]
	basis = np.zeros((dim, dim))
	basis[:, 0] = v
	basis[:, 1] = np.array([-v[1], v[0], 0])
	norm_vec2 = np.linalg.norm(basis[:, 1])
	if norm_vec2:
		basis[:, 1] = basis[:, 1] / norm_vec2
	else:
		basis[:, 1] = [1, 0, 0]
	basis[:, 2] = np.cross(basis[:, 0], basis[:, 1])
	norm_vec = np.linalg.norm(basis[:, 2])
	if norm_vec:
		basis[:, 2] = basis[:, 2] / norm_vec
	return basis
