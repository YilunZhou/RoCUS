
import sys
import numpy as np
from scipy.signal import convolve2d

class Modulator():
	def set_arena(self, arena, target=[1, 1], mod_margin=0.01):
		self.target = np.array(target)
		self.mod_margin = mod_margin
		occ_grid = convolve2d(1 - arena.occ_grid.copy(), np.ones((3, 3)), mode='same') < 1
		ccs = find_connected_components(occ_grid)
		res = arena.grid_res
		bd = arena.grid_bd
		grid_xs, grid_ys = np.linspace(-bd, bd, res), np.linspace(-bd, bd, res)
		self.gammas = []
		for i, cc in enumerate(ccs):
			cc_set = set([tuple(c) for c in cc])
			bdy = []
			for x, y in cc_set:
				if ((x - 1, y) not in cc_set) or ((x + 1, y) not in cc_set) or ((x, y - 1) not in cc_set) or ((x, y + 1) not in cc_set):
					bdy.append([x, y])
			bdy = np.array(bdy)
			cc_points = np.stack([grid_xs[cc[:, 1]], grid_ys[cc[:, 0]]], axis=1)
			bdy_points = np.stack([grid_xs[bdy[:, 1]], grid_ys[bdy[:, 0]]], axis=1)
			gamma = GammaFromPolygon(bdy_points, np.mean(cc_points, axis=0))
			self.gammas.append(gamma)

	def linear_controller(self, x, max_norm=1):
		x_dot = self.target - x
		n = np.linalg.norm(x_dot)
		if n < max_norm:
			return x_dot
		else:
			return x_dot / n * max_norm

	def modulation_HBS(self, pt, orig_ds, margin=0.01):
		epsilon = sys.float_info.epsilon
		gamma_vals = np.stack([gamma(pt) for gamma in self.gammas])
		# calculate each individual modulated control
		x_dot_mods = [get_individual_modulation(pt, orig_ds, gamma, self.mod_margin) for gamma in self.gammas]
		# calculate weighted average of magnitude
		ms = np.log(np.maximum(gamma_vals - 1, 0) + epsilon)
		logprod = ms.sum()
		bs = np.exp(logprod - ms)
		weights = bs / bs.sum()
		x_dot_mags = np.array([np.linalg.norm(d) for d in x_dot_mods])
		avg_mag = np.dot(weights, x_dot_mags)
		x_dot_mods = np.array(x_dot_mods).T
		x_dot_mods[:, x_dot_mags > 0] = x_dot_mods[:, x_dot_mags > 0] / x_dot_mags[x_dot_mags > 0]
		x_dot = orig_ds / np.linalg.norm(orig_ds)
		avg_ds_dir = get_weighted_sum(ref_dir=x_dot, dirs=x_dot_mods, weights=weights)
		x_mod_final = avg_mag * avg_ds_dir
		return x_mod_final

	def modulate(self, x):
		orig_ds = self.linear_controller(np.array(x))
		d = self.modulation_HBS(x, orig_ds)
		return d

def neighbor_idxs(i, j, H, W):
	i_s = set([i, min(i + 1, H - 1), max(i - 1, 0)])
	j_s = set([j, min(j + 1, W - 1), max(j - 1, 0)])
	neighbors = set([(i, j) for i in i_s for j in j_s])
	neighbors.remove((i, j))
	return neighbors

def find_connected_components(occ_grid):
	occ_grid = occ_grid.copy()
	ccs = []
	while occ_grid.sum() != 0:
		i, j = np.argwhere(occ_grid)[0]
		cc = set([])
		cc = find_single_cc(occ_grid, i, j)
		ccs.append(np.array(list(cc)))
		for ci, cj in cc:
			occ_grid[ci, cj] = 0
	return ccs

def find_single_cc(occ_grid, i, j):
	queue = [(i, j)]
	found = []
	explored = set([(i, j)])
	while len(queue) != 0:
		cur = queue.pop()
		found.append(cur)
		for ni, nj in neighbor_idxs(cur[0], cur[1], *occ_grid.shape):
			if (ni, nj) in explored or occ_grid[cur[0], cur[1]] == 0:
				continue
			explored.add((ni, nj))
			queue.append((ni, nj))
	assert len(found) == len(set(found))
	return found

class GammaFromPolygonOrig():
	def __init__(self, dists, center):
		self.dists = dists
		self.n_bins = len(dists)
		self.center = center
		self.angle_res = 2 * np.pi / self.n_bins

	def __call__(self, pt):
		pt = np.array(pt) - self.center
		ang = np.arctan2(pt[1], pt[0])
		if ang < 0:
			ang = ang + 2 * np.pi
		idx1 = int(ang / self.angle_res)
		idx2 = (idx1 + 1) % self.n_bins
		d1, d2 = self.dists[idx1], self.dists[idx2]
		a1, a2 = self.angle_res * idx1, self.angle_res * idx2
		m = np.array([np.cos(a1) * d1, np.sin(a1) * d1])
		n = np.array([np.cos(a2) * d2, np.sin(a2) * d2])
		t = (pt[0] * (n[1] - m[1]) - pt[1] * (n[0] - m[0])) / (n[1] * m[0] - n[0] * m[1])
		return t

	def grad(self, pt):
		pt = np.array(pt) - self.center
		ang = np.arctan2(pt[1], pt[0])
		if ang < 0:
			ang = ang + 2 * np.pi
		idx1 = int(ang / self.angle_res)
		idx2 = (idx1 + 1) % self.n_bins
		d1, d2 = self.dists[idx1], self.dists[idx2]
		a1, a2 = self.angle_res * idx1, self.angle_res * idx2
		m = np.array([np.cos(a1) * d1, np.sin(a1) * d1])
		n = np.array([np.cos(a2) * d2, np.sin(a2) * d2])
		grad = np.array([ (n[1] - m[1]) / (n[1] * m[0] - n[0] * m[1]), 
						 -(n[0] - m[0]) / (n[1] * m[0] - n[0] * m[1])])
		return grad / (np.linalg.norm(grad) + sys.float_info.epsilon)

class GammaFromPolygon():
	def __init__(self, bdy_points, center, max_nbins=50):
		self.bdy_points = bdy_points
		self.center = center
		n_bins = min(int(len(bdy_points) / 3), max_nbins)
		angle_res = 2 * np.pi / n_bins
		dists = [None] * n_bins
		for point in bdy_points:
			angle = np.arctan2(point[1] - center[1], point[0] - center[0])
			dist = np.linalg.norm(point - center)
			if angle < 0:
				angle += 2 * np.pi
			bin_idx = int(angle / angle_res)
			assert bin_idx <= len(dists)
			if bin_idx == len(dists):
				bin_idx = 0
			if dists[bin_idx] is None or dist > dists[bin_idx]:
				dists[bin_idx] = dist
		angles = np.linspace(0, 2 * np.pi, n_bins + 1)[:-1] + angle_res / 2
		angle_dists = [ad for ad in zip(angles, dists) if ad[1] is not None]
		angles, dists = zip(*angle_dists)
		self.angles = np.array(angles)
		self.dists = np.array(dists)

	def __call__(self, pt):
		pt = np.array(pt) - self.center
		ang = np.arctan2(pt[1], pt[0])
		if ang < 0:
			ang = ang + 2 * np.pi
		idx2 = (self.angles < ang).sum() % len(self.angles)
		idx1 = idx2 - 1
		if idx1 == -1:
			idx1 = len(self.angles) - 1
		d1, d2 = self.dists[idx1], self.dists[idx2]
		a1, a2 = self.angles[idx1], self.angles[idx2]
		m = np.array([np.cos(a1) * d1, np.sin(a1) * d1])
		n = np.array([np.cos(a2) * d2, np.sin(a2) * d2])
		t = (pt[0] * (n[1] - m[1]) - pt[1] * (n[0] - m[0])) / (n[1] * m[0] - n[0] * m[1])
		return t

	def grad(self, pt):
		pt = np.array(pt) - self.center
		ang = np.arctan2(pt[1], pt[0])
		if ang < 0:
			ang = ang + 2 * np.pi
		idx2 = (self.angles < ang).sum() % len(self.angles)
		idx1 = idx2 - 1
		if idx1 == -1:
			idx1 = len(self.angles) - 1
		d1, d2 = self.dists[idx1], self.dists[idx2]
		a1, a2 = self.angles[idx1], self.angles[idx2]
		m = np.array([np.cos(a1) * d1, np.sin(a1) * d1])
		n = np.array([np.cos(a2) * d2, np.sin(a2) * d2])
		grad = np.array([ (n[1] - m[1]) / (n[1] * m[0] - n[0] * m[1]), 
						 -(n[0] - m[0]) / (n[1] * m[0] - n[0] * m[1])])
		return grad / (np.linalg.norm(grad) + sys.float_info.epsilon)

def get_orthogonal_basis(v):
	v = np.array(v)
	v_norm = np.linalg.norm(v)
	assert v_norm > 0, 'v must be non-zero'
	v = v / v_norm
	basis = np.zeros((2, 2))
	basis[:, 0] = v
	basis[:, 1] = [v[1], -v[0]]
	return basis

def get_decomposition_matrix(pt, gamm_grad, ref_pt):
	adapt_threshold = 0.05
	ref_dir = ref_pt - pt
	ref_norm = np.linalg.norm(ref_dir)
	if ref_norm > 0:
		ref_dir = ref_dir / ref_norm
	dot_prod = np.dot(gamm_grad, ref_dir)
	if np.abs(dot_prod) < adapt_threshold:
		if not np.linalg.norm(gamm_grad): # zero
			gamm_grad = - ref_dir
		else:
			if dot_prod < 0:
				dir_norm = -1
			else:
				dir_norm = 1
			weight = np.abs(dot_prod) / adapt_threshold
			dirs = np.stack([ref_dir, dir_norm * gamm_grad], axis=1)
			weights = np.array([weight, 1 - weight])
			ref_dir = get_weighted_sum(ref_dir=gamm_grad, dirs=dirs, weights=weights)
	E_orth = get_orthogonal_basis(gamm_grad)
	E = E_orth.copy()
	E[:, 0] = - ref_dir
	return E, E_orth

def get_weighted_sum(ref_dir, dirs, weights):
	assert np.linalg.norm(ref_dir) > 0, 'ref_dir cannot be 0'
	ref_dir = ref_dir / np.linalg.norm(ref_dir)
	dirs = dirs[:, weights > 0]
	weights = weights[weights > 0]
	if len(weights) == 1:
		return dirs.flatten()
	norms = np.linalg.norm(dirs, axis=0)
	dirs[:, norms > 0] = dirs[:, norms > 0] / norms[norms > 0]
	basis = get_orthogonal_basis(ref_dir)
	dirs_ref_space = np.zeros(np.shape(dirs))
	for j in range(dirs.shape[1]):
		dirs_ref_space[:,j] = basis.T.dot(dirs[:, j])
	dirs_dir_space = dirs_ref_space[1:, :]
	norms = np.linalg.norm(dirs_dir_space, axis=0)
	dirs_dir_space[:, norms > 0] = (dirs_dir_space[:, norms > 0] / np.tile(norms[norms > 0], (1, 1)))
	cos_dir = dirs_ref_space[0, :]
	if np.sum(cos_dir > 1) or np.sum(cos_dir < -1):
		cos_dir = np.min(np.vstack((cos_dir, np.ones(len(weights)))), axis=0)
		cos_dir = np.max(np.vstack((cos_dir, -np.ones(len(weights)))), axis=0)
	dirs_dir_space = dirs_dir_space * np.arccos(cos_dir)
	weighted_sum_dirspace = (dirs_dir_space * weights).sum(axis=1)
	norm = np.linalg.norm(weighted_sum_dirspace)
	if norm != 0:
		s = (np.cos(norm), np.sin(norm) / norm * weighted_sum_dirspace)
		pre_transform = np.concatenate([[np.cos(norm)], np.sin(norm) / norm * weighted_sum_dirspace])
		weighted_sum = np.dot(basis, pre_transform)
	else:
		weighted_sum = basis[:,0]
	return weighted_sum

def get_individual_modulation(pt, orig_ds, gamma, margin=0.01, tangent_scaling_max=5):
	gamma_val = gamma(pt)
	if gamma_val > 1e9:  # too far away, no modulation
		return orig_ds
	elif gamma_val < 1 + margin:  # inside obstacle (including margin). perform repulsion
		rel_pt = pt - gamma.center
		speed =  (((1 + margin) / gamma_val) ** 5 - (1 + margin)) * 5
		if np.linalg.norm(rel_pt) != 0: # nonzero
			x_dot_mod = rel_pt / np.linalg.norm(rel_pt) * speed
		else:
			x_dot_mod = np.array([speed, 0])
		return x_dot_mod
	else:  # oustide obstacle. perform normal modulation
		gamma_grad = gamma.grad(pt)
		gamma_grad = gamma_grad / np.linalg.norm(gamma_grad)
		ref_pt = gamma.center
		E, E_orth = get_decomposition_matrix(pt, gamma_grad, ref_pt)
		invE = np.linalg.inv(E)
		inv_gamma = 1 / gamma_val
		tangent_scaling = max(1, tangent_scaling_max - (1 - inv_gamma))
		D = np.diag([1 - inv_gamma, tangent_scaling * (1 + inv_gamma)])
		M = np.matmul(np.matmul(E, D), invE)
		x_dot_mod = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()
		return x_dot_mod
