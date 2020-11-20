
import numpy as np

class NearestNeighbor():
	def __init__(self):
		self.points = []
	def insert(self, pt):
		self.points.append(pt)
	def nearest(self, pt):
		dists = np.linalg.norm(np.array(self.points) - pt, axis=1)
		idx = np.argmin(dists)
		return {'idx': idx, 'dist': np.min(dists), 'point': self.points[idx]}

class Node():
	def __init__(self, pt, parent=None):
		self.pt = pt
		self.parent = parent

class Tree():
	def __init__(self, root_pt):
		self.root = Node(root_pt)
		self.nodes = [self.root]
	def insert(self, pt, parent_idx):
		self.nodes.append(Node(pt, self.nodes[parent_idx]))
	def get_path_from_root(self, idx):
		path = [self.nodes[idx]]
		while path[-1].parent is not None:
			path.append(path[-1].parent)
		path = np.array([p.pt for p in path[::-1]])
		return path
	def __len__(self):
		return len(self.nodes)
	def __getitem__(self, idx):
		return self.nodes[idx]

class RRT():
	def __init__(self, control_res=0.03, collision_res=0.01):
		self.control_res = control_res
		self.collision_res = collision_res

	def attempt_connection(self, p, arena, tree, neighbors, from_point):
		if from_point == 'nearest':
			nearest_dict = neighbors.nearest(p)
			idx = nearest_dict['idx']
			dist = nearest_dict['dist']
			pt = nearest_dict['point']
		else:
			assert isinstance(from_point, int)
			idx = from_point
			pt = tree[idx].pt
			dist = np.linalg.norm(np.array(p) - pt)
		num_checks = int(dist / self.collision_res) + 1
		xs = np.linspace(pt[0], p[0], num_checks)
		ys = np.linspace(pt[1], p[1], num_checks)
		xys = np.stack((xs, ys), axis=1)
		if np.all(arena.gamma_batch(xys) > 1):
			neighbors.insert(p)
			tree.insert(p, idx)
			return True
		else:
			return False

	def get_path(self, arena, controller_kernel=None):
		start = [-1, -1]
		end = [1, 1]
		tree = Tree(start)
		neighbors = NearestNeighbor()
		neighbors.insert(start)
		if self.attempt_connection(end, arena, tree, neighbors, from_point=-1):
			return tree.get_path_from_root(len(tree) - 1)
		for i in range(1000):
			if controller_kernel is not None:
				p = controller_kernel[i]
			else:
				p = np.random.uniform(low=-1, high=1, size=2)
			if self.attempt_connection(p, arena, tree, neighbors, from_point='nearest'):
				if self.attempt_connection(end, arena, tree, neighbors, from_point=-1):
					return tree.get_path_from_root(len(tree) - 1)
		return None

	def increase_resolution(self, xys):
		xys = np.array(xys)
		new_xys = [xys[0]]
		for xy1, xy2 in zip(xys[:-1], xys[1:]):
			dist = np.linalg.norm(xy1 - xy2)
			if dist < self.control_res:
				new_xys.append(xy2)
			else:
				num_cuts = int(dist / self.control_res) + 1
				xs = np.linspace(xy1[0], xy2[0], num_cuts + 1)[1:]
				ys = np.linspace(xy1[1], xy2[1], num_cuts + 1)[1:]
				new_xys.extend(list(zip(xs, ys)))
		new_xys = np.array(new_xys)
		return new_xys
