
import time
import numpy as np

from klampt import WorldModel
from klampt.model import ik

import pybullet, gym, pybulletgym_rocus

class RRT():
	def __init__(self, control_res=0.05, collision_res=0.01):
		self.rrt_utils = RRTUtils()
		self.control_res = control_res
		self.collision_res = collision_res

	def attempt_connection(self, p, tree, neighbors, from_point):
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
		configs = np.linspace(p, pt, num_checks)
		if all(self.rrt_utils.in_free(cfg) for cfg in configs):
			neighbors.insert(p)
			tree.insert(p, idx)
			return True
		else:
			return False

	def get_path(self, target_loc, kernel):
		start = [0, -0.3, 0.0, -2, 0, 2.0, np.pi / 4]
		end = self.rrt_utils.target_loc_to_joint_config(target_loc)
		tree = Tree(start)
		neighbors = NearestNeighbor()
		neighbors.insert(start)
		if self.attempt_connection(end, tree, neighbors, from_point=-1):
			return tree.get_path_from_root(len(tree) - 1)
		for i in range(1000):
			p = kernel[i]
			if self.attempt_connection(p, tree, neighbors, from_point='nearest') and \
			   self.attempt_connection(end, tree, neighbors, from_point=-1):
				return tree.get_path_from_root(len(tree) - 1)
		return None

	def increase_resolution(self, traj):
		traj = np.array(traj)
		new_traj = [traj[0]]
		for start, end in zip(traj[:-1], traj[1:]):
			max_movement = (start - end).max()
			if max_movement < self.control_res:
				new_traj.append(end)
			else:
				num_cuts = int(max_movement / self.control_res) + 1
				segments = np.linspace(start, end, num_cuts + 1)[1:]
				new_traj.extend(list(segments))
		new_traj = np.array(new_traj)
		return new_traj

class IK():
	def __init__(self):
		self.world = WorldModel()
		self.robot = self.world.loadRobot('franka_panda/panda_model_w_table.urdf')
		self.robot.setJointLimits(
			[-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0.0, 0.0, 0.0], 
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0, 0.0, 0.0]
		)
		self.initial_cfg_left = [0] * 7 + [-np.pi / 2, -np.pi / 2, np.pi / 2 + 0.3, - np.pi / 2, 0, np.pi / 2, 3 * np.pi / 4] + [0] * 3
		self.initial_cfg_right = [0] * 7 + [-np.pi / 2, np.pi / 2, np.pi / 2 - 0.3, - np.pi / 2, 0, np.pi / 2, 3 * np.pi / 4] + [0] * 3
		self.grasptarget_link = self.robot.link(16)
		time.sleep(1)

	def solve(self, target_loc):
		if isinstance(target_loc, np.ndarray):
			target_loc = list(target_loc.flat)
		if target_loc[0] < 0:
			initial_cfg = self.initial_cfg_left
		else:
			initial_cfg = self.initial_cfg_right
		self.robot.setConfig(initial_cfg)
		objective = ik.objective(self.grasptarget_link, local=[0, 0, 0], world=target_loc)
		flag = ik.solve(objective, iters=1000, tol=1e-4)
		cfg = self.robot.getConfig()[7:14]
		return flag, cfg, initial_cfg[7:14]

class RRTUtils():
	def __init__(self):
		self.ik = IK()
		self.plan_env = gym.make('PandaReacher-v0', shelf=True)
		self.plan_env.reset()
		self.p = self.plan_env.robot._p
		self.drive_res = 100

	def getConfig(self):
		config = self.p.getJointStates(0, range(7))
		config = np.array([e[0] for e in config])
		return config

	def setConfig(self, cfg):
		for j in range(7):
			self.p.resetJointState(0, j, cfg[j], 0)
		self.p.setJointMotorControlArray(bodyIndex=0, jointIndices=range(7), controlMode=pybullet.POSITION_CONTROL,
										 targetPositions=cfg, positionGains=[0.5] * 7, velocityGains=[1.0] * 7)

	def drive_to_position(self, from_joint_config, to_joint_config):
		waypoints = np.linspace(from_joint_config, to_joint_config, self.drive_res)
		self.setConfig(from_joint_config)
		for waypoint in waypoints:
			self.p.setJointMotorControlArray(bodyIndex=0, jointIndices=range(7), controlMode=pybullet.POSITION_CONTROL,
											 targetPositions=waypoint, positionGains=[0.5] * 7, velocityGains=[1.0] * 7)
			self.p.stepSimulation()
		end_config = self.getConfig()
		if not self.in_collision(end_config):
			return end_config
		for u in np.arange(0.01, 1, 0.01):
			target_config = (1 - u) * end_config + u * np.array(from_joint_config)
			if not self.in_collision(target_config):
				return target_config
		raise Exception('not returning', target_loc)

	def in_collision(self, config):
		self.setConfig(config)
		self.p.stepSimulation()
		contacts = self.p.getContactPoints()
		for c in contacts:
			_, bodyA, bodyB, linkA, linkB, posA, posB, norm, dist, nf, _, _, _, _ = c
			if dist < 0:
				return True
		return False

	def in_free(self, config):
		return not self.in_collision(config)

	def target_loc_to_joint_config(self, target_loc):
		flag, cfg, initial_cfg = self.ik.solve(target_loc)
		self.plan_env.reset(target_loc=target_loc)
		cfg = self.drive_to_position(initial_cfg, cfg)
		assert not self.in_collision(cfg), target_loc
		return cfg

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
