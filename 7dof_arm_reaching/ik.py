
import math as m
import numpy as np

import time
from klampt import WorldModel
from klampt.model.collide import WorldCollider
from klampt.model import ik

class IK():
	def __init__(self):
		self.world = WorldModel()
		self.robot = self.world.loadRobot(
			'franka_panda/panda_model_w_table.urdf')
		self.robot.setJointLimits(
			[-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0.0, 0.0, 0.0], 
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0, 0.0, 0.0]
		)
		self.initial_cfg_left = [0] * 7 + [-m.pi / 2, -m.pi / 2, m.pi / 2 + 0.3, - m.pi / 2, 0, m.pi / 2, 3 * m.pi / 4] + [0] * 3
		self.initial_cfg_right = [0] * 7 + [-m.pi / 2, m.pi / 2, m.pi / 2 - 0.3, - m.pi / 2, 0, m.pi / 2, 3 * m.pi / 4] + [0] * 3
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
