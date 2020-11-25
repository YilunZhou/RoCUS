
import time
from copy import deepcopy as copy

import numpy as np

from klampt import WorldModel
from klampt.model import ik

class IK():
	def __init__(self):
		self.world = WorldModel()
		self.robot = self.world.loadRobot('franka_panda/panda_model_w_table.urdf')
		self.robot.setJointLimits(
			[-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0.0, 0.0, 0.0], 
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0, 0.0, 0.0]
		)
		self.grasptarget_link = self.robot.link(16)
		time.sleep(1)

	def __call__(self, from_cfg, to_ee_loc, return_flag=False):
		if isinstance(to_ee_loc, np.ndarray):
			to_ee_loc = list(to_ee_loc.flat)
		from_cfg_full = copy(self.robot.getConfig())
		from_cfg_full[7:14] = from_cfg
		self.robot.setConfig(from_cfg_full)
		objective = ik.objective(self.grasptarget_link, local=[0, 0, 0], world=to_ee_loc)
		flag = ik.solve(objective, iters=1000, tol=1e-4)
		cfg = self.robot.getConfig()[7:14]
		if not return_flag:
			return cfg
		else:
			return cfg, flag
