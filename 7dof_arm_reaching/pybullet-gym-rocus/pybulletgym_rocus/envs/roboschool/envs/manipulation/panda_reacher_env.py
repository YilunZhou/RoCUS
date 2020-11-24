from pybulletgym_rocus.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym_rocus.envs.roboschool.robots.manipulators.panda_reacher import PandaReacher
from pybulletgym_rocus.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np


class PandaReacherEnv(BaseBulletEnv):
    def __init__(self, shelf=False):
        self.robot = PandaReacher(shelf=shelf)
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81, timestep=0.0020, frame_skip=5)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        dist = self.robot.calc_potential()
        reward = 100 * dist - np.square(a).sum()
        return state, reward, False, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0.3, 0.3, 0.3, 0.0, 0, 0.625)
