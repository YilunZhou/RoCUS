from gym.envs.registration import register

register(
	id='PandaReacher-v0',
	entry_point='pybulletgym_rocus.envs.roboschool.envs.manipulation.panda_reacher_env:PandaReacherEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

def get_list():
	envs = ['- ' + spec.id for spec in gym.pgym.envs.registry.all() if spec.id.find('Bullet') >= 0 or spec.id.find('MuJoCo') >= 0]
	return envs
