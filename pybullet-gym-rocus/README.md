# pybullet-gym-rocus

This directory contains dependency `pybullet-gym-rocus`, a customized version of [`pybullet-gym`](https://github.com/benelot/pybullet-gym). This version contains the Franka Panda Arm robot data and environment, can be installed side-by-side with the original`pybullet-gym`, and uses `pybulletgym_rocus` as the package name (i.e. `import pybulletgym_rocus`). 

To install, simply run `pip install -e .` in this directory. It will install [`pybullet`](https://pybullet.org/wordpress/) for you if necessary, but you need to install [`openai-gym`](https://gym.openai.com/) on your own (i.e. `pip install gym`). 
