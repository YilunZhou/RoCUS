# RoCUS: Robot Controller Understanding via Sampling

This is the code repository for [_RoCUS: Robot Controller Understanding via Sampling_](paper/paper.pdf) by Yilun Zhou, Serena Booth, Nadia Figueroa, and Julie Shah, implementing the two problem domains studied in the paper.

## Paper, Supplemental, and Video  
[paper/paper.pdf](paper/paper.pdf) has the full paper including supplementary materials. 

[paper/supplementary.pdf](paper/supplementary.pdf) has only the supplementary materials. 

YouTube Video: [https://youtu.be/IZigHZ4Gvf4](https://youtu.be/IZigHZ4Gvf4)


## Codebase

![Comparing 2D Navigation Controllers: RRT, IL, and DS](/figures/compare_controllers.png)

* In `2d_navigation`, a robot needs to navigate from the starting to the target position while avoiding irregularly shaped obstacles. Three controllers are implemented: rapidly-exploring random tree, dynamical system, and deep imitation learning.
* In `7dof_arm_reaching`, a 7 degree-of-freedom (DoF) Franka Panda robot arm mounted on the side of a table needs to reach to a specific location on the table while avoiding a T-shaped divider. Three controllers are implemented: rapidly-exploring random tree, dynamical system, and deep reinforcement learning.

Please refer to the respective `<problem-domain>/README.md` for detailed instructions on replicating (and extending) the experiments.

## Requirements
Python version of at least 3.7 is required to run the code.

### Common Dependencies
The code should run with reasonably recent versions of the packages listed below, but if you encounter any problems, [`requirements.txt`](requirements.txt) lists the exact versions of all dependencies used to develop these codes.

The standard suite of scientific computing packages is required: [`numpy`](https://numpy.org/), [`scipy`](https://www.scipy.org/), [`scikit-learn`](https://scikit-learn.org/stable/), and [`matplotlib`](https://matplotlib.org/). In addition, [`pytorch`](https://pytorch.org/) and [`tqdm`](https://github.com/tqdm/tqdm) are also required.

### 7DoF Arm Reaching
This domain additionally depends on [`klampt`](http://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/) and [`pybullet-gym`](https://github.com/benelot/pybullet-gym) (which further depends on [`pybullet`](https://pybullet.org/wordpress/) and [`openai-gym`](https://gym.openai.com/)).

The recommended way to install `pybullet-gym` is in the "editable mode", via `pip install -e`. This package contains a large number of robots, but on the other hand not the Franka Panda Arm that we use. In addition, since `pybullet-gym` is not officially archived in [PyPI](https://pypi.org/) and to prevent package collision in case that you already have `pybullet-gym` installed, we provide a custom package called [`pybullet-gym-rocus`](pybullet-gym-rocus/) that can be installed side by side with `pybullet-gym` and contains only the Franka Panda environment required for this code. Please see [`pybullet-gym-rocus/README.md`](pybullet-gym-rocus/README.md) for detailed installation instruction.

## Questions? 
Please contact us at `{yilun, serenabooth, nadiafig, julie_a_shah}@csail.mit.edu`.
