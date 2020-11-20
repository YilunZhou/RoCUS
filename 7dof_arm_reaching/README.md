# RoCUS: Robot Controller Understanding via Sampling

This directory, `7dof_arm_reaching`, contains code defining our DS, RRT, and PPO (RL) models. Before running this code, 
you will need to first install `pybullet-gym`; see the instructions in `pybullet-gym-rocus/README.md`. 

While our `2d_navigation` code has been cleaned up for relatively user engagement, this arm reaching codebase needs a 
refactor. We will clean and update this codebase prior to publication.

## Running the MCMC Sampler
The main entry point is `run.py`, which contains a list of sampling instances, each implemented as a parameter-free function. This script can be executed via 
1. `python run.py`, which will bring up an interactive dialogue to prompt you to select an instance; or
2. `python run.py <sampling-instance-name>` where `<sampling-instance-name>` is the name of the function. 

All sampling instances require a pickle file of prior samples for the respective controller, which is used to computer `sigma` from `alpha`. The prior samples for all three controllers have been generated and placed in `samples/<controller>_prior.pkl`. 

If you wish to implement a custom sampling instance in `run.py`, you need to decorate it with `@register`, similar to existing ones. 

## General Code Structure
* `behavior.py` implements all the behavior functions (e.g., trajectory distance). Specifically, each behavior function takes two arguments, a trajectory and an environment instance. It will return two variables, `behavior_value` and `acceptable`. If the trajectory behavior is not "acceptable", this sample will be automatically discarded by the MCMC sampler. For example, if the behavior is only meaningfully defined for successful trajectories (e.g. trajectory length), then the `behavior_func` can set `acceptable = False` for unsuccessful trajectories. There is also a special behavior constructor, `neg_behavior`, that takes in a behavior function and outputs a behavior function with the negative values while still observing the `acceptable` flag. 
* `controller.py` implements controllers (RRT and RL), which relies on functionalities in the respective `<controller>/` folder. 
* `kernel.py` implements all the kernels, including those on environments (i.e. tasks) and controllers. A kernel keeps track of a random variable, and can change it via `propose()` and `revert()` to be called by the MCMC sampler. They also implement a `sample_prior()` function to sample the random variable from its distribution. All kernels are sub-classes of `TransitionKernel` class, which implements a dummy deterministic kernel with a value of constant 0 (useful for deterministic controllers). 
* `sampler.py` implements the core sampling functionality, with `sample()` implementing the MH algorithm. This algorithm take two key parameters, `env_kernel` and `controller_kernel`, which are instances of subclasses of the `TransitionKernel` class (defined in `kernel.py`). It is assumed that the controller randomness is independent from that of the environment, i.e. _p(u) = p(u | e)_. 

## Questions?
Contact us: {yilun, serenabooth, nadiafig, julie_a_shah}@csail.mit.edu. 