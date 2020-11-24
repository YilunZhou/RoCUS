
# 2D Navigation Problem

## Controllers
We implemented three controllers, dynamical system (DS), imitation learning (IL), and rapidly exploring random tree (RRT). The implementation details are in the respective `<controller>/` folder. 

## Running the MCMC Sampler

### Prior Generation
All sampling instances require a pickle file of prior samples for the respective controller, which is used to computer `sigma` from `alpha`. The prior samples for all three controllers have been generated and placed in `samples/<controller>_prior.pkl`. If you wish to re-generate these prior files, you can run `python sample_priors.py` and follow the instructions there. 

### MCMC Sampling
The main entry point is `run.py`, which contains a list of sampling instances, each implemented as a parameter-free function. This script can be executed via 
1. `python run.py`, which will bring up an interactive dialogue to prompt you to select an instance; or
2. `python run.py <sampling-instance-name>` where `<sampling-instance-name>` is the name of the function. 
The samples will be saved to the `samples/` directory. 

If you wish to implement a custom sampling instance in `run.py`, you need to decorate it with `@register`, similar to existing ones. 

### Visualization
To visualize samples, you can run `python visualize_samples.py --prior <prior-file> --posterior <posterior-file> --save-fn <image-file>` where both `<prior-file>` and `<posterior-file>` are the `.pkl` files representing samples (i.e. `samples/xxx.pkl`). This script will plot the trajectories and obstacle configurations. The left panel will show the trajectories. The right panel shows the obstacle distribution: if one of `--prior` and `--posterior` is specified, the obstacle distribution for that is shown; if both are specified, the difference in obstacle distribution is shown, with red color representing regions with higher likelihood of obstacle presence under the posterior than the prior. if `--save-fn` is specified, then the image will be saved to `<image-file>`. 

In addition to to the prior samples, we have also generated DS, IL and RRT posterior samples for minimal straight line deviation behavior, which are stored in the `samples/` directory. Their corresponding visualizations are stored in `figures/` directory, but if you want to re-generate them, you can run
```sh
python visualize_samples.py --prior ds_prior.pkl --poseterior ds_min_straightline_deviation.pkl --save-fn figures/ds_min_straightline_deviation.png
python visualize_samples.py --prior il_prior.pkl --poseterior il_min_straightline_deviation.pkl --save-fn figures/il_min_straightline_deviation.png
python visualize_samples.py --prior rrt_prior.pkl --poseterior rrt_min_straightline_deviation.pkl --save-fn figures/rrt_min_straightline_deviation.png
```

## General Code Structure
* `environment.py` implements the `RBF2dGymEnv`, which is the environment with a `gym`-like interface. It requires an `RBFArena` for obstacle definition and a `PhysicsSimulator` for simulating non-elastic collision. Both classes are defined in `environment.py`. 
* `controller.py` implements all controllers (DS, IL, and RRT), which relies on functionalities in the respective `<controller>/` folder. Please refer to `<controller>/README.md` for more detail. 
* `kernel.py` implements all the kernels, including those on environments (i.e. tasks) and controllers. A kernel keeps track of a random variable, and can change it via `propose()` and `revert()` to be called by the MCMC sampler. They also implement a `sample_prior()` function to sample the random variable from its distribution. All kernels are sub-classes of `TransitionKernel` class, which implements a dummy deterministic kernel with a value of constant 0 (useful for deterministic controllers). 
* `behavior.py` implements all the behavior functions (e.g., trajectory distance). Specifically, each behavior function takes two arguments, a trajectory and an environment instance. It will return two variables, `behavior_value` and `acceptable`. If the trajectory behavior is not "acceptable", this sample will be automatically discarded by the MCMC sampler. For example, if the behavior is only meaningfully defined for successful trajectories (e.g. trajectory length), then the `behavior_func` can set `acceptable = False` for unsuccessful trajectories. There is also a special behavior constructor, `neg_behavior`, that takes in a behavior function and outputs a behavior function with the negative values while still observing the `acceptable` flag. 
* `sampler.py` implements the core sampling functionality, with `sample()` implementing the MH algorithm. Parameters `env_kernel` and `controller_kernel` are instances of subclasses of the `TransitionKernel` class (defined in `kernel.py`). It is assumed that the controller randomness is independent from that of the environment, i.e. _p(u) = p(u | e)_. 
* `visualize_samples.py` visualizes either the trajectory or the obstacle distributions for prior and/or posterior samples. 
* `visualize_controller.py` visualizes the trajectory generated by a controller along with the obstacle configuration. You can run `python visualize_controller.py` and follow the prompt to visualize one of three controllers. It is also helpful for debugging new controllers. 

## Implementing a New Controller
If you wish to implement a custom controller, you can do so by inheriting the `Controller` class and implement all necessary functions. If your controller is stochatic, you also need to implement a custom controller kernel for it. You can do so by inheriting the `TransitionKernel` class in `kernel.py`. Please refer to its documentation for more information. 
