# Rapidly-Exploring Random Tree

The `RRT` class in `rrt.py` contains core functionalities of the RRT controller. 

* `RRT().get_path()` implements the RRT algorithm (Algorithm 1 in the paper). If `controller_kernel` parameter of `RRT().get_path()` is provided, it functions as the "hypothetically infinite tape of random configurations" and the function behavior is deterministic. Otherwise, the function samples random configurations and the behavior is stochastic. Collision checking resolution is specified by `collision_res`. 
* `RRT().increase_resolution()` takes the path from the previous function and segments into shorter sections with each section length no more than the specified `control_res`. 

Please refer to Section V of the paper for more details. 
