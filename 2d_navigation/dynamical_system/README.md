# Dynamical System

The `Modulator` class in `modulation.py` contains core functionalities of the modulation. `Modulator().get_modulated_direction()` returns the direction of motion in the task space. 

The `IK` class in `ik.py` is responsible for converting the task space motion to joint space. `IK().__call__()` returns the joint space configuration for a given task space target.  
