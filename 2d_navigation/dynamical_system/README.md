# Dynamical System

The `Modulator` class in `modulation.py` contains core functionalities of the dynamical system controller.

`Modulator().set_arena()` needs to be called once for each environment, to compute the gamma functions. Then the modulated control is computed by `Modulator().modulate()`. 
