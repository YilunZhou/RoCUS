# Reinforcement Learning

`ppo.py` implements the proximal policy gradient (PPO) algorithm for the `PandaReacher-v0` environment. If you wish to train your own model, simply run `python ppo.py` and confirm to overwrite the existing `progress.log` and `best.pt`. The training and test rewards will be written to `progress.log`, and the best model will be saved to `best.pt`. In addition, `iter_N.pt` will also be saved every 1000 iterations, with `N = 1000, 2000, ...`. 
