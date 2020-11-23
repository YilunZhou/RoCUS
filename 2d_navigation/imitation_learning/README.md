# Imitation Learning

The imitation learning policy is a neural network that predicts the heading angle given the robot's 2D current position and 16D lidar reading (in 16 equally spaced directions). 

If you want to train your own imitation learner, please follow the steps below. _Note that since the findings in the paper depend on stochasticity in model training (and perhaps also data generation), re-training the model may lead to different findings._

## Data Generation
In this directory, run `python generate_data.py` to generate data. You will be asked to confirm overwriting the current data file at `data_and_model/data.npz`. You can also change the file location in the script. 

## Model Training
In this directory, run `python train_model.py` to train the model on the data. You will be asked to confirm overwriting the current model file and training log at `data_and_model/{best.pt|train.log}`. You can also change these file locations in the script. 
