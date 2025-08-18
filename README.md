# DRL-RMBSA
There are codes for training and testing the DRL agent for dynamic service provisioning in multi-band elastic optical networks.

Run DRL_RMBSA_Train.py and DRL_RMBSA_PCA_Train.py to train the DRL agent using DRL-RMBSA and DRL-RMBSA-PCA algorithms, respectively.  
Run DRL_Agent_Test.py to test the performance of the DRL agent with the best models saved during training.

Required packages for building the agents are:

stable-baselines3 1.8.0  
sb3-contrib 1.7.0  
torch 1.12.0

Required package for building the multi-band environment is:
optical-rl-gym https://github.com/carlosnatalino/optical-rl-gym?tab=readme-ov-file

Please contact ab20471@bristol.ac.uk for any related question.
