# DRL-RMBSA
The repository contains the codes for training and testing the deep reinforcement learning (DRL) agent for dynamic service provisioning in multi-band elastic optical networks (MB-EONs). The DRL-based algorithms and heuristic algorithms are evaluated in a simulated MB-EON with NSFNET topology, using L+C+S band and without traffic grooming. For more details, please check this paper: https://ieeexplore.ieee.org/abstract/document/11131684.

Run **DRL_RMBSA_Train.py** and **DRL_RMBSA_PCA_Train.py** to train the DRL agent using DRL-RMBSA and DRL-RMBSA-PCA algorithms.  
Run **DRL_Agent_Test.py** to test the performance of the DRL agent with the best model saved during training.  
Run **Heuristics_Test.py** to run heuristics including KSP-FB-FF, KSP-MinMaxF, and KSP-HCP-HMF.

Main python packages version used in this study is:

python 3.7.0  
stable-baselines3 1.8.0  
sb3-contrib 1.7.0  
torch 1.12.0    
gym 0.21.0  
optical-rl-gym 0.0.2a0 (installation and more details: https://github.com/carlosnatalino/optical-rl-gym?tab=readme-ov-file)

one tip to enhance the training stability is to manually set **next_non_terminal = 1.0** in  function **compute_returns_and_advantage** (line 368-405) in **stable_baselines/common/buffers.py**, i.e., treat all next states as non-terminal during return and advantage computation. 

The channels/paths modulation format profile **NSFNET_modulation_table.mat** is generated based on the reference https://ieeexplore.ieee.org/abstract/document/10892225.

The data used for plotting in the paper is stored at **Results_NSFNET**.  

Please contact ab20471@bristol.ac.uk for any related question.
