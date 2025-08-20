# DRL-RMBSA
There are codes for training and testing the DRL agent for dynamic service provisioning in multi-band elastic optical networks.

Run **DRL_RMBSA_Train.py** and **DRL_RMBSA_PCA_Train.py** to train the DRL agent using DRL-RMBSA and DRL-RMBSA-PCA algorithms.  
Run **DRL_Agent_Test.py** to test the performance of the DRL agent with the best models saved during training.  
Run **Heuristics_Test.py** to run heuristics including KSP-FB-FF, KSP-MinMaxF, and KSP-HCP-HMF.

Main python packages version used in this study is:

python 3.7.0
stable-baselines3 1.8.0  
sb3-contrib 1.7.0  
torch 1.12.0  
optical-rl-gym https://github.com/carlosnatalino/optical-rl-gym?tab=readme-ov-file  
gym 0.21.0

one tip to enhance the training stability is to set **next_non_terminal** to 1.0 all the time in  **compute_returns_and_advantage** (line 368-405) in **stable_baselines/common/buffers.py**, i.e., treat all next states as non-terminal during return and advantage computation. 

The channels/paths modulation format profile is generateed based on this reference https://ieeexplore.ieee.org/abstract/document/10892225, and stored in **NSFNET_modulation_table.mat**.

The data used for plotting in the paper is stored **file Results_NSFNET**.  

Please contact ab20471@bristol.ac.uk for any related question.
