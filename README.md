Anonymized Codebase for Multiagent Evolutionary Reinforcement Learning (MERL)

#################################
          Code labels
#################################

train.py: Training Script that runs training

core/runner.py: Rollout worker

core/agent.py: Agent encapsulating the learning algorithm, population, and replay buffer for an agent

core/buffer.py: Cyclic Replay buffer

envs/env_wrapper.py: Wrapper around the RoverDomain 

core/models.py: Neural Network Models - Q and Multi-head Actor

core/neuroevolution.py: Implements Sub-Structured Based Neuroevolution (SSNE) 

core/off_policy_algo.py: Implements the off_policy_gradient learner TD3/DDPG/MATD3/MADDPG

core/mod_utils.py: Helper functions

envs/rover_domain: ROver Domain Environment files


################################################################# 
Guide to set up env
################################################################# 

1. Setup Conda
    - Install Anaconda3
    - conda create -n $ENV_NAME$ python=3.6.1
    - source activate $ENV_NAME$

2. Install Pytorch version 1.0
    - Refer to https://pytorch.org/ for instructions
    - conda install pytorch torchvision -c pytorch [GPU-version]

3. Install Numpy, Cython, Scipy, and Matplotlib
    - pip install numpy==1.15.4
    - pip install cython==0.29.2
    - pip install scipy==1.1.0
    - pip install matplotlib
    

################################################################# 
Guide to run experiments for Rover
################################################################# 

1. MERL --> python train.py -popsize 10 -rollsize 50 -config $DESIRED_CONFIG$ -frames 2 -seed $SEED$
2. EA --> python train.py -popsize 10 -rollsize 0 -config $DESIRED_CONFIG$ -frames 2 -seed $SEED$
3. MADDPG_global --> python train.py -popsize 0 -rollsize 50 -config $DESIRED_CONFIG$ -frames 2 -seed $SEED$ -maddpg 1 -reward global
4. MADDPG_mixed --> python train.py -popsize 0 -rollsize 50 -config $DESIRED_CONFIG$ -frames 2 -seed $SEED$ -maddpg 1 -reward mixed
5. MATD3_global --> python train.py -popsize 0 -rollsize 50 -config $DESIRED_CONFIG$ -frames 2 -seed $SEED$ -matd3 1 -reward global
6. MATD3_mixed --> python train.py -popsize 0 -rollsize 50 -config $DESIRED_CONFIG$ -frames 2 -seed $SEED$ -matd3 1 -reward mixed

CONFIGS AND SEED USED IN THE PAPER 

DESIRED CONFIGS = {3_1, 4_2, 6_3, 8_4, 10_5, 12_6, 14_7} 

SEED = {2019, 2020, 2021, 2022, 2023} 


################################################################# 
Guide to run experiments for Predator-prey, Keep-away and Physical Deception
################################################################# 
Note that the code for the adversarial domains were a different branch and are thus copied into a separate folder for simplicity.
To run predator-prey or cooperative navigation, browse into their respective folders and run the following:


1. MERL --> python train.py -popsize 10 -rollsize 10 -config $DESIRED_CONFIG$ -frames N -seed $SEED$
2. EA --> python train.py -popsize 10 -rollsize 0 -config $DESIRED_CONFIG$ -frames N -seed $SEED$
3. MADDPG --> python train.py -popsize 0 -rollsize 50 -config $DESIRED_CONFIG$ -frames N -seed $SEED$ -maddpg 1 
4. MATD3 --> python train.py -popsize 0 -rollsize 50 -config $DESIRED_CONFIG$ -frames N -seed $SEED$ -matd3 1 


CONFIGS AND SEED USED IN THE PAPER 

DESIRED CONFIGS = {simple_tag, hard_tag, simple_push and simple_adversary} representing easy predator-prey, hard predator-prey, keep-away and physical deception respectively. 

SEED = {2018, 2019, 2020, 2021, 2022} 