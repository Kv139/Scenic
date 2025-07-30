from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.webots import WebotsSimulatorGeneric

import gymnasium as gym
import numpy as np

from controller import Supervisor

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,PPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import time
import gc
import random

from control import generic_controller

scenic.setDebuggingOptions(fullBacktrace=True)

start = time.time()

supervisor = Supervisor() # Collect the Supervisor node from the simulation
print("starting")

simulator = WebotsSimulatorGeneric(supervisor) # Create an instance of the WebotsSImulator with the corresponding node

print("supervisor collected")
prefix = scenic.__file__[:-22]


action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
observation_space = gym.spaces.Dict({
    "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
    "sensor": gym.spaces.Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), # defines the range of observations of the agent
})


scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                model="scenic.simulators.webots.model",
                                mode2D=False,
                                params={})
max_steps = 10000
episodes  = 40
total_timesteps = max_steps * episodes


env = Monitor(ScenicGymEnv(scenario, 
                simulator, 
                render_mode=None, 
                max_steps=max_steps, 
                action_space=action_space,
                observation_space=observation_space)) # max_step is max step for an episode - Create an enviroment instance
observation, obs = env.reset()

budget = max_steps * episodes # total steps
action = [0,0] # init
episode_length = 0
controller = generic_controller()

for _ in range(budget):
    episodic_reward = 0
    action = controller.apply_control(state=observation) # sample new action

    assert action is not None

    for _ in range(3): # run each action for 3 timesteps
        observation, reward, terminated, truncated, info = env.env.step(action) # apply action for 5 timesteps
        episode_length += 1
        episodic_reward += reward
    
        if terminated or truncated:
            print(f"Steps in episode was : {episode_length}") # track length of episode
            print(f"Total reward gained was : {episodic_reward}")  # total cumulative rewards per episode
            env.reset()
            action = [0,0]
            episode_length = 0
            observation,reward,terminated,truncated,info = env.step(action)


episodic_rewards = env.get_episode_rewards()


end = time.time()

print(f"Generic training of {budget} steps finised in {start-end / 60} minutes")

