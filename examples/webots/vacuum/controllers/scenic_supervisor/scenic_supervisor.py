from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.newtonian_gym import NewtonianSimulator
from scenic.simulators.webots import WebotsSimulator

import gymnasium as gym
import numpy as np

from controller import Supervisor

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,PPO


 
supervisor = Supervisor() # Collect the Supervisor node from the simulation
simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node


prefix = scenic.__file__[:-22]
scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                   model="scenic.simulators.webots.model",
                                   mode2D=False) # generate the scenario from the corresponding Scenic file



action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
observation_space = gym.spaces.Box(low=np.array([-1,-1,0,0,0,0]), high=np.array([1,1,1,1,1,1]),shape=(6,),dtype=np.float64) # defines the range of observations of the agent
max_steps = 1000
env = ScenicGymEnv(scenario, 
                   simulator, 
                   render_mode=None, 
                   max_steps=max_steps, 
                   action_space=action_space,
                   observation_space=observation_space) # max_step is max step for an episode - Create an enviroment instance

episodes=20
total_timesteps = max_steps * episodes
print(total_timesteps)

model = PPO("MlpPolicy", env, verbose=2) # Create an instance of an agent 
model.learn(total_timesteps=total_timesteps)          # train the agent over a set number of steps
model.save("vacuum_agent")               # Save the model after training








