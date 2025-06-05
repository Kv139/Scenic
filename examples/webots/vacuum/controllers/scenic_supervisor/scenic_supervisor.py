from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.newtonian_gym import NewtonianSimulator
from scenic.simulators.webots import WebotsSimulator

import gymnasium as gym
from stable_baselines3 import PPO
import os

from controller import Supervisor,robot

# Can use this and maybe do some synchronization with the robot controller
 
print("Begining Supervisor Script")
supervisor = Supervisor()
print("Supervisor node collected")
simulator = WebotsSimulator(supervisor)

# So maybe I can actually set some of the device controls here

prefix = scenic.__file__[:-22]
print(prefix)
#prefix = "\c:\Users\ksv14\Downloads\TEMP\test\.venv\scenic\src"

print(prefix + "examples/webots/vacuum/vacuum.scenic")
scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                   model="scenic.simulators.webots.model",
                                   mode2D=False)



action_space = gym.spaces.Box(low=0.0, high=16.129,shape=(2,))
observation_space = gym.spaces.Box(low=0, high=float('inf'),shape=(8,))

env = ScenicGymEnv(scenario, 
                   simulator, 
                   None, 
                   max_steps=100, 
                   action_space=action_space,
                   observation_space=observation_space) # max_step is max step for an episode

for _ in range(10):
    env.reset()
    env.step([0,0])
    episode_over = False
    while not episode_over:
        action = action_space.sample() 
        observation, reward, terminated, truncated, info = env.step(action=action)
        print(f"reward: {reward}")
        episode_over = terminated or truncated

env.close()


