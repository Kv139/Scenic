from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.newtonian_gym import NewtonianSimulator
from scenic.simulators.webots import WebotsSimulator
import numpy as np
import gymnasium as gym
import os

from controller import Supervisor

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

print("Creating env")
env = ScenicGymEnv(scenario, simulator, None, max_steps=100,) # max_step is max step for an episode
print("Env was created")
env.reset()
print("Env Reset")
episode_over = False

while not episode_over:

    v1 = np.random.randint(0,5)
    v2 = np.random.randint(0,5)
    action = [v1,v2]
    # edit
    
    observation, reward, terminated, truncated, info = env.step(action=action)
    print(observation)
    episode_over = terminated or truncated

env.close()

