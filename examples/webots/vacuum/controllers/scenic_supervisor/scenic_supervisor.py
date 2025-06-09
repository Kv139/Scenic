from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.newtonian_gym import NewtonianSimulator
from scenic.simulators.webots import WebotsSimulator

import gymnasium as gym
import numpy as np

from controller import Supervisor

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,PPO


 
print("Begining Supervisor Script")
supervisor = Supervisor()
print("Supervisor node collected")
simulator = WebotsSimulator(supervisor)


prefix = scenic.__file__[:-22]
scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                   model="scenic.simulators.webots.model",
                                   mode2D=False)



action_space = gym.spaces.Box(low=0.0, high=1,shape=(2,))
observation_space = gym.spaces.Box(low=0, high=float('10000'),shape=(8,),dtype=np.float64)

env = ScenicGymEnv(scenario, 
                   simulator, 
                   render_mode=None, 
                   max_steps=1000, 
                   action_space=action_space,
                   observation_space=observation_space) # max_step is max step for an episode
#TODO FIX THIS -> WARNING: DEF IROBOT_CREATE Create > HingeJoint CREATE_LEFT_WHEEL > RotationalMotor "left wheel motor": The requested velocity 260.145 exceeds 'maxVelocity' = 16.129.

#check_env(env, skip_render_check=True)

model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=1000000)
model.save("vacuum_agent")




