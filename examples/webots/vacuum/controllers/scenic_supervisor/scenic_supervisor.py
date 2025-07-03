from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.newtonian_gym import NewtonianSimulator
from scenic.simulators.webots import WebotsSimulator

import gymnasium as gym
import numpy as np

from controller import Supervisor

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,PPO
from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy



 
supervisor = Supervisor() # Collect the Supervisor node from the simulation
simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node


prefix = scenic.__file__[:-22]
scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                   model="scenic.simulators.webots.model",
                                   mode2D=False) # generate the scenario from the corresponding Scenic file



action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
array_size = 1 #find in simulator.py by ctrl f'ing array_size
observation_space = gym.spaces.Dict({
    "sensor": gym.spaces.Box(low=np.array([-1,-1,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), # defines the range of observations of the agent
    "cleaned": gym.spaces.MultiBinary((207, 207)) 
})
max_steps = 250
env = ScenicGymEnv(scenario, 
                   simulator, 
                   render_mode=None, 
                   max_steps=max_steps, 
                   action_space=action_space,
                   observation_space=observation_space) # max_step is max step for an episode - Create an enviroment instance
env = Monitor(env)

episodes= 5
total_timesteps = max_steps * episodes
print(total_timesteps)

model = PPO("MultiInputPolicy", env, verbose=2) # Create an instance of an agent 
model.set_parameters("PPO_vacuum_agent")
model.learn(total_timesteps=total_timesteps)          # train the agent over a set number of steps
#model.save("PPO_vacuum_agent")               # Save the model after training

episode_rewards = env.get_episode_rewards()
print(episode_rewards)
total_pc = 0
for i in range(1, len(episode_rewards)):
    total_pc += (episode_rewards[i] - episode_rewards[i - 1]) / np.abs(episode_rewards[i - 1])
print("Average normalized percent difference: " + str(total_pc / (len(episode_rewards) - 1)))
    
# print(rewards_per_step)
# print(env.get_episode_lengths())
# current_streak = 0
# epsilon = 0.01
# streak_threshold = 5
# converged = False
# for i in range(1, len(rewards_per_step)):
#     a, b = max(rewards_per_step[i], rewards_per_step[i - 1]), min(rewards_per_step[i], rewards_per_step[i - 1])
#     #ensure a is larger
#     if(b * (1 + epsilon) >= a):
#         current_streak += 1
#         if(current_streak >= streak_threshold):
#             print("Converged after " + str(i + 1) + " episodes.")
#             converged = True
#             break
#     else:
#         current_streak = 1
# if not converged:
#     print("Did not converge within training episodes")









