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

import matplotlib.pyplot as plt
import time

import yaml
with open("../../../../../config.yaml") as f:
    raw = yaml.safe_load(f)

start = time.time()

supervisor = Supervisor() # Collect the Supervisor node from the simulation
simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node


prefix = scenic.__file__[:-22]
scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                model="scenic.simulators.webots.model",
                                mode2D=False) # generate the scenario from the corresponding Scenic file



action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
array_size = 1 #find in simulator.py by ctrl f'ing array_size
observation_space = gym.spaces.Dict({
    "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
    "sensor": gym.spaces.Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), # defines the range of observations of the agent
    "position": gym.spaces.Box(low=np.array([-2.6, -2.6]), high=np.array([2.6, 2.6]), shape=(2,),dtype=np.float64),
})
max_steps = raw["supervisor"]["max_steps"]
env = ScenicGymEnv(scenario, 
                simulator, 
                render_mode=None, 
                max_steps=max_steps,  
                action_space=action_space,
                observation_space=observation_space) # max_step is max step for an episode - Create an enviroment instance
env = Monitor(env)

episodes= raw["supervisor"]["episodes"]
total_timesteps = max_steps * episodes
print(total_timesteps)

model = PPO("MultiInputPolicy", env, verbose=2, learning_rate=0.0002,ent_coef=0.05)
# Create an instance of an agent 
if(raw["supervisor"]["base_model"] != "na"):
    model.set_parameters(raw["supervisor"]["base_model"]) # Load the parameters of a previously trained agent
if(raw["supervisor"]["is_training"]):
    model.learn(total_timesteps=total_timesteps)          # train the agent over a set number of steps
    model.save(raw["supervisor"]["model_name"])            # Save the model after training

if(not raw["supervisor"]["is_training"]):
    mean_rwd, std_reward = evaluate_policy(model, env, n_eval_episodes=3,render=False, deterministic=False)
    print(f"After evaluation mean reward was : {mean_rwd} with std: {std_reward}")

env.env.logScores()

episodic_rewards = env.get_episode_rewards()
print(episodic_rewards)
total_pc = 0
if(len(episodic_rewards) >= 2):
    for i in range(1, len(episodic_rewards)):
        total_pc += (episodic_rewards[i] - episodic_rewards[i - 1]) / np.abs(episodic_rewards[i - 1])
    print("Average normalized percent difference: " + str(total_pc / (len(episodic_rewards) - 1)))

fig,ax = plt.subplots()

ax.stem(range(len(episodic_rewards)), episodic_rewards)

file_name = "PPO_policy" + str(total_timesteps)  + ".png"
plt.savefig(file_name,format='png')
plt.show()


end = time.time()

print(f" training time was {(end - start) / 60} minutes for {total_timesteps} timesteps")


