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
    "sectional_coverage": gym.spaces.Box(low=np.zeros(16), high=np.ones(16), shape=(16,),dtype=np.float64),
    "current_section": gym.spaces.Box(low=np.array([0]), high=np.array([15]), shape=(1,),dtype=int)
})
max_steps = 10000
env = ScenicGymEnv(scenario, 
                   simulator, 
                   render_mode=None, 
                   max_steps=max_steps, 
                   action_space=action_space,
                   observation_space=observation_space) # max_step is max step for an episode - Create an enviroment instance
env = Monitor(env)

episodes= 60
total_timesteps = max_steps * episodes
print(total_timesteps)

model = PPO("MultiInputPolicy", env, verbose=2) # Create an instance of an agent 
model.set_parameters("PPO_vacuum_agent")
model.learn(total_timesteps=total_timesteps)          # train the agent over a set number of steps
model.save("PPO_vacuum_agent")               # Save the model after training

mean_rwd, std_reward = evaluate_policy(model, env, n_eval_episodes=10,render=False)
print(f"After evaluation mean reward was : {mean_rwd} with std: {std_reward}")

episode_rewards = env.get_episode_rewards()
print(episode_rewards)
total_pc = 0
for i in range(1, len(episode_rewards)):
    total_pc += (episode_rewards[i] - episode_rewards[i - 1]) / np.abs(episode_rewards[i - 1])
print("Average normalized percent difference: " + str(total_pc / (len(episode_rewards) - 1)))

episodic_rewards = env.get_episode_rewards

fig,ax = plt.subplots()

ax.scatter(len(episodic_rewards), episodic_rewards)

ax.set(xlim=(np.min(episodic_rewards+100)),
       ylim=(np.max(episodic_rewards+100)))
plt.show()
file_name = "MLP_policy" + str(total_timesteps)  + ".png"
plt.save(file_name,format='png')





