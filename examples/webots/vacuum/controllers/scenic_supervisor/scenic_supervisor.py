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
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt



 
supervisor = Supervisor() # Collect the Supervisor node from the simulation
simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node


prefix = scenic.__file__[:-22]
scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                   model="scenic.simulators.webots.model",
                                   mode2D=False) # generate the scenario from the corresponding Scenic file



action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
observation_space = gym.spaces.Box(low=np.array([-1,-1,0,0,0,0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1,1,1,1,5.09,5.09]),shape=(12,),dtype=np.float64) # defines the range of observations of the agent
max_steps = 10000
env = ScenicGymEnv(scenario, 
                   simulator, 
                   render_mode=None, 
                   max_steps=max_steps, 
                   action_space=action_space,
                   observation_space=observation_space) # max_step is max step for an episode - Create an enviroment instance

env = Monitor(env)

episodes=80
total_timesteps = max_steps * episodes
print(total_timesteps)

model = PPO("MlpPolicy", env, verbose=2,device='cpu', n_steps=2048)  # Create an instance of an agent 
#model = PPO.load("PPO_vacuum_agent")
model.learn(total_timesteps=total_timesteps)          # train the agent over a set number of steps
model.save("PPO_vacuum_agent")                       # Save the model after training

mean_rwd, std_reward = evaluate_policy(model, env, n_eval_episodes=10,render=False)

print(f"After evaluation mean reward was : {mean_rwd} with std: {std_reward}")


episodic_rewards = env.get_episode_rewards

fig,ax = plt.subplots()

ax.scatter(len(episodic_rewards), episodic_rewards)

ax.set(xlim=(np.min(episodic_rewards+100)),
       ylim=(np.max(episodic_rewards+100)))
plt.show()
file_name = "MLP_policy" + str(total_timesteps)  + ".png"
plt.save(file_name,format='png')





