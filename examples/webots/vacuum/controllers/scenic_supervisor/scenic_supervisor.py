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
import time
import gc

start = time.time()

supervisor = Supervisor() # Collect the Supervisor node from the simulation
simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node
prefix = scenic.__file__[:-22]


action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
observation_space = gym.spaces.Dict({
    "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
    "sensor": gym.spaces.Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), # defines the range of observations of the agent
    "position": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
    "rotation": gym.spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]), shape=(4,), dtype=np.float64)
})


scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                model="scenic.simulators.webots.model",
                                mode2D=False,
                                params={"is_couch":False})
max_steps = 1000
episodes  = 1
total_timesteps = max_steps * episodes

env = Monitor(ScenicGymEnv(scenario, 
                simulator, 
                render_mode=None, 
                max_steps=max_steps, 
                action_space=action_space,
                observation_space=observation_space)) # max_step is max step for an episode - Create an enviroment instance


model = PPO("MultiInputPolicy", env, verbose=2,seed=20,learning_rate=0.0002,ent_coef=0.05) # Create an instance of an agent 
model.learn(total_timesteps=total_timesteps)
model.save("PPO_vacuum_agent")

#env.close() avoid closing the env as it will destroy the simulator instance aswell
# Cleanup to avoid assertion error when instantiating new scenario
del model
del env
gc.collect()


scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                        model="scenic.simulators.webots.model",
                        mode2D=False,
                        params={"is_couch":True})

             
env = Monitor(ScenicGymEnv(scenario, 
        simulator, 
        render_mode=None, 
        max_steps=max_steps, 
        action_space=action_space,
        observation_space=observation_space))


model = PPO.load("PPO_vacuum_agent", env=env, verbose=2)
model.learn(total_timesteps=total_timesteps)
#model.save("PPO_vacuum_agent")
 
episodic_rewards = env.get_episode_rewards()
fig,ax = plt.subplots()

ax.stem(range(len(episodic_rewards)), episodic_rewards)

file_name = "../episode_rewards/PPO_policy_" + str(total_timesteps)  + ".png"
plt.savefig(file_name,format='png')
plt.show()
    

mean_rwd, std_reward = evaluate_policy(model, env, n_eval_episodes=3,render=False, deterministic=False)

print(f"After evaluation mean reward was : {mean_rwd} with std: {std_reward}") # Save the model after training

end = time.time()

print(f" training time was {(end - start) / 60} minutes for {total_timesteps} timesteps")

