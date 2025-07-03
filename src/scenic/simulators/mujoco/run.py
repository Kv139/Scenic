import scenic
from scenic.gym import ScenicGymEnv
from scenic.simulators.mujoco.simulator import MujocoSimulator
from scenic.core.scenarios import Scene

import numpy as np
import gymnasium as gym

if __name__ == "__main__":

    observation_space = gym.spaces.Box(-np.inf, np.inf, (23,), np.float64)
    action_space      = gym.spaces.Box(-2.0, 2.0, (7,), np.float64)


    simulator = MujocoSimulator(xml="",use_default_arena=False)
    scenario = scenic.scenarioFromFile("simple.scenic")

    env = ScenicGymEnv(scenario, simulator, render_mode="human", observation_space=observation_space, action_space=action_space,max_steps=100000)
    env.reset()
    episode_over = False
    while not episode_over:
        action = env.action_space.sample() # dummy here

       # print(action)

        observation, reward, terminated, truncated, info = env.step(action)
        #print(observation[:12])
        episode_over = terminated or truncated

    env.close()
