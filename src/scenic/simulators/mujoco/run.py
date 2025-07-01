import scenic
from scenic.gym import ScenicGymEnv
from scenic.simulators.mujoco.simulator import MujocoSimulator
from scenic.core.scenarios import Scene

if __name__ == "__main__":
    SAMPLES = 1

    for sample_index in range(SAMPLES):
        simulator = MujocoSimulator(xml="")
        scenario = scenic.scenarioFromFile("simple.scenic")


        env = ScenicGymEnv(scenario, simulator, render_mode=None)
        env.reset()
        episode_over = False
        while not episode_over:
            action = env.action_space.sample() # dummy here

            observation, reward, terminated, truncated, info = env.step(action)
            print(observation)
            episode_over = terminated or truncated

        env.close()
