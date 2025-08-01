from scenic.core.simulators import Simulator, Simulation, TerminationType
from scenic.core.scenarios import Scenario
from scenic.core.errors import setDebuggingOptions
import gymnasium as gym
from gymnasium import spaces
from typing import Callable

import random
import numpy as np

import pandas as pd
import os

setDebuggingOptions(verbosity=2)

#TODO make ResetException

file_path = "../../../../../output.csv"
point_file_path = "../../../../../point_outputs/"
base_buffer_path = "../../../../../buffers/"

def write_csv(name, coverage, collisions, discrete_collisions, rewards):
    rows = [[f"coverage_{name}"] + list(coverage),
            [f"collisions_{name}"] + list(collisions),
            [f"discrete collisions_{name}"] + list(discrete_collisions),
            [f"rewards_{name}"] + list(rewards)
            ]
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False, mode='a', header=False)
    
def write_point_records(name, episode, timewise_points):
    rows = [f"Episode_{episode}"] + list(timewise_points)
    df = pd.DataFrame(rows)
    df.to_csv(f"{point_file_path}{name}_points.csv", index=False, mode='a',header=False)
class ResetException(Exception):
    def __init__(self):
        super().__init__("Resetting")

class ScenicGymEnv(gym.Env):
    """
    verifai_sampler now not an argument added in here, but one specified int he Scenic program
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} # TODO placeholder, add simulator-specific entries
    
    def __init__(self, 
                 scenario : Scenario,
                 simulator : Simulator,
                 render_mode=None, 
                 max_steps = 1000,
                 observation_space : spaces.Dict = spaces.Dict(),
                 action_space : spaces.Dict = spaces.Dict(),
                 record_scenic_sim_results : bool = True,
                 feedback_fn : callable = lambda x: x,
                 raw = None
                 ): # empty string means just pure scenic???

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode
        self.max_steps = max_steps - 1 # FIXME, what was this about again?
        self.simulator = simulator
        self.scenario = scenario
        self.simulation_results = []

        self.feedback_result = None
        self.loop = None
        self.record_scenic_sim_results = record_scenic_sim_results
        self.feedback_fn = feedback_fn
        
        #CHANGEABLE STUFF, CHECK BEFORE EACH RUN ALSO CHECK THAT THE BUFFER IS EMPTY
        self.training_method = raw["training"]["training_method"]
        self.use_plr = raw["training"]["use_plr"]
        self.use_verifai = raw["training"]["use_verifai"]
        self.buffer_p = raw["training"]["buffer_p"] # probability of resampling
        self.truncate = raw["simulator"]["truncate"]
        #Random: default
        #LP: Prioritized level replay based off learning potential
        #EL: episode length, probability distribution based off inverse of episode length, need termination
        
        #changeable stuff for saving data to csv
        self.save_to_csv = raw["training"]["output_to_csv"] # whether to save data to csv
        self.record_points = raw["training"]["output_to_csv"]
        self.run_name = raw["supervisor"]["model_name"] # name of the run, used for saving data to csv
        self.total_steps = raw["supervisor"]["max_steps"] * raw["supervisor"]["episodes"] # total number of timesteps to run, used for saving data to csv
        
        self.is_testing = not raw["supervisor"]["is_training"]
        
        if self.is_testing:
            self.training_method = "Random"
            self.use_plr = False
            self.use_verifai = False
            self.save_to_csv = False 
            self.record_points = False
            
        if(self.use_plr):
            self.buffer_path = f"{base_buffer_path}{self.run_name}_buffer"
            os.makedirs(self.buffer_path, exist_ok=True)
            
        if(self.training_method == "EL"):
            self.truncate = True
        
        #load arrays
        self.resampling_weights =  np.array([]) # resamling weights of scenes in the buffer
        self.buffer_last_reward = np.array([]) # last reward of the scenes in the buffer(for lp)
        
        #extra variables for the run loop
        self.working_index = -1
        self.flag = 0
        self.counting_reward = 0
        self.steps_taken = 0
        self.total_steps_taken = 0
        
        if self.use_plr and self.training_method not in ("EL", "LP"):
            raise ValueError(
                f"use_plr=True but training_method={self.training_method!r}. "
                "Must be one of 'LP' or 'EL' if use_plr is enabled."
            )
        
        #information recording
        self.episode_coverages = []  
        self.episode_collisions = []  
        self.episode_discrete_collisions = []
        self.episode_rewards = []
        self.timewise_points = None

    def _make_run_loop(self):
        while True:
            try:
                print(f"Training using {self.training_method}")
                scene = None
                self.is_resampling = random.uniform(0, 1) < self.buffer_p
                #sample or resample scenes
                if not self.use_plr and not self.use_verifai:
                    scene, _ = self.scenario.generate(feedback=self.feedback_result)
                elif self.training_method == "LP" and len(self.resampling_weights) != len(self.buffer_last_reward):
                    # resample scene
                    self.flag = 0
                    self.working_index = len(self.buffer_last_reward) - 1
                    with open(f"{self.buffer_path}/scene_{self.working_index}.bin", "rb") as f:
                        scene = self.scenario.sceneFromBytes(f.read())
                    print("Double sampling")
                elif self.use_plr and self.is_resampling and len(self.resampling_weights) > 0:
                    # resample from buffer
                    self.flag = 1
                    prob_distribution = self.resampling_weights / np.sum(self.resampling_weights)
                    self.working_index = np.random.choice(len(self.resampling_weights), p=prob_distribution)
                    with open(f"{self.buffer_path}/scene_{self.working_index}.bin", "rb") as f:
                        scene = self.scenario.sceneFromBytes(f.read())
                    print(f"Resampling from buffer with index {self.working_index}")
                else:
                    # sample new scene
                    self.flag = 2
                    scene, _ = self.scenario.generate(feedback=self.feedback_result)
                    with open(f"{self.buffer_path}/scene_{len(self.resampling_weights)}.bin", "wb") as f:
                        f.write(self.scenario.sceneToBytes(scene=scene))
                    self.working_index = len(self.resampling_weights)
                    print(f"Sampling new scene with index {self.working_index}")
                
                self.counting_reward = 0
                with self.simulator.simulateStepped(scene, maxSteps=self.max_steps) as simulation:
                    self.steps_taken = 0
                    # this first block before the while loop is for the first reset call
                    done = lambda: not (simulation.result is None) or (simulation.get_truncation() and self.truncate) # allows for early truncation
                    truncated = lambda: (self.steps_taken >= self.max_steps)  # TODO handle cases where it is done right on maxsteps
                    observation = simulation.get_obs()
                    info = simulation.get_info() 
                    actions = yield observation, info
                    simulation.actions = actions # TODO add action dict to simulation interfaces
                    while not done():
                        # Probably good that we advance first before any action is set.
                        # this is consistent with how reset works
                        simulation.advance()
                        self.steps_taken += 1
                        self.total_steps_taken += 1
                        observation = simulation.get_obs()
                        info = simulation.get_info()
                        reward = simulation.get_reward()
                        self.counting_reward += reward
                        if simulation.covered_spaces != None and simulation.coverage_timesteps != None:
                            self.timewise_points = list(zip(simulation.coverage_timesteps, simulation.covered_spaces))
                        if done():
                            if simulation.result is None:
                                simulation.terminateSimulation(TerminationType.terminatedByUser, "early truncation")
                            print("Simulation done")
                            if self.record_points:
                                write_point_records(self.run_name, len(self.episode_collisions) + 1, self.timewise_points)
                            self.logScores()
                            if not self.use_verifai:
                                self.feedback_result = self.feedback_fn(simulation.result)
                            if self.record_scenic_sim_results:
                                self.simulation_results.append(simulation.result)
                            # simulation.destroy() # FIXME...might redundant?
                            actions = yield observation, reward, done(), truncated(), info
                            
                            break # a little unclean right here

                        actions = yield observation, reward, done(), done(), info
                        simulation.actions = actions # TODO add action dict to simulation interfaces
                    print("Exitedwhile loop")
            except ResetException:
                if self.total_steps_taken >= self.total_steps and self.save_to_csv:
                    write_csv(self.run_name, self.episode_coverages, self.episode_collisions, self.episode_discrete_collisions, self.episode_rewards)
                print("reset exception caught")
                print(f"Episode coverages: {self.episode_coverages}")
                print(f"Mean and std of coverages: {np.mean(self.episode_coverages)} and {np.std(self.episode_coverages)}")
                print(f"Episode collisions: {self.episode_collisions}")
                print(f"Mean and std of collisions: {np.mean(self.episode_collisions)} and {np.std(self.episode_collisions)}")
                print(f"Episode discrete collisions: {self.episode_discrete_collisions}")
                print(f"Mean and std of discrete collisions: {np.mean(self.episode_discrete_collisions)} and {np.std(self.episode_discrete_collisions)}")
                print(f"Excel splittable: {np.mean(self.episode_coverages)},{np.std(self.episode_coverages)},{np.mean(self.episode_collisions)},{np.std(self.episode_collisions)},{np.mean(self.episode_discrete_collisions)},{np.std(self.episode_discrete_collisions)}")
                continue

    def reset(self, seed=None, options=None): # TODO will setting seed here conflict with VerifAI's setting of seed?
        # only setting enviornment seed, not torch seed?
        super().reset(seed=seed)
        if self.loop is None:
            print("self loop doesnt exist, creating new one")
            self.loop = self._make_run_loop()
            observation, info = next(self.loop) # not doing self.scene.send(action) just yet
        else:
            observation, info = self.loop.throw(ResetException())


        return observation, info
        
    def step(self, action):
        assert not (self.loop is None), "self.loop is None, have you called reset()?"

        observation, reward, terminated, truncated, info = self.loop.send(action)
        
        if terminated or truncated:
            self.episode_coverages.append(info.get("coverage", 0))
            self.episode_collisions.append(info.get("collisions", 0))
            self.episode_discrete_collisions.append(info.get("discrete_collisions", 0))
            self.episode_rewards.append(self.counting_reward)
        
        return observation, reward, terminated, truncated, info
    
    def get_coverage(self):
        return sum(self.episode_coverages) / len(self.episode_coverages) if self.episode_coverages else 0 

    def render(self): # TODO figure out if this function has to be implemented here or if super() has default implementation
        """
        likely just going to be something like simulation.render() or something
        """
        # FIXME for one project only...also a bit hacky...
        # self.env.render()
        pass

    def close(self):
        self.simulator.destroy()
        
    def logScores(self):
        if self.training_method == "Random":
            return
        if self.training_method != "EL" and self.training_method != "LP":
            return
        total_reward = self.counting_reward
        if(total_reward == 0):
            print("TOTAL REWRAD is 0! suspiciosu!")
        #log rewards and learning potential
        if self.flag == 0:
            #double sampling, so we know we are using LP
            if(self.working_index >= len(self.buffer_last_reward)):
                print(f"Warning: working index {self.working_index} is out of bounds for buffer_last_reward with length {len(self.buffer_last_reward)}")
            else:
                lp = abs(total_reward - self.buffer_last_reward[self.working_index]) + 1e-8
                if self.use_verifai:
                    self.feedback_result = -lp
                self.resampling_weights = np.append(self.resampling_weights, lp)
                self.buffer_last_reward[self.working_index] = total_reward
                print("finished double sampling")
        elif self.flag == 1:
            #resampling
            if self.training_method == "LP":
                lp = abs(total_reward - self.buffer_last_reward[self.working_index]) + 1e-8
                if self.use_verifai:
                    self.feedback_result = -lp
                self.resampling_weights[self.working_index] = lp
                self.buffer_last_reward[self.working_index] = total_reward
            elif self.training_method == "EL":
                inverse_reward = 1 / (self.steps_taken + 100)
                if(self.steps_taken <= 50):
                    inverse_reward = 0
                self.resampling_weights[self.working_index] = inverse_reward
            else:
                print("BIG ISSUE, resmaled but not special PLR")
        else:
            if self.training_method == "LP":
                self.buffer_last_reward = np.append(self.buffer_last_reward, total_reward)
            elif self.training_method == "EL":
                inverse_reward = 1 / (self.steps_taken + 100)
                if(self.steps_taken <= 50):
                    inverse_reward = 0
                self.resampling_weights = np.append(self.resampling_weights, inverse_reward)