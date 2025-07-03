import math
from collections.abc import Callable
from typing import List

from scenic.core.object_types import Object
import dm_control 
from dm_control import mjcf
import numpy as np


from stable_baselines3 import PPO

import os
import mujoco

class MujocoBody(Object):
    """Abstract class for Mujoco objects."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.mjcf_model = None
        self.elements = {}

        if not hasattr(self, "xml"): # check before overwriting
            self.xml = ""

        if self.xml != "": 
            try:
                self.mjcf_model = mjcf.from_xml_string(self.xml)
            except ValueError as e:
                print(self.xml)
            self.body_name = self.mjcf_model.model + "/"
        else:
            self.mjcf_model = None  

    def model(self):
        return self.mjcf_model


class DynamicMujocoBody(MujocoBody):
    def __init__(self,  *args, **kwargs):
        super().__init__(args, **kwargs)
        self.xml = ""

    def control(self, model, data):
       raise NotImplementedError("Error: control not implemented for object")

class MujocoAgent(MujocoBody):
    "Special class for RL agent.."
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.agent = True     # allows for step based action
        self.xml = ""
        

class Pusher(MujocoAgent):
    """Copied from Robert's report - specific instance of a pusher bot """
    def __init__(self, *args, **kwargs):
        with open("./models/pusher.xml", "r") as f:
             self.xml = f.read()
        super().__init__( *args, **kwargs)
        root_path = os.getcwd() 

        self.observation = np.zeros(7, dtype=np.float32)
        self.joints = [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint"
        ]

        self.full_joint_name = [self.body_name + joint for joint in self.joints]

    def control_obs(self,data):
        joint_positions = []
        joint_velocities = []
        for name in self.full_joint_name:
            joint_positions.append(data.joint(name).qpos[0])
            joint_velocities.append(data.joint(name).qvel[0])
    
        # why are there these added zeros in the observation space
        observation = joint_positions + joint_velocities + [0 for i in range(9)] 

        # ctrl = self.controller.predict(observation) -- IDK 
        return observation

    def apply_control(self,data, action):

        for i, joint in enumerate(self.joints):
            full_joint_name = self.body_name + "/"  + f"unnamed_actuator_{i}"

            actuator = data.actuator(full_joint_name)

        #    print(action.shape)
            actuator.ctrl = action[i]