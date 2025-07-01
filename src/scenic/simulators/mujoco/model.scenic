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

        xml = args[0]["xml"] if "xml" in args[0] else None
        self.mjcf_model = None
        self.elements = {}
        if xml:
            try:
                self.mjcf_model = mjcf.from_xml_string(xml)
            except ValueError as e:
                print(xml)
            self.body_name = self.mjcf_model.model + "/"
        else:
            self.mjcf_model = None    
    def model(self):
        return self.mjcf_model


class DynamicMujocoBody(MujocoBody):
    """Dynamic Mujoco Body"""
    def __init__(self, xml: str="", *args, **kwargs):
        super().__init__(xml, *args, **kwargs)

    def control(self, model, data):
        raise NotImplementedError("Error: control not implemented for object")


class Pusher(DynamicMujocoBody):
    """Copied from Robert's report """

    def __init__(self, xml: str="", sbe_model: str=None, *args, **kwargs):
        super().__init__(xml, *args, **kwargs)
        root_path = os.getcwd()

        if not sb3_model:
            sb3_model = "PPO_pusher.zip"

        self.controller = PPO.load(os.path.join(root_path, sb3_model), device="cpu")

    def control(self,model,data):
        joints = [
            "r_shoulder_pan_joint",
            "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint",
            "r_elbow_flex_joint",
            "r_wrist_flex_joint",
            "r_wrist_roll_joint"
        ]

        joint_positions = []
        joint_velocities = []


        for joint in joints:
            full_joint_name = self.body_name + joint
            joint_positions.append(data.joint(full_joint_name).qpos[0])
            joint_velocities.append(data.joint(full_joint_name).qvel[0])
        

        observation = joint_positions + joint_velocities + [0 for i in range(9)]

        ctrl = self.controller.predict(observation)


        for i, joint in enumerate(joints):
            full_joint_name = self.body_name + "/"  + f"unnamed_actuator_{i}"

            actuator = data.actuator(full_joint_name)

            actuator.ctrl = ctrl[0][i]