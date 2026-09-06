"""Simulator interface for ManiSkill."""

import math
import os
import tempfile
from typing import Optional

import numpy as np

from scenic.core.simulators import Simulation, Simulator
from scenic.core.utils import buildingScenicDocumentation
from scenic.core.vectors import Orientation, Vector

try:
    import mani_skill
except ImportError as e:
    raise ModuleNotFoundError(
        "ManiSkill scenarios require the optional ManiSkill dependencies. "
        "Install with `pip install scenic[maniskill]`."
    ) from e

import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
import sapien
import torch

# Default camera configuration
DEFAULT_CAMERA_EYE = [0.0, -1.0, 1.0]
DEFAULT_CAMERA_TARGET = [0.0, 0.0, 0.0]
DEFAULT_ROBOT_POSITION = [0.0, 0.0, 0.0]


class ScenicEnv(BaseEnv):
    """Custom ManiSkill environment for Scenic scenarios."""

    def __init__(
        self,
        *args,
        robot_uids="none",
        objects_to_create=[],
        camera_configs=None,
        shader_pack="default",
        **kwargs,
    ):
        self.robot_uids = robot_uids
        self.scenic_objects = {}
        self.objects_to_create = objects_to_create
        self.scene_camera_configs = camera_configs
        self.shader_pack = shader_pack
        self.obj_name_ids = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        """Return camera configurations for the scene.

        Uses scene-defined cameras if available, otherwise creates a default camera.
        """
        if self.scene_camera_configs:
            return self.scene_camera_configs

        # Fallback to default camera configuration
        pose = sapien_utils.look_at(eye=DEFAULT_CAMERA_EYE, target=DEFAULT_CAMERA_TARGET)
        return [
            CameraConfig(
                "base_camera",
                pose,
                width=640,
                height=480,
                fov=1.0,
                near=0.01,
                far=100.0,
                shader_pack=self.shader_pack,
            )
        ]

    def _load_agent(self, options: dict, initial_agent_poses=None, build_separate=False):
        """Load the robot agent into the scene."""
        super()._load_agent(options, sapien.Pose(p=DEFAULT_ROBOT_POSITION))

    def _get_obs_agent(self):
        """Return agent observation data.

        For scenes without a robot, return an empty dict instead of
        trying to read proprioception from a nonexistent agent.
        """
        if self.agent is None:
            return {}
        # obs = super()._get_obs_agent()
        return super()._get_obs_agent()
    
    def get_name_id(self,obj):
        if obj not in self.obj_name_ids:
            id = len(self.obj_name_ids) + 1

        self.obj_name_ids[obj] = id
        return id

    def set_scenic_scene(self, objects_to_create, camera_configs):
        """Swap in the contents of a new Scenic scene.

        The actors themselves are (re)built the next time the environment is
        reconfigured, i.e. on the next ``reset(options={"reconfigure": True})``.
        This lets a single environment (and hence a single GPU/renderer context)
        be reused across many simulations.
        """
        self.objects_to_create = objects_to_create
        self.scene_camera_configs = camera_configs
        self.scenic_objects = {}
        self.obj_name_ids = {}

    def get_state_dict(self):
        """Return environment state.

        For scenes without a robot, return only the scene simulation state
        instead of trying to read controller state from a nonexistent agent.
        """
        if self.agent is None:
            return self.scene.get_sim_state()
        return super().get_state_dict()

    def build_actor_from_trimesh(self, scene, obj, mesh):
        """Build a SAPIEN actor from a trimesh object."""
        if getattr(obj, "skipGeneration", False):
            return None

        if getattr(obj, "isSimpleGround", False):
            return build_ground(
                scene,
                floor_width=int(obj.width),
                floor_length=int(obj.length),
                xy_origin=(obj.position[0], obj.position[1]),
                altitude=obj.position[2],
                name=obj.name,
            )

        # Use a temporary directory for mesh files
        if not hasattr(obj, "name"):
            obj.name = f"scenic_maniskill_default_name{self.get_name_id(obj)}_mesh.obj"
        tmp_dir = tempfile.gettempdir()
        obj_filename = os.path.join(tmp_dir, f"scenic_maniskill_{obj.name}_mesh.obj")
        mesh.export(obj_filename)

        builder = scene.create_actor_builder()

        width = getattr(obj, "width", 1.0)
        length = getattr(obj, "length", 1.0)
        height = getattr(obj, "height", 1.0)

        if obj.name == "table": # Special hanndling for teh tables collision box
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, height / 2]),
                half_size=(width / 2, length / 2, height / 2),
            )

        else:
            builder.add_multiple_convex_collisions_from_file(
            filename=obj_filename,
            pose=sapien.Pose(),
            scale=(width,length, height),
            decomposition="coacd",
        )

        if hasattr(obj, "visualFilename") and obj.visualFilename is not None:
            visualOffset = getattr(obj, "visualOffset", (0.0, 0.0, 0.0))
            visualScale = getattr(obj, "visualScale", (1.0, 1.0, 1.0))

            builder.add_visual_from_file(
                filename=obj.visualFilename,
                pose=sapien.Pose(p=[visualOffset[0], visualOffset[1], visualOffset[2]]),
                scale=visualScale,
            )
        else:
            color = getattr(obj, "color", [0.8, 0.8, 0.8, 1.0])

            builder.add_visual_from_file(
                filename=obj_filename,
                pose=sapien.Pose(),
                scale=(width, length, height),
                material=sapien.render.RenderMaterial(
                    base_color=color,
                ),
            )

        builder.set_name(obj.name)
        builder.set_initial_pose(
            sapien.Pose([obj.position[0], obj.position[1], obj.position[2]])
        )

        if getattr(obj, "kinematic", False):
            actor = builder.build_kinematic(name=obj.name)
        else:
            actor = builder.build(name=obj.name)

        if obj.name == "cubeA":
            self.cubeA = actor
        if obj.name == "cubeB":
            self.cubeB = actor

        return actor

    def _load_scene(self, options: dict):
        # Actors from any previous scene were destroyed by the reconfiguration,
        # so start from a clean mapping.
        self.scenic_objects = {}
        for obj in self.objects_to_create:
            mesh = obj.shape.mesh
            self.scenic_objects[obj.name] = self.build_actor_from_trimesh(
                self.scene, obj, mesh
            )

    def _initialize_episode(self, env_idx, options: dict):
        """Initialize the robot configuration for the episode.

        Sets the robot's joint angles from the Scenic scene if provided,
        otherwise uses the robot's default configuration.
        """
        # No robots in the scene
        if self.agent is None:
            return

        # Find the robot object in the scene
        robot_objects = [
            obj
            for obj in self.objects_to_create
            if hasattr(obj, "uuid") and obj.uuid == "panda"
        ]

        if not robot_objects:
            # No robot configuration specified, use default
            return

        robot = robot_objects[0]

        if (
            hasattr(robot, "jointAngles")
            and robot.jointAngles
            and len(robot.jointAngles) > 0
        ):
            # Use user-specified joint angles
            qpos = np.array(robot.jointAngles, dtype=np.float32)

            # Validate joint angles match robot DOF
            robot_dof = self.agent.robot.dof
            if len(qpos) != robot_dof:
                raise ValueError(
                    f"Joint angles length ({len(qpos)}) does not match robot DOF ({robot_dof}). "
                    f"For Panda robot, provide {robot_dof} joint angles."
                )

            qpos = torch.from_numpy(qpos)
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose(DEFAULT_ROBOT_POSITION))
        # If no joint angles specified, the agent will use its default configuration


# Register the ManiSkill environment in normal runs, but skip this during doc builds.
if not buildingScenicDocumentation():
    ScenicEnv = register_env("ScenicEnv")(ScenicEnv)


class ManiSkillSimulator(Simulator):
    """Simulator interface for ManiSkill.

    The underlying ManiSkill environment is created once and reused by every
    `ManiSkillSimulation` this simulator creates: each new simulation swaps in
    its own objects/cameras and reconfigures the existing environment rather
    than building a fresh one. This avoids leaking a GPU/renderer context per
    episode, which is what makes repeated resets during RL training fall over.
    """

    def __init__(self, render=False, stride=1, shader_pack="default", obs_mode="rgb"):
        super().__init__()
        self.last_simulation = None
        self.render = render
        self.stride = stride
        self.shader_pack = shader_pack
        self.obs_mode = obs_mode
        self.env: Optional[gym.Env] = None
        # Environment settings that are fixed at construction time; if a scene
        # needs different ones we have no choice but to rebuild the environment.
        self._env_key = None


    def closeEnvironment(self):
        """Close the shared environment, if one has been created."""
        if self.env is not None:
            self.env.close()
            self.env = None
            self._env_key = None

    def createSimulation(self, scene, **kwargs):
        self.last_simulation = ManiSkillSimulation(
            scene,
            simulator=self,
            render=self.render,
            stride=self.stride,
            shader_pack=self.shader_pack,
            **kwargs,
        )
        return self.last_simulation

    def destroy(self):
        self.closeEnvironment()
        super().destroy()


class ManiSkillSimulation(Simulation):
    """Simulation interface for ManiSkill."""

    def __init__(
        self,
        scene,
        *,
        simulator=None,
        timestep=None,
        render=False,
        stride=1,
        shader_pack="default",
        **kwargs,
    ):
        # The simulator owns the ManiSkill environment; we only borrow it.
        self.maniskill_simulator = simulator
        self.env: Optional[gym.Env] = None
        self._done = False
        self._obs = torch.zeros(1, 8)
        self._info = None
        self._reward = None
        self.action = None
        self.maniskill_scene = scene
        self.objects_to_create = []
        self.camera_configs = []
        self.stride = stride
        self.stride_idx = 0
        self.step_count = 0
        self.shader_pack = shader_pack

        # Should it be headless (false) or render (true)?
        self.render = render

        super().__init__(scene, timestep=(timestep or 1 / 60), **kwargs)

    def extract_robot_uid(self):
        robot_objects = [
            obj for obj in self.maniskill_scene.objects if hasattr(obj, "uuid")
        ]

        if len(robot_objects) > 1:
            raise RuntimeError("Multiple robots are not supported yet.")

        if not robot_objects:
            return "none"

        return robot_objects[0].uuid

    def setup(self):
        # Calls the createObjectInSimulator method for each object in the scene
        super().setup()

        self.extract_camera_configs()

        robot_uid = self.extract_robot_uid()

        if self.maniskill_simulator is None:
            raise RuntimeError(
                "ManiSkillSimulation must be created by a ManiSkillSimulator, "
                "which owns the underlying ManiSkill environment."
            )

        # Reuse the simulator's environment; it is only built on the first call.
        self.env, reconfigure = self.maniskill_simulator.getEnvironment(
            robot_uid, self.objects_to_create, self.camera_configs
        )

        self.step_count = 0
        # Reconfiguring rebuilds the scene's actors from this simulation's
        # objects without tearing down the environment itself.
        options = {"reconfigure": True} if reconfigure else None
        self._obs, self._info = self.env.reset(seed=0, options=options)

    def extract_camera_configs(self):
        """Extract camera configurations from Scenic objects.

        Converts Scenic Camera objects to ManiSkill CameraConfig objects.
        Uses the camera's orientation to determine the look-at target.
        """
        for obj in self.maniskill_scene.objects:
            if hasattr(obj, "__class__") and obj.__class__.__name__ == "Camera":
                # Convert scenic camera to ManiSkill CameraConfig
                eye_pos = [obj.position.x, obj.position.y, obj.position.z]

                # Calculate target position from camera orientation
                # Use the camera's forward direction (1 unit ahead)
                if hasattr(obj, "heading"):
                    # For 2D heading, assume camera looks horizontally
                    target_x = obj.position.x + math.cos(obj.heading)
                    target_y = obj.position.y + math.sin(obj.heading)
                    target_z = obj.position.z
                    target_pos = [target_x, target_y, target_z]
                else:
                    # Fallback: look at origin
                    target_pos = [0.0, 0.0, 0.0]

                pose = sapien_utils.look_at(eye=eye_pos, target=target_pos)

                camera_config = CameraConfig(
                    obj.name,
                    pose,
                    width=getattr(obj, "img_width", 640),
                    height=getattr(obj, "img_height", 480),
                    fov=getattr(obj, "fov", 1.0),
                    near=getattr(obj, "near", 0.01),
                    far=getattr(obj, "far", 100.0),
                    shader_pack=getattr(obj, "shader_pack", self.shader_pack),
                )

                self.camera_configs.append(camera_config)

    def createObjectInSimulator(self, obj):
        # Skip camera objects as they're handled separately
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "Camera":
            return
        self.objects_to_create.append(obj)

    def step(self):
        if self.should_terminate():
            return

        if self.env is None:
            raise RuntimeError("Environment not set up. Call setup() first.")

        if self.actions is None:
            if self.env.action_space is None:
                self.actions = None
            else:
                self.actions = self.env.action_space.sample() * 0.0

        obs, reward, term, trunc, info = self.env.step(self.actions)
        self._obs = obs
        self.step_count += 1

        if self.stride_idx == 0:
            self.action = self.get_action_from_observation(obs)

        self._reward = reward
        self._done = bool(term or trunc)
        self.stride_idx = (self.stride_idx + 1) % self.stride

        if self.render:
            viewer = self.env.unwrapped.viewer

            # If the GUI window has already been closed, stop the simulation cleanly.
            if viewer is not None and viewer.closed:
                self.screen = "Dead"
                return

            self.env.render()

        # Check for termination condition
        if self.should_terminate():
            self._done = True

    def should_terminate(self):
        """Override this method to implement custom termination logic."""
        return False

    def get_action_from_observation(self, obs):
        """Override this method to implement custom action selection."""
        if self.env.action_space is None:
            return None
        return self.env.action_space.sample() * 0.0

    def success(self):
        return self.success

    def get_step_count(self):
        return self.step_count

    def get_obs(self):
        return self._obs

    def get_info(self):
        return self._info

    def sampler_feedback(self):
        return None

    def tensor_to_vector(self, tensor):
        if tensor.dim() != 1 or tensor.size(0) != 3:
            raise ValueError("Tensor must be 1D with 3 elements.")
        return Vector(tensor[0].item(), tensor[1].item(), tensor[2].item())

    def getProperties(self, obj, properties):
        if self.env is None:
            raise ValueError("Environment is not initialized.")

        ms_obj = self.env.unwrapped.scenic_objects[obj.name]

        if getattr(obj, "skipGeneration", False):
            return dict(
                position=Vector(10, 0, 0),
                velocity=Vector(0, 0, 0),
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                speed=0.0,
                angularVelocity=Vector(0, 0, 0),
                angularSpeed=0.0,
            )

        position = self.tensor_to_vector(ms_obj.pose.p[0])
        if getattr(obj, "kinematic", False):
            velocity = Vector(0, 0, 0)
        else:
            velocity = self.tensor_to_vector(ms_obj.get_linear_velocity()[0])
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        q = ms_obj.pose.q[0]
        orientation = Orientation.fromQuaternion((q[0], q[1], q[2], q[3]))
        if getattr(obj, "kinematic", False):
            angular_velocity = Vector(0, 0, 0)
        else:
            angular_velocity = self.tensor_to_vector(ms_obj.get_angular_velocity()[0])
        angular_speed = (
            angular_velocity.x**2 + angular_velocity.y**2 + angular_velocity.z**2
        ) ** 0.5

        values = dict(
            position=position,
            velocity=velocity,
            pitch=orientation.pitch,
            yaw=orientation.yaw,
            roll=orientation.roll,
            speed=speed,
            angularVelocity=angular_velocity,
            angularSpeed=angular_speed,
        )

        return values

    def is_done(self):
        return self._done

    def get_reward(self):
        return self._reward

    def destroy(self):
        # The environment belongs to the simulator and outlives this simulation,
        # so just drop our references to it and to this scene's contents. The
        # actors stay alive until the next simulation reconfigures the scene.
        # Note we rebind rather than clear these lists, since the environment
        # still holds the ones we handed it.
        self.objects_to_create = []
        self.camera_configs = []
        self.env = None
