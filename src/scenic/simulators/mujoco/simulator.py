import math
import time
import copy
import os

import mujoco
import mujoco.viewer
from dm_control import mjcf
import numpy as np
from scenic.core.simulators import Simulation, Simulator 
from scipy.spatial.transform._rotation import Rotation
from scenic.core.type_support import toOrientation

from scenic.core.vectors import Vector
from scenic.core.shapes import BoxShape, SpheroidShape, MeshShape, CylinderShape


class MujocoSimulator(Simulator):
  def __init__(self, xml='', actual = False, use_default_arena=True):
    super().__init__()
    self.xml=xml
    self.actual=actual
    self.use_default_arena = use_default_arena
  
  def createSimulation(self, scene, **kwargs):
    return MujocoSimulation(scene, self.xml, self.actual, self.use_default_arena, **kwargs)

class MujocoSimulation(Simulation):
  '''
  `Simulation object for Mujoco.
  '''

  def __init__(self, scene, xml='', actual = False, use_default_arena=True, **kwargs):
    self.xml=xml
    self.actual=actual
    self.scene=scene
    self.use_default_arena=use_default_arena
    
    self.mujocohandle=None

    if "timestep" in kwargs:
      kwargs.pop('timestep')

    super().__init__(scene, timestep=.001, **kwargs)

  def setup(self):
    super().setup()

    if self.xml != "":
      self.model = mujoco.MjModel.from_xml_string(self.xml)
      self.data = mujoco.MjData(self.model)
    else:
      mjcf_model = mjcf.RootElement(model="model")

      if self.use_default_arena:
        mjcf_model.compiler.set_attributes(angle="radian",
                                          coordinate="local",
                                          inertiafromgeom="true")

        mjcf_model.default.joint.set_attributes(armature=0,
                                              damping=1,
                                              limited="false")

        mjcf_model.default.geom.set_attributes(friction = [0.5],
                                              solimp=[0.99, 0.99, 0.01],
                                              solref=[0.01, 0.5])

        mjcf_model.option.set_attributes(gravity=[0, 0, -9.81],
                                        timestep=self.timestep)

        mjcf_model.asset.add("texture",
                            type="skybox",
                            builtin="gradient",
                            rgb1=[1, 1, 1],
                            rgb2=[.6, .8, 1],
                            width=256,
                            height=256)

        mjcf_model.asset.add("texture",
                            name="texplane",
                            type="2d",
                            builtin="checker",
                            rgb1=[1, 1, 1],
                            rgb2=[.1, .1, 2],
                            width=512,
                            height=512)

        mjcf_model.asset.add("material",
                            name="MatPlane",
                            texture="texplane",
                            texrepeat=[1, 1],
                            texuniform="true")

        mjcf_model.worldbody.add("light",
                                pos=[0, 1, 1],
                                dir=[0, -1, -1],
                                diffuse=[1, 1, 1])

        mjcf_model.worldbody.add("geom",
                                condim=3,
                                material="MatPlane",
                                name="ground",
                                pos=[0, 0, 0],
                                size=[10, 10, 0.1],
                                type="plane")

      for i, obj in enumerate(self.objects):
        # Maybe this should be cleaned up because atm if you instantiate any native scenic
        # objects this is going to crash the program
        obj_mjcf_model = obj.model()
     
        if obj_mjcf_model:
          mjcf_model.attach(obj_mjcf_model)
          
        else:
          print("Debugging path/names in simulator.py \n \n ")
          print(f"{i}\\_body \\n \\n ")
          obj.body_name = f"{i}\_body"

          mjcf_model.worldbody.add("body",
                                  name=obj.body_name,
                                  pos=[obj.position[0], obj.position[1], obj.position[2]])

          mjcf_model.worldbody.body[i].add("joint",
                                          name=f"{i}\\_joint",
                                          type="free",
                                          damping=0.001)

          quaternion = copy.copy(obj.orientation.q)
          shifted_quaternion = np.roll(quaternion, 1)
          mjcf_model.worldbody.body[i].add("geom",
                                          name=f"{i}\\_geom",
                                          quat=shifted_quaternion,
                                          size=[obj.width, obj.length, obj.height],
                                          rgba=obj.color,
                                          type=self._scenicToMujoco(obj.shape, "object.shape"),
                                          density=100)

      self.mjcf_model = mjcf_model
      
      self.xml = mjcf_model.to_xml_string(filename_with_hash=False)

      self.model = mujoco.MjModel.from_xml_string(self.xml)
      self.data = mujoco.MjData(self.model)

    self.mujocohandle = mujoco.viewer.launch_passive(self.model, self.data)

  def _scenicToMujoco(self, property, property_name):

    if property_name == "object.shape":
      body_geom_shape_map = {BoxShape: "box", SpheroidShape: "sphere", MeshShape: "mesh", CylinderShape: "cylinder"}
      return body_geom_shape_map[type(property)]

  def createObjectInSimulator(self, obj):
    if self.mujocohandle == None: pass 
    else: 
      print("Mujoco does not handle creation of objects after intialization")
      return -1

  def step(self):
    for i, obj in enumerate(self.objects):
      if hasattr(obj, "control"):
        obj.control(self.model, self.data)
    
    mujoco.mj_step(self.model, self.data)
    self.mujocohandle.sync()
    if self.actual:
      time.sleep(self.timestep)

  def getProperties(self, obj, properties):
    body_name = obj.body_name

    x,y,z=self.data.body(body_name).subtree_com
    position=Vector(x,y,z)

    # get angular velocity and speed
    a,b,c=self.data.body(body_name).cvel[3:6]
    angularVelocity = Vector(a,b,c)
    angularSpeed = math.hypot(*angularVelocity)

    # get velocity and speed
    a,b,c=self.data.body(body_name).cvel[0:3]
    velocity=Vector(a,b,c)
    speed = math.hypot(*velocity)

    cart_orientation=self.data.body(body_name).ximat 
    a,b,c,d,e,f,g,h,i=cart_orientation
    new_mat = [[a,b,c],[d,e,f,],[g,h,i]]
    r = Rotation.from_matrix(new_mat)
    # ori = toOrientation(r)
    yaw, pitch, roll = obj.yaw, obj.pitch, obj.roll #obj.parentOrientation.localAnglesFor(r)

    values = dict(
      position=position,
      velocity=velocity,
      speed=speed,
      angularSpeed=angularSpeed,
      angularVelocity=angularVelocity,
      yaw=yaw,#x
      pitch=pitch, #x
      roll=roll,#x
    )
    return values

  def destroy(self):
    if self.mujocohandle != None:
      self.mujocohandle.close()


  def get_obs(self):
    return {}
  
  def get_info(self):
    return {}
  
  def get_reward(self):
    return {}
