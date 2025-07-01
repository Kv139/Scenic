"""Interface to Webots for dynamic simulations.

This interface is intended to be instantiated from inside the controller script
of a Webots `Robot node`_ with the ``supervisor`` field set to true. Such a
script can create a `WebotsSimulator` (passing in a reference to the supervisor
node) and then call its `simulate` method as usual to run a simulation. For an
example, see :file:`examples/webots/generic/controllers/scenic_supervisor.py`.

Scenarios written for this interface should use our generic Webots world model
:doc:`scenic.simulators.webots.model` or a model derived from it. Objects which
are instances of `WebotsObject` will be matched to Webots nodes; see the model
documentation for details.

.. _Robot node: https://www.cyberbotics.com/doc/reference/robot
"""

from collections import defaultdict
import ctypes
import math
from os import path
import tempfile
from textwrap import dedent

import numpy as np
import trimesh

from scenic.core.regions import MeshVolumeRegion
from scenic.core.simulators import Simulation, Simulator
from scenic.core.type_support import toOrientation
from scenic.core.vectors import Vector
from scenic.simulators.webots.utils import ENU, WebotsCoordinateSystem
from controller import DistanceSensor


class WebotsSimulator(Simulator):
    """`Simulator` object for Webots.

    Args:
        supervisor: Supervisor node handle from the Webots Python API.
    """

    def __init__(self, supervisor):
        super().__init__()
        self.supervisor = supervisor
        topLevelNodes = supervisor.getRoot().getField("children")
        worldInfo = None
        for i in range(topLevelNodes.getCount()):
            child = topLevelNodes.getMFNode(i)
            if child.getTypeName() == "WorldInfo":
                worldInfo = child
                break
        if not worldInfo:
            raise RuntimeError("Webots world does not contain a WorldInfo node")
        system = worldInfo.getField("coordinateSystem").getSFString()
        self.coordinateSystem = WebotsCoordinateSystem(system)

    def createSimulation(self, scene, **kwargs):
        return WebotsSimulation(
            scene, self.supervisor, coordinateSystem=self.coordinateSystem, **kwargs
        )


class WebotsSimulation(Simulation):
    """`Simulation` object for Webots.

    Attributes:
        supervisor: Webots supervisor node used for the simulation. This is
            exposed for the use of scenarios which need to call Webots APIs
            directly; e.g. :scenic:`simulation().supervisor.setLabel({...})`.
    """

    def __init__(self, scene, supervisor, coordinateSystem=ENU, *, timestep, **kwargs):
        self.supervisor = supervisor
        self.coordinateSystem = coordinateSystem
        self.mode2D = scene.compileOptions.mode2D
        self.nextAdHocObjectId = 1
        self.usedObjectNames = defaultdict(lambda: 0)

        self.timestep = supervisor.getBasicTimeStep() / 1000 if timestep is None else timestep
        # directory to store proto files for adhoc webots objects
        self.tmpMeshDir = tempfile.mkdtemp()

        self.supervisor_node = self.supervisor.getSelf()

        self.left_motor = self.supervisor.getDevice("right wheel motor")
        self.right_motor = self.supervisor.getDevice("left wheel motor")

        self.sensor_right = self.supervisor.getDevice("cliff_right")
        self.sensor_front_right = self.supervisor.getDevice("cliff_front_right")

        self.sensor_left = self.supervisor.getDevice("cliff_left")
        self.sensor_front_left = self.supervisor.getDevice("cliff_front_left")

        self.sensor_back = self.supervisor.getDevice("cliff_back")

        self.bumper_left = self.supervisor.getDevice("bumper_left")
        self.bumper_right = self.supervisor.getDevice("bumper_right")

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.velocity_ranges = [0,16.129]

        self.covered_spaces = []
        self.invalid_action = False
        self.total_reward = 0

        self.enable_sensors = False
        self.actions = [0,0]
        self.observation = np.zeros(7) # TODO Need to fix obs and initialziation
        self.ms = round(1000 * self.timestep)

        self.room_width = 5.09    # meters — change as needed
        self.room_length = 5.09   # meters — change as needed
        self.granularity = 0.05    # meters (matches rounding precision)

        self.total_spaces = int((self.room_width / self.granularity) * (self.room_length / self.granularity))
        self.total_steps = 0
        self.best_coverage = 0,0
        self.collisions = 0


        super().__init__(scene, timestep=timestep, **kwargs)

    def setup(self):
        super().setup()
        # Reset Webots simulation
        self.supervisor.simulationResetPhysics()


    def createObjectInSimulator(self, obj):
        if not hasattr(obj, "webotsName"):
            return  # not a Webots object

        # Find the name of the Webots node for this object.
        name = None
        if obj.webotsAdhoc is not None:
            # Dynamically generate object from Scenic object
            objectRawMesh = obj.shape.mesh
            objectScaledMesh = MeshVolumeRegion(
                mesh=objectRawMesh,
                dimensions=(obj.width, obj.length, obj.height),
            ).mesh
            objFilePath = path.join(self.tmpMeshDir, f"{self.nextAdHocObjectId}.obj")
            trimesh.exchange.export.export_mesh(objectScaledMesh, objFilePath)



            name = self._getAdhocObjectName(self.nextAdHocObjectId)
            protoName = (
                "ScenicObjectWithPhysics" if isPhysicsEnabled(obj) else "ScenicObject"
            )
          
 
            objFilePath = str(objFilePath).replace("\\", "\\\\")# Temporary fix, not sure if this is the right way to do this? hmm

            protoDef = dedent(
                f"""
                DEF {name} {protoName} {{
                    url "{objFilePath}"
                }}
                """
            )

            rootNode = self.supervisor.getRoot()
            rootChildrenField = rootNode.getField("children")
            rootChildrenField.importMFNodeFromString(-1, protoDef)
            self.nextAdHocObjectId += 1
        else:
            if obj.webotsName:
                name = obj.webotsName
            else:
                ty = obj.webotsType
                if not ty:
                    raise RuntimeError(f"object {obj} has no webotsName or webotsType")
                nextID = self.usedObjectNames[ty]
                self.usedObjectNames[ty] += 1
                if nextID == 0 and self.supervisor.getFromDef(ty):
                    name = ty
                else:
                    name = f"{ty}_{nextID}"

        # Get handle to Webots node.
        webotsObj = self.supervisor.getFromDef(name)
        if webotsObj is None:
            raise SimulationCreationError(f"Webots object {name} does not exist in world")
        obj.webotsObject = webotsObj
        obj.webotsName = name

        # Set the fields of the Webots object:

        # position
        if self.mode2D:  # 2D compatibility mode
            # Set initial elevation if unspecified
            if obj.elevation is None:
                pos = webotsObj.getField("translation").getSFVec3f()
                spos = self.coordinateSystem.positionToScenic(pos)
                obj.elevation = spos[2]

            # Overwrite Z value with elevation
            pos = self.coordinateSystem.positionFromScenic(
                Vector(obj.position.x, obj.position.y, obj.elevation) + obj.positionOffset
            )
            webotsObj.getField("translation").setSFVec3f(pos)
        else:
            pos = self.coordinateSystem.positionFromScenic(
                obj.position + obj.positionOffset
            )
            webotsObj.getField("translation").setSFVec3f(pos)

        # orientation
        offsetOrientation = toOrientation(obj.rotationOffset)
        webotsObj.getField("rotation").setSFRotation(
            self.coordinateSystem.orientationFromScenic(
                obj.orientation, offsetOrientation
            )
        )

        # density
        densityField = getFieldSafe(webotsObj, "density")
        if densityField is not None:
            if obj.density is None:
                # Get initial value for property if unspecified
                obj.density = densityField.getSFFloat()
            else:
                densityField.setSFFloat(float(obj.density))

        # battery
        battery = getattr(obj, "battery", None)
        if battery:
            if not isinstance(battery, (tuple, list)) or len(battery) != 3:
                raise TypeError(f'"battery" of {name} does not have 3 components')
            field = webotsObj.getField("battery")
            field.setMFFloat(0, battery[0])
            field.setMFFloat(1, battery[1])
            field.setMFFloat(2, battery[2])

        # customData
        customData = getattr(obj, "customData", None)
        if customData:
            if not isinstance(customData, str):
                raise TypeError(f'"customData" of {name} is not a string')
            webotsObj.getField("customData").setSFString(customData)

        # controller
        if obj.controller:
            controllerField = webotsObj.getField("controller")
            curCont = controllerField.getSFString()
            if obj.controller != curCont:
                # the following operation also causes the controller to be restarted
                controllerField.setSFString(obj.controller)
            elif obj.resetController:
                webotsObj.restartController()

    def step(self): # action should be some low level control commands for the robot
        if not self.enable_sensors: 
               # print("Protections failed sensors were not initialized before calling") 
               # TODO more elegant fix here, sensor need to be adaquetly initialized before the simlation begins stepping
                self.init_step()

        self.total_steps += 1

        # TODO Normalize observation space, docmumnet sensor value ranges, and signals for crashing etc...
        self.observation = np.array([self.actions[0], self.actions[1], self.sensor_left.getValue()/800, self.sensor_right.getValue()/800, # ensures that null values are not returned from unintialized sensors
                self.sensor_front_right.getValue()/800, self.sensor_front_left.getValue()/800, self.sensor_back.getValue()/800, self.pos[0], self.pos[1]])       

        self.transform_vel()
        self.left_motor.setVelocity(self.actions[0]) 
        self.right_motor.setVelocity(self.actions[1])
        self.supervisor.step(self.ms)

        covered_count, converage_ratio = self.get_coverage_metric()
       
        if converage_ratio > self.best_coverage[1]:
            self.best_coverage = covered_count, converage_ratio


    def init_step(self):
        """
        Initialize all the sensors and devices on the robot
        """
        self.sensor_right.enable(self.ms)
        self.sensor_front_right.enable(self.ms)

        self.sensor_front_left.enable(self.ms)
        self.sensor_left.enable(self.ms)

        self.bumper_left.enable(self.ms)
        self.bumper_right.enable(self.ms)

        self.sensor_back.enable(self.ms)

        self.supervisor.step(self.ms) # Need to step the simulation once after initializing the sensors!
        pos = self.granularity * np.round(np.array(self.supervisor_node.getPosition()[:2]) / self.granularity) #need to verify
        self.pos = pos # initialize the position
        self.enable_sensors = True


    def getProperties(self, obj, properties):
        webotsObj = getattr(obj, "webotsObject", None)
        if not webotsObj:  # static object with no Webots counterpart
            return {prop: getattr(obj, prop) for prop in properties}

        pos = webotsObj.getField("translation").getSFVec3f()
        x, y, z = self.coordinateSystem.positionToScenic(pos)
        lx, ly, lz, ax, ay, az = webotsObj.getVelocity()
        vx, vy, vz = self.coordinateSystem.positionToScenic((lx, ly, lz))
        velocity = Vector(vx, vy, vz)
        speed = math.hypot(*velocity)
        angularSpeed = math.hypot(ax, ay, az)

        offsetOrientation = toOrientation(obj.rotationOffset)
        globalOrientation = self.coordinateSystem.orientationToScenic(
            webotsObj.getField("rotation").getSFRotation(), offsetOrientation
        )
        yaw, pitch, roll = obj.parentOrientation.localAnglesFor(globalOrientation)

        values = dict(
            position=Vector(x, y, z),
            velocity=velocity,
            speed=speed,
            angularSpeed=angularSpeed,
            angularVelocity=Vector(ax, ay, az),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            elevation=z,
        )

        if hasattr(obj, "battery"):
            field = webotsObj.getField("battery")
            val = (field.getMFFloat(0), obj.battery[1], obj.battery[2])
            values["battery"] = val

        return values

    def destroy(self):
        # Destroy adhoc objects generated at the beginning of the simulation
        print(f" total episode reward was {self.total_reward}")

        print(f"This is the metric: {self.metric()}")
        print(f"Covered {self.best_coverage[0]} cells out of {self.total_spaces} ({self.best_coverage[1]*100:.2f}%) \n")

        for i in range(1, self.nextAdHocObjectId):
            name = self._getAdhocObjectName(i)
            node = self.supervisor.getFromDef(name)
            if node is not None: # ensure that the node actually exisits in the simulation before destroying it
                node.remove()
            self.supervisor.step(self.ms) # TODO this fixe crashing error on repeated reset calls! I DO NOT KNOW WHY.... temp fix, need to figure out underlying cause
    
    def _getAdhocObjectName(self, i: int) -> str:
        return f"SCENIC_ADHOC_{i}"

    def get_coverage_reward(self, pos):
        reward = 0
        #important parameter
        radius = .335/2
        x_range = np.arange(pos[0] - radius - self.granularity, pos[0] + radius + self.granularity, self.granularity)
        y_range = np.arange(pos[1] - radius - self.granularity, pos[1] + radius + self.granularity, self.granularity)
        x_range_combined, y_range_combined = np.meshgrid(x_range, y_range, indexing="xy")
        mask = (x_range_combined - pos[0])**2 + (y_range_combined - pos[1])**2 <= radius**2
        circle_points = [
            (
                round(self.granularity * round(x / self.granularity), 3),
                round(self.granularity * round(y / self.granularity), 3)
            )
            for x, y in np.vstack((x_range_combined[mask],
                                y_range_combined[mask])).T
        ]
        for point in circle_points:
            if(point not in self.covered_spaces):
                reward += 1
                self.covered_spaces.append(point)
        return reward
    
    def metric(self):

        avg_reward = self.total_reward / self.total_steps if self.total_steps > 0 else 0
        exploration = len(self.covered_spaces)
        collision_rate = self.collisions / self.total_steps if self.total_steps > 0 else 0

        score = avg_reward - 10 * collision_rate + 0.1 * exploration

        return {
            "total_reward": self.total_reward,
            "average_reward": avg_reward,
            "steps": self.total_steps,
            "collision_count": self.collisions,
            "exploration_score": exploration,
            "final_score": score
        }
    
    def get_coverage_metric(self):
        # Number of unique positions visited
        covered_count = len(self.covered_spaces)
        # Coverage ratio (fraction of total spaces covered)
        coverage_ratio = covered_count / self.total_spaces
        # Optionally: return both count and percentage
        return covered_count, coverage_ratio  


    def get_reward(self): # "any dummy for now will be okay"
        """
        Calculate the reward based off of the current state
        """
        pos = self.granularity * np.round(np.array(self.supervisor_node.getPosition()[:2]) / self.granularity) #need to verify

        self.pos = pos

        reward = 0
        reward += self.get_coverage_reward([pos[0], pos[1]])

        if reward == 0:
            reward += -1 
        if self.invalid_action:
            reward = -100
            self.invalid_action = False

        if np.all(self.observation[:2] > 0):
            reward += .5 # small reward for driving forward

        if (np.any(self.observation[2:] < 0.1) ): # if any distance sensor is low penalize
            vm = np.mean(np.abs([self.actions[0], self.actions[1]]))
            reward += -abs(vm)/self.velocity_ranges[1]
            self.collisions += 1        
        elif (self.bumper_left == 1) or (self.bumper_right == 1):
            vm = np.mean(np.abs([self.actions[0], self.actions[1]]))
            reward += -abs(vm)/self.velocity_ranges[1]
            self.collisions += 1

        self.total_reward += reward
        return reward
    
    def get_info(self):
        """
        Any information about the system/state that should be retained
        """
        return {}
     
    def get_obs(self):
        """
        Return the current state of the enviroment
        """
        return self.observation
    
    def transform_vel(self): 
        """
        Maps the actors actions from (-1,1) to the actual motor range
        of the robot system
        """
        self.actions[0] = self.actions[0] * self.velocity_ranges[1] 
        self.actions[1] = self.actions[1] * self.velocity_ranges[1]
        if np.any(np.abs(self.actions) > 16.139):
            print("Error with velocity comp:")
            self.invalid_action = True
            print(f"Actions: {self.actions[0], self.actions[1]}")
            self.actions[0] = 0
            self.actions[1] = 0 # set invalid action to 0 instead

            

def getFieldSafe(webotsObject, fieldName):
    """Get field from webots object. Return null if no such field exists.

    Needed to workaround this issue (https://github.com/cyberbotics/webots/issues/5646)

    Args:
        webotsObject: webots object
        fieldName: name of the field to look for

    Returns:
        Field|None: Field object if the field with the given name exists. None otherwise.
    """

    field = webotsObject.getField(fieldName)
    # this seems to always return some object, but return None if field is None
    if field is None:
        return None

    # if field is valid, it has a valid pointer
    if isinstance(field._ref, ctypes.c_void_p) and field._ref.value is not None:
        # then the field is valid and we return the reference
        return field

    # if the pointer points to None, then the field does not exist on this object
    return None


def isPhysicsEnabled(webotsObject):
    """Whether or not physics is enabled for this `WebotsObject`"""
    if webotsObject.webotsAdhoc is None:
        return webotsObject
    if isinstance(webotsObject.webotsAdhoc, dict):
        return webotsObject.webotsAdhoc.get("physics", True)
    raise TypeError(f"webotsAdhoc must be None or a dictionary")

