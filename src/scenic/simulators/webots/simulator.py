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
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import math
from os import path
import tempfile
import matplotlib.patches as patches
from textwrap import dedent

import numpy as np
import trimesh

from scenic.core.regions import MeshVolumeRegion
from scenic.core.simulators import Simulation, Simulator
from scenic.core.type_support import toOrientation
from scenic.core.vectors import Vector
from scenic.simulators.webots.utils import ENU, WebotsCoordinateSystem
from controller import DistanceSensor

from trimesh.creation import box

import yaml

import math, numpy as np
from trimesh.proximity import closest_point
from trimesh.proximity import ProximityQuery

episodes = 0
saved_stepcount = 0
class WebotsSimulator(Simulator):
    """`Simulator` object for Webots.

    Args:
        supervisor: Supervisor node handle from the Webots Python API.
    """
    episode_count = 0
    current_simulation = None
    last_avg_return = None
    
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

    def createSimulation(self, scene, lidar_max_range=None, **kwargs): #accept lidar_max_range
            self.episode_count += 1
            self.current_simulation = WebotsSimulation(
                scene,
                self.supervisor,
                coordinateSystem=self.coordinateSystem,
                parent_simulator=self,
                # lidar_max_range=lidar_max_range, #Pass it to WebotsSimulation
                **kwargs
            )
            return self.current_simulation


class WebotsSimulation(Simulation):
    """`Simulation` object for Webots.

    Attributes:
        supervisor: Webots supervisor node used for the simulation. This is
            exposed for the use of scenarios which need to call Webots APIs
            directly; e.g. :scenic:`simulation().supervisor.setLabel({...})`.
    """
    def __init__(self, scene, supervisor, coordinateSystem=ENU, *, timestep, parent_simulator=None, lidar_max_range=None, **kwargs): # Accept lidar_max_range
        #room data
        self.room_width = 5.    
        self.room_length = 5.   
        self.granularity = 0.05    
        self.total_spaces = (2 * np.floor(self.room_width / (2*self.granularity)) + 1)**2 - 4 #-4 for each of the corners
        self.obj_dims = []
        
        #collisions & collision detection
        self.collisions = 0
        self.discrete_collisions = 0
        self.colliding = False
        self.collision_safeguard = 0
        self.inter_penalty = False
        self.prox_checks = []
        self.spheres = []
        
        #metrics and rewards
        self.best_coverage = 0,0
        self.covered_spaces = []
        self.coverage_timesteps = []
        self.invalid_action = False
        self.total_reward = 0
        
        #simulation data
        self.time_elapsed = 0
        self.total_reward = 0
        self.total_steps = 0
        
        #i dont know
        self.supervisor = supervisor
        self.coordinateSystem = coordinateSystem
        self.mode2D = scene.compileOptions.mode2D
        self.nextAdHocObjectId = 1
        self.usedObjectNames = defaultdict(lambda: 0)
        self.timestep = supervisor.getBasicTimeStep() / 1000 if timestep is None else timestep
        # directory to store proto files for adhoc webots objects
        self.tmpMeshDir = tempfile.mkdtemp()
        self.supervisor_node = self.supervisor.getSelf()

        #device inputs
        self.left_motor = self.supervisor.getDevice("right wheel motor")
        self.right_motor = self.supervisor.getDevice("left wheel motor")

        self.sensor_right = self.supervisor.getDevice("cliff_right")
        self.sensor_front_right = self.supervisor.getDevice("cliff_front_right")

        self.sensor_left = self.supervisor.getDevice("cliff_left")
        self.sensor_front_left = self.supervisor.getDevice("cliff_front_left")

        self.sensor_back = self.supervisor.getDevice("cliff_back")
        self.sensor_actual_left = self.supervisor.getDevice("actual_left")
        self.sensor_actual_right = self.supervisor.getDevice("actual_right")
        self.LIDAR = self.supervisor.getDevice("LIDAR")


        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.velocity_ranges = [0,16.129]

        self.enable_sensors = False
        self.actions = [0,0]
        self.ms = round(1000 * self.timestep)

        #observation space

        self.observation = {
            "velocity": np.zeros(2), 
            #"sensor": np.zeros(7),
            "lidar": np.full(32, 0.25), #there are 32 lidar sensors
            "position": np.zeros(2),
            "rotation": [0,0,0,0]
            # "sectional_coverage":np.zeros(16),
            # "current_section": 0
        } # TODO Need to fix obs and initialziation        
              
        
        super().__init__(scene, timestep=timestep, **kwargs)

    def setup(self):
        super().setup()
        # Reset Webots simulation
        self.supervisor.simulationResetPhysics()
        self.compute_total_tiles()


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
        if hasattr(obj, 'width') and hasattr(obj, 'length'):
            name = (obj.webotsName or obj.webotsType or "")
            if "SCENIC_ADHOC7" in name:
                for i in range(4):
                    self.obj_dims.append((0.1, 0.1))
            else:
                self.obj_dims.append((float(obj.width), float(obj.length)))
            
            
    def compute_total_tiles(self):
        room_area = self.room_width * self.room_length
        object_area = sum(width * length for width, length in self.obj_dims)
        cleanable_area = room_area - object_area

        tile_area = self.granularity ** 2
        total_tiles = int(cleanable_area / tile_area)
        self.total_spaces = total_tiles
        #print(f"Computed total cleanable tiles: {total_tiles}")
                
    def get_coverage_metric(self):
        # Number of unique positions visited
        covered_count = len(self.covered_spaces)
        # Coverage ratio (fraction of total spaces covered)
        coverage_ratio = covered_count / self.total_spaces
        # Optionally: return both count and percentage
        return covered_count, coverage_ratio          
                
    def step(self): # action should be some low level control commands for the robot
        if not self.enable_sensors: 
               # print("Protections failed sensors were not initialized before calling") 
               # TODO more elegant fix here, sensor need to be adaquetly initialized before the simlation begins stepping
                self.init_step()

        rot = np.array(self.supervisor_node.getField("rotation").getSFVec2f(), dtype=np.float32)

        self.total_steps += 1
        pos = self.granularity * np.round(np.array(self.supervisor_node.getPosition()[:2]) / self.granularity)
        # TODO Normalize observation space, docmumnet sensor value ranges, and signals for crashing etc...
        #if episode is under 10 input_val=2.6, else input_val=100- self.current_total_coverage_sum
        #self.current_total_coverage_sum max assuming 100% acorss 10 episodes is 10
        input_val = -0.26 * self.current_total_coverage_sum + 5.2
        # input_val = -0.03 * self.current_total_coverage_sum + 1
        if self.current_total_coverage_sum < 5:
            input_val = 5.2
            # input_val = 1
        else:
            input_val = -0.26 * self.current_total_coverage_sum + 5.2
            # input_val = -0.03 * self.current_total_coverage_sum + 1


        raw_lidar = np.array(self.LIDAR.getRangeImage(), dtype=np.float64)
        raw_lidar = np.nan_to_num(raw_lidar, nan=input_val, posinf=input_val, neginf=0.25)
        #max value is input_val, min value is 0.25
        raw_lidar = np.clip(raw_lidar, 0.25, input_val)

        #-----------------------------------------------------------------------
        # # #print episode
        # print(f"Episode number: {episodes}")
        ##print lidar values, not normalized
        #print(f"LIDAR values: {raw_lidar}") #print lidar values for debugging
        # #print sum of covered spaces last 10 episodes
        #print (f"sum of last 10 episodes {self.current_total_coverage_sum:.5f}")
        # print (f"input_val (max on lidar) {input_val:.5f}")
        # #print coverage percentage
        # print(f"Coverage percentage: {self.best_coverage[1] * 100:.2f}%")
        # min_lidar = min(self.observation["lidar"])
        # print(min_lidar)
        # if (min_lidar < 0.4):
        #print ("this is the couch", self.records["couchPosition"][len(self.records["couchPosition"]) - 1][1])
        
        #------------------------------------------------------------------
        
        #print(f"Sum of covered spaces last 10 episodes: {np.sum(self.last_10_episode_coverages)}") #print sum of covered spaces last 10 episodes
        #print(f"input_val used for nan and posinf: {input_val}")
        #raw_lidar = np.clip(raw_lidar, 0.2, self.objects[0].clip_range)  # ensure it stays in observation space bounds
        # raw_lidar = (raw_lidar - 0.2) / (self.objects[0].clip_range - 0.2) # change 2.6 to max range
        #print(f"LIDAR (min/avg/max): {np.min(raw_lidar):.5f}/{np.mean(raw_lidar):.5f}/{np.max(raw_lidar):.5f}")
        # Assemble observation
        self.observation = {
            "velocity": np.array([self.actions[0], self.actions[1]]),
            # "sensor": np.array([self.sensor_left.getValue()/800, self.sensor_right.getValue()/800, # ensures that null values are not returned from unintialized sensors
            #     self.sensor_front_right.getValue()/800, self.sensor_front_left.getValue()/800, self.sensor_back.getValue()/800, self.sensor_actual_left.getValue()/800,  
            #                          self.sensor_actual_right.getValue()/800]),
            "position": np.array(pos),
            "lidar": raw_lidar,
            "rotation" : np.array([rot[0], rot[1], rot[2], rot[3]]),
            # "sectional_coverage": self.sectional_coverage / (self.total_spaces / 16),
            # "current_section": self.posToIdx(pos)
        }
        self.transform_vel()
        self.left_motor.setVelocity(self.actions[0]) 
        self.right_motor.setVelocity(self.actions[1])
        self.supervisor.step(self.ms)
        self.time_elapsed += self.timestep
        covered_count, coverage_ratio = self.get_coverage_metric()
        if coverage_ratio > self.best_coverage[1]:
            self.best_coverage = covered_count, coverage_ratio
        
    def getObjects(self):
        for obj in self.objects:
            if "floor" in str(obj).lower() or "vacuum" in str(obj).lower():
                continue
            x, y, z = obj.position
            yaw = obj.heading
            c, s = math.cos(yaw), math.sin(yaw)
            # 4×4 yaw+translate
            T = np.array([
                [ c, -s, 0, x],
                [ s,  c, 0, y],
                [ 0,  0, 1, z],
                [ 0,  0, 0, 1]
            ])
            base = obj.shape._mesh                     
            dims = (obj.width, obj.length, obj.height)  
            mesh = MeshVolumeRegion(mesh=base, dimensions=dims).mesh
            mesh_in_world = mesh.copy()
            mesh_in_world.apply_transform(T)
            self.prox_checks.append(ProximityQuery(mesh_in_world))
            self.spheres.append([obj.position] + [mesh_in_world.bounding_sphere.primitive.radius])

    def init_step(self):
        """
        Initialize all the sensors and devices on the robot
        """
        self.sensor_right.enable(self.ms)
        self.sensor_front_right.enable(self.ms)

        self.sensor_front_left.enable(self.ms)
        self.sensor_left.enable(self.ms)
        self.LIDAR.enable(self.ms)


        self.sensor_back.enable(self.ms)

        self.sensor_actual_left.enable(self.ms)
        self.sensor_actual_right.enable(self.ms)

        self.supervisor.step(self.ms) # Need to step the simulation once after initializing the sensors!
        pos = self.granularity * np.round(np.array(self.supervisor_node.getPosition()[:2]) / self.granularity) #need to verify
        self.pos = pos # initialize the position
        self.enable_sensors = True
        self.getObjects()


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
    
    def metric(self):

        avg_reward = self.total_reward / self.total_steps if self.total_steps > 0 else 0
        exploration = len(self.covered_spaces)
        collision_rate = self.collisions / self.total_steps if self.total_steps > 0 else 0

        score = avg_reward - 10 * collision_rate + 0.1 * exploration

        return {
            "total_reward": self.total_reward,
            "average_reward": avg_reward,
            "steps": self.total_steps,
            "time_elapsed": self.time_elapsed,
            "collision_count": self.collisions,
            "exploration_score": exploration,
            "final_score": score
        }

    def create_heatmap(self, coordinates):
        if not coordinates:
            print("No data to plot.")
            return

        x, y = zip(*coordinates)
        x = np.asarray(x)
        y = np.asarray(y)

        data = np.vstack([x, y])

        if len(np.unique(data, axis=1)) <= 1:
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='red', s=100)
            plt.title('Single Coordinate Point')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')
            plt.xlim(-2.545, 2.545)
            plt.ylim(-2.545, 2.545)
            plt.grid(True)
            plt.show()
            return

        try:
            kde = gaussian_kde(data)
        except np.linalg.LinAlgError:
            print("Can't perform KDE (likely due to linear data).")
            return

        # Fixed grid range
        x_min, x_max = -2.545, 2.545
        y_min, y_max = -2.545, 2.545

        xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

        # Create plot with furniture overlays
        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.pcolormesh(xi, yi, zi, shading='gouraud', cmap='coolwarm')
        fig.colorbar(c, ax=ax, label='Density')

        #  furniture 
        try:
            couch = patches.Rectangle(
                (
                    self.records["couchPosition"][-1][1][0] - 0.375,
                    self.records["couchPosition"][-1][1][1] - 1
                ),
                0.75,
                2,
                linewidth=2,
                edgecolor='black',
                facecolor='black'
            )
            ax.add_patch(couch)

            ax.text(
                self.records["couchPosition"][-1][1][0],     
                self.records["couchPosition"][-1][1][1],   
                'couch',
                color='white', 
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )

            coffee_table = patches.Rectangle(
                (
                    self.records["coffee_tablePosition"][-1][1][0] - 0.25,
                    self.records["coffee_tablePosition"][-1][1][1] - 0.75
                ),
                0.5,
                1.5,
                linewidth=2,
                edgecolor='black',
                facecolor='black'
            )
            ax.add_patch(coffee_table)

            ax.text(
                self.records["coffee_tablePosition"][-1][1][0],     
                self.records["coffee_tablePosition"][-1][1][1],   
                'Coffe Table',
                color='white', 
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )


            dining_table = patches.Rectangle(
                (
                    self.records["DiningTablePosition"][-1][1][0] - 0.375,
                    self.records["DiningTablePosition"][-1][1][1] - 0.375
                ),
                0.75,
                0.75,
                linewidth=2,
                edgecolor='black',
                facecolor='black'
            )
            ax.add_patch(dining_table)

            ax.text(
                self.records["DiningTablePosition"][-1][1][0],     
                self.records["DiningTablePosition"][-1][1][1],   
                'Dining Table',
                color='white', 
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )

            Chair1 = patches.Rectangle(
                (
                    self.records["chair_1Position"][-1][1][0] - 0.2,
                    self.records["chair_1Position"][-1][1][1] - 0.2
                ),
                0.4,
                0.4,
                linewidth=2,
                edgecolor='black',
                facecolor='black'
            )
            ax.add_patch(Chair1)

            ax.text(
                self.records["chair_1Position"][-1][1][0],     
                self.records["chair_1Position"][-1][1][1],   
                'Chair 1',
                color='white', 
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )

            Chair3 = patches.Rectangle(
                (
                    self.records["chair_3Position"][-1][1][0] - 0.2,
                    self.records["chair_3Position"][-1][1][1] - 0.2
                ),
                0.4,
                0.4,
                linewidth=2,
                edgecolor='black',
                facecolor='black'
            )
            ax.add_patch(Chair3)

            ax.text(
                self.records["chair_3Position"][-1][1][0],     
                self.records["chair_3Position"][-1][1][1],   
                'Chair 3',
                color='white', 
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
            
        except (KeyError, IndexError):
            print("Furniture positions missing or improperly formatted.")

        ax.set_title('Heatmap of Covered Spaces with Furniture')
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True)

        if episodes % 2 == 0:
            filename = fr"C:\Users\PC-5\Downloads\SIP_Work_Graphs\heatmap_episode{episodes}.png"
            plt.savefig(filename, dpi=300)
        # if episodes == 5:
        #     plt.savefig("heatmap_episode5.png", dpi=300)
        # if episodes == 10:
        #     plt.savefig("heatmap_episode10.png", dpi=300)
    
    def destroy(self):
        global episodes
        episodes += 1
        if episodes % 10 == 0:
            self.create_heatmap(self.covered_spaces)
        print(f"Episode number: {episodes}")
        #print(f"This is the metric: {self.metric()}")
        print(f"Covered {self.best_coverage[0]} cells out of {self.total_spaces} ({self.best_coverage[1]*100:.2f}%)")
        # Destroy adhoc objects generated at the beginning of the simulation
        #print(f" total episode reward was {self.total_reward}")

        for i in range(1, self.nextAdHocObjectId):
            name = self._getAdhocObjectName(i)
            node = self.supervisor.getFromDef(name)
            if node is not None: # ensure that the node actually exisits in the simulation before destroying it
                node.remove()
            self.supervisor.step(self.ms) # TODO this fixe crashing error on repeated reset calls! I DO NOT KNOW WHY.... temp fix, need to figure out underlying cause
    def _getAdhocObjectName(self, i: int) -> str:
        return f"SCENIC_ADHOC_{i}"
    
    def posToIdx(self, pos):
        idx = np.array([0, 0])
        for i in range(0, 2):
            if(pos[i] <= self.room_width / 4 * -1):
                idx[i] = 0
            elif(pos[i] <= 0):
                idx[i] = 1
            elif(pos[i] <= self.room_width / 4):
                idx[i] = 2
            else:
                idx[i] = 3
        return 4 * idx[0] + idx[1]
        

    def get_coverage_reward(self, granularity, pos):
            reward = 0
            #important parameter
            radius = .335/2
            x_range = np.arange(pos[0] - radius - granularity, pos[0] + radius + granularity, granularity)
            y_range = np.arange(pos[1] - radius - granularity, pos[1] + radius + granularity, granularity)
            x_range_combined, y_range_combined = np.meshgrid(x_range, y_range, indexing="xy")
            mask = (x_range_combined - pos[0])**2 + (y_range_combined - pos[1])**2 <= radius**2
            circle_points = [
                (
                    round(granularity * round(x / granularity), 3),
                    round(granularity * round(y / granularity), 3)
                )
                for x, y in np.vstack((x_range_combined[mask],
                                    y_range_combined[mask])).T
            ]
            for point in circle_points:
                if(point not in self.covered_spaces):
                    reward += 1
                    self.covered_spaces.append(point)
                    self.coverage_timesteps.append(self.total_steps)
            if reward == 0:
                reward += -.001
            return reward

    def checkCollisions(self):
        minDist = 0.01
        for i in range(len(self.prox_checks)):
            if math.dist(self.spheres[i][0], self.records["EgoPosition"][len(self.records["EgoPosition"]) - 1][1]) > .335/2 + self.spheres[i][1] + minDist:
                continue    
            if(abs(self.prox_checks[i].signed_distance(np.array([self.records["EgoPosition"][len(self.records["EgoPosition"]) - 1][1]]))) < .335/2 + minDist):
                return True
        return False

    def get_reward(self): # "any dummy for now will be okay"
        """
        Calculate the reward based off of the current state
        """
        pos = self.granularity * np.round(np.array(self.supervisor_node.getPosition()[:2]) / self.granularity) #need to verify
        pos = tuple(pos.tolist())
        reward = self.get_coverage_reward(self.granularity, [pos[0], pos[1]])
        
        # if np.all(self.observation["velocity"] > 0):
        #     reward += .2 # small reward for driving forwa
        
        if (self.checkCollisions()): # if any distance sensor is low penalize
            reward += -1
            self.collision_safeguard += 1
            self.collisions += 1
            if not self.colliding:
                self.colliding = True
                self.discrete_collisions += 1
        else:
            self.collision_safeguard = 0
            self.colliding = False
        if self.collision_safeguard >= 40 and not self.inter_penalty:
            reward += -100
            self.inter_penalty = True
        
        if self.invalid_action:
            reward += -100
            print("Invalid action")
            self.invalid_action = False
        self.total_reward += reward
        return reward

    def get_info(self):
        """
        Any information about the system/state that should be retained
        """
        cleaned_cells = len(self.covered_spaces)
        total_cleanable_tiles = self.total_spaces
        coverage_percent = (cleaned_cells / total_cleanable_tiles) * 100 if total_cleanable_tiles > 0 else 0

        return {
            "cleaned_cells": cleaned_cells,
            "total_cleanable_tiles": total_cleanable_tiles,
            "coverage": coverage_percent,
            "collisions": self.collisions,
            "discrete_collisions": self.discrete_collisions
        }
     
     
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

        if np.any(np.abs(self.actions) > self.velocity_ranges[1]):
            print("Error with velocity comp:")
            self.invalid_action = True
            self.actions[0] = 0
            self.actions[1] = 0 # set invalid action to 0 instead
    
    def get_truncation(self):
        if self.collision_safeguard > 50:
            return True
        else:
            return False

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

