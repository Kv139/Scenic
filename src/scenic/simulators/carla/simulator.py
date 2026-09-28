"""Simulator interface for CARLA."""

try:
    import carla
except ImportError as e:
    raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

import math
import os

import scenic.core.errors as errors

if errors.verbosityLevel == 0:  # suppress pygame advertisement at zero verbosity
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame

from scenic.domains.driving.simulators import DrivingSimulation, DrivingSimulator
from scenic.simulators.carla.utils import actors as carla_actor_utils
import scenic.simulators.carla.utils.utils as utils
import scenic.simulators.carla.utils.visuals as visuals
from scenic.syntax.veneer import verbosePrint


class CarlaSimulator(DrivingSimulator):
    """Implementation of `Simulator` for CARLA."""

    def __init__(
        self,
        carla_map,
        map_path,
        address="127.0.0.1",
        port=2000,
        timeout=10,
        render=True,
        record="",
        timestep=0.1,
        traffic_manager_port=None,
        bubble_size=None,
    ):
        super().__init__()
        verbosePrint(f"Connecting to CARLA on port {port}")
        self.client = carla.Client(address, port)
        self.client.set_timeout(timeout)  # limits networking operations (seconds)
        if carla_map is not None:
            try:
                self.world = self.client.load_world(carla_map)
            except Exception as e:
                raise RuntimeError(f"CARLA could not load world '{carla_map}'") from e
        else:
            if str(map_path).endswith(".xodr"):
                with open(map_path) as odr_file:
                    self.world = self.client.generate_opendrive_world(odr_file.read())
            else:
                raise RuntimeError("CARLA only supports OpenDrive maps")

        if traffic_manager_port is None:
            traffic_manager_port = port + 6000
        self.tm = self.client.get_trafficmanager(traffic_manager_port)
        self.tm.set_synchronous_mode(True)

        # Set to synchronous with fixed timestep
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.timestep = timestep
        if timestep > 0.1:
            settings.fixed_delta_seconds = 0.1  # NOTE: Should not exceed 0.1
        else:
            settings.fixed_delta_seconds = timestep
        self.world.apply_settings(settings)
        verbosePrint("Map loaded in simulator.")

        self.render = render  # visualization mode ON/OFF
        self.record = record  # whether to use the carla recorder
        self.scenario_number = 0  # Number of the scenario executed
        if bubble_size is not None:
            self.bubble_size = bubble_size
        else:
            self.bubble_size = None

    def createSimulation(self, scene, *, timestep, **kwargs):
        if timestep is not None and timestep != self.timestep:
            raise RuntimeError(
                "cannot customize timestep for individual CARLA simulations; "
                "set timestep when creating the CarlaSimulator instead"
            )

        self.scenario_number += 1
        return CarlaSimulation(
            scene,
            self.client,
            self.tm,
            self.render,
            self.record,
            self.scenario_number,
            timestep=self.timestep,
            bubble_size=self.bubble_size,
            **kwargs,
        )

    def destroy(self):
        super().destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(False)


class CarlaSimulation(DrivingSimulation):
    def __init__(
        self,
        scene,
        client,
        tm,
        render,
        record,
        scenario_number,
        timestep,
        bubble_size=None,
        **kwargs,
    ):
        self.client = client
        self.world = self.client.get_world()
        self.current_frame = None
        self.map = self.world.get_map()
        self.blueprintLib = self.world.get_blueprint_library()
        self.tm = tm
        self.render = render
        self.record = record
        self.scenario_number = scenario_number
        self.cameraManager = None
        if timestep > 0.1:
            self.sim_ticks_per = int(round(timestep / 0.1))
            assert math.isclose(self.sim_ticks_per, timestep / 0.1)
        else:
            self.sim_ticks_per = 1
        self.timestep = timestep
        print(
            f"Setting sim_ticks_per as: {self.sim_ticks_per} and timestep as: {timestep}"
        )

        if bubble_size is not None:
            self.bubble_size = bubble_size
            self.hybrid_physics = True
        else:
            self.hybrid_physics = False
        self.total_spawned_actors = 0
        self.steps_taken = 0

        super().__init__(scene, timestep=timestep, **kwargs)

    def setup(self):
        print("Instantiating new simulator instance")
        weather = self.scene.params.get("weather")
        if weather is not None:
            if isinstance(weather, str):
                self.world.set_weather(getattr(carla.WeatherParameters, weather))
            elif isinstance(weather, dict):
                self.world.set_weather(carla.WeatherParameters(**weather))

        # Setup HUD
        if self.render:
            self.displayDim = (1280, 720)
            self.displayClock = pygame.time.Clock()
            self.camTransform = 0
            pygame.init()
            pygame.font.init()
            self.hud = visuals.HUD(*self.displayDim)
            self.display = pygame.display.set_mode(
                self.displayDim, pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.cameraManager = None

        if self.record:
            if not os.path.exists(self.record):
                os.mkdir(self.record)
            name = "{}/scenario{}.log".format(self.record, self.scenario_number)
            # Carla is looking for an absolute path, so convert it if necessary.
            name = os.path.abspath(name)
            self.client.start_recorder(name)

        # Create objects.
        super().setup()

        # Set up camera manager and collision sensor for ego
        if self.render:
            camIndex = 0
            camPosIndex = 0
            egoActor = self.objects[0].carlaActor
            if self.hybrid_physics:
                egoActor.role_name = "hero"
                self.tm.set_hybrid_physics_mode(True)
                self.tm.set_hybrid_physics_radius(self.bubble_size)

            self.cameraManager = visuals.CameraManager(self.world, egoActor, self.hud)
            self.cameraManager._transform_index = camPosIndex
            self.cameraManager.set_sensor(camIndex)
            self.cameraManager.set_transform(self.camTransform)

        self.world.tick()  ## allowing manualgearshift to take effect    # TODO still need this?

        for obj in self.objects:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.apply_control(
                    carla.VehicleControl(manual_gear_shift=False)
                )

        self.current_frame = self.world.tick()

        for obj in self.objects:
            if obj.speed is not None and obj.speed != 0:
                raise RuntimeError(
                    f"object {obj} cannot have a nonzero initial speed "
                    "(this is not yet possible in CARLA)"
                )

    def createObjectInSimulator(self, obj):
        blueprint = carla_actor_utils.prepare_actor_blueprint(obj, self.blueprintLib)

        # Set up transform
        loc = utils.scenicToCarlaLocation(
            obj.position,
            world=self.world,
            blueprint=obj.blueprint,
            snapToGround=obj.snapToGround,
        )
        rot = utils.scenicToCarlaRotation(obj.orientation)
        transform = carla.Transform(loc, rot)

        # Create Carla actor
        carlaActor = self.world.try_spawn_actor(blueprint, transform)
        if carlaActor is None:
            # raise SimulationCreationError(f"Unable to spawn object {obj}")
            return None
        obj.carlaActor = carlaActor

        carlaActor.set_simulate_physics(obj.physics)

        carla_actor_utils.configure_actor(obj, self.world, self.blueprintLib)
        if isinstance(carlaActor, carla.Vehicle) and obj.behavior is None:
            carlaActor.set_autopilot(True, self.tm.get_port())
        self.total_spawned_actors += 1

        # Adding sensors if available
        if obj.sensors:
            for sensor_key, sensor in obj.sensors.items():
                sensor_bp = self.blueprintLib.find(sensor.blueprint)
                for key, val in sensor.attributes.items():
                    if sensor_bp.has_attribute(key):
                        sensor_bp.set_attribute(key, str(val))

                if isinstance(sensor.convert, str):
                    try:
                        sensor.convert = carla.ColorConverter.names[sensor.convert]
                    except KeyError:
                        raise ValueError(
                            f"Unknown CARLA ColorConverter '{sensor.convert}'"
                        )

                transform = carla.Transform(
                    carla.Location(
                        x=sensor.offset[1],  # forward (Scenic +Y) -> CARLA +X
                        y=sensor.offset[0],  # right   (Scenic +X) -> CARLA +Y
                        z=sensor.offset[2],  # up      (same)
                    ),
                    carla.Rotation(
                        pitch=sensor.rotation[1],
                        yaw=-sensor.rotation[0],  # CARLA yaw is clockwise
                        roll=sensor.rotation[2],
                    ),
                )

                carla_sensor = self.world.spawn_actor(
                    sensor_bp, transform, attach_to=obj.carlaActor
                )
                carla_sensor.listen(sensor.onData)
                sensor.carla_sensor = carla_sensor

        return carlaActor

    def executeActions(self, allActions):
        super().executeActions(allActions)

        # Apply control updates which were accumulated while executing the actions
        for obj in self.agents:
            ctrl = obj._control
            if ctrl is not None:
                obj.carlaActor.apply_control(ctrl)
                obj._control = None

    def step(self):
        # Run simulation for one timestep
        for _ in range(self.sim_ticks_per):
            self.current_frame = self.world.tick()
        self.steps_taken += 1

        # Wait for sensors to get updates
        for obj in self.objects:
            if obj.sensors:
                for sensor in obj.sensors.values():
                    while sensor.frame != self.current_frame:
                        pass

        # Render simulation
        if self.render:
            self.cameraManager.render(self.display)
            pygame.display.flip()

        if self.steps_taken % 100 == 0:
            print(
                f"Total spawned actors at step: {self.steps_taken} spawned actors is: {self.total_spawned_actors}"
            )
            print(f"Total objects: {len(self.objects)}")

    def getProperties(self, obj, properties):
        actor = getattr(obj, "carlaActor", None)
        if actor is not None and actor.is_alive:
            return carla_actor_utils.actor_properties(actor, obj.parentOrientation)
        return None

    def destroy(self):
        for obj in self.objects:
            if not hasattr(obj, "carlaActor"):
                continue  # Remove this later -- just for CoSim testing purposes
            if obj.sensors:
                for sensor in obj.sensors.values():
                    if sensor.carla_sensor is not None and sensor.carla_sensor.is_alive:
                        sensor.carla_sensor.stop()
                        sensor.carla_sensor.destroy()
            if obj.carlaActor is not None:
                if isinstance(obj.carlaActor, carla.Vehicle):
                    obj.carlaActor.set_autopilot(False, self.tm.get_port())
                if isinstance(obj.carlaActor, carla.Walker):
                    carla_actor_utils.destroy_walker_controller(obj.carlaController)
                obj.carlaActor.destroy()
        if self.render and self.cameraManager:
            self.cameraManager.destroy_sensor()

        self.client.stop_recorder()

        self.world.tick()
        super().destroy()
