"""Orchestrate CARLA/METS-R co-simulation and vehicle ownership.

Native state, protocol handling, and native routing live in ``metsr``. CARLA actor
support and controller integration live in the local utility modules.
"""

import math
import os
import sys

import pygame

from scenic.core.object_types import Object
from scenic.core.regions import CircularRegion
from scenic.core.simulators import SimulationCreationError
from scenic.core.vectors import Orientation, Vector
from scenic.domains.driving.simulators import DrivingSimulation, DrivingSimulator
from scenic.simulators.carla import actor_utils, coordinate_utils
from scenic.simulators.carla.global_route_planner import GlobalRoutePlanner
from scenic.simulators.cosim import pcla_utils, runtime_utils
from scenic.simulators.metsr import bridge_utils, network_utils, routing_utils
from scenic.simulators.metsr.bridge_utils import METSRControlError, METSRCosimState
from scenic.simulators.metsr.client import METSRClient
from scenic.simulators.metsr.server_utils import build_metsr_vis_url
from scenic.syntax.veneer import verbosePrint

try:
    import carla
except ImportError as e:
    raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

from scenic.simulators.cosim import bubble_utils


class CosimSimulator(DrivingSimulator):
    """Reusable CARLA/METS-R connections and options for one implementation.

    Scenic calls createSimulation for each run. CosimSimulation below is that
    per-run state, not a second simulator backend. Rendering is controlled by
    render, CARLA recording by record, and run statistics by run_name.
    """

    def __init__(
        self,
        carla_map,
        map_path,
        xml_map,
        bubble_size=50,
        address="127.0.0.1",
        carla_port=2000,
        metsr_host="localhost",
        metsr_port=4000,
        timestep=0.1,
        sim_timestep=0.05,
        traffic_manager_port=None,
        metsr_sim_dir=None,
        timeout=20,
        verbose=False,
        render=True,
        record="",
        run_name=None,
        metsr_viz_port=8765,
        metsr_render_freq=None,
        network_metadata=None,
    ):
        super().__init__()
        self.timestep = timestep
        self.sim_timestep = sim_timestep

        if self.timestep < self.sim_timestep:
            if self.sim_timestep % self.timestep == 0:
                self.sim_ticks_per_carla = int(self.sim_timestep / self.timestep)
                self.sim_ticks_per_metsr = 1
            else:
                assert (
                    False
                ), f"Cannot correctly identify timestep with metsr tick: {self.sim_timestep} and carla tick: {self.timestep}"
        elif self.timestep == self.sim_timestep:
            self.sim_ticks_per_metsr = self.sim_ticks_per_carla = 1
        else:
            assert (
                False
            ), f"Invalid timestep with METSR: {sim_timestep} and CARLA: {timestep}"

        self.map_path = map_path
        self.network_metadata = (
            network_utils.load_network_metadata(xml_map)
            if network_metadata is None
            else network_metadata
        )

        self.bubble_size = bubble_size
        self.render = render
        self.record = record
        self.run_name = run_name
        self.metsr_sim_dir = metsr_sim_dir
        self.metsr_render_freq = runtime_utils.render_step_interval(
            timestep, metsr_render_freq
        )

        # Setting up the Carla Simulator
        verbosePrint(f"Connection to CARLA on port {carla_port}")
        self.carla_client = carla.Client(address, carla_port)
        self.carla_client.set_timeout(timeout)
        if carla_map is not None:
            try:
                print(f"Loading carla world: {carla_map}")
                self.world = self.carla_client.load_world(carla_map)
            except Exception as e:
                raise RuntimeError(f"CARLA could not load world '{carla_map}'") from e
        else:
            print(f"Loading carla Map: {map_path}")
            # TODO figure out how to properly do the map handling here
            if str(map_path).endswith(".xodr"):
                with open(map_path) as odr_file:
                    self.world = self.carla_client.generate_opendrive_world(
                        odr_file.read()
                    )
            else:
                raise RuntimeError("CARLA only supports OpenDrive maps")

        if traffic_manager_port is None:
            traffic_manager_port = carla_port + 6000
            assert (
                traffic_manager_port != metsr_port
            ), f"Specified Traffic manager port {traffic_manager_port} is not available"
        self.tm = self.carla_client.get_trafficmanager(traffic_manager_port)
        self.tm.set_synchronous_mode(True)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        assert sim_timestep <= 0.1, "timestep must be less that 0.1"
        settings.fixed_delta_seconds = sim_timestep
        self.world.apply_settings(settings)
        verbosePrint("Map loaded in simulator.")

        verbosePrint("Carla was initialized correctly proceeding to Metsr")
        self.metsr_viz_port = metsr_viz_port

        # Setting up Metsr simulator
        self.metsr_client = METSRClient(
            host=metsr_host, port=metsr_port, sim_folder=metsr_sim_dir, verbose=verbose
        )

        if self.render:
            verbosePrint(
                "Starting METS-R visualization server, client will timeout in 30 seconds"
            )
            self.metsr_client.start_viz(
                server_port=self.metsr_viz_port, startup_timeout=60
            )
            print(f"Please connect to METS-R at port: {self.metsr_viz_port}")

        verbosePrint("Clients have successfully been initialized")

    def createSimulation(self, scene, *, timestep, **kwargs):
        if timestep is not None and timestep != self.timestep:
            raise RuntimeError(
                "cannot customize timestep for individual co-simulations; "
                "set timestep when creating the CosimSimulator instead"
            )
        return CosimSimulation(
            scene=scene,
            carla_client=self.carla_client,
            metsr_client=self.metsr_client,
            timestep=self.timestep,
            sim_ticks_per_carla=self.sim_ticks_per_carla,
            sim_ticks_per_metsr=self.sim_ticks_per_metsr,
            tm=self.tm,
            bubble_size=self.bubble_size,
            render=self.render,
            record=self.record,
            network_metadata=self.network_metadata,
            run_name=self.run_name,
            metsr_viz_port=self.metsr_viz_port,
            metsr_render_freq=self.metsr_render_freq,
            **kwargs,
        )

    def destroy(self):
        self.metsr_client.stop_viz()
        self.metsr_client.close()
        super().destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(False)


class CosimSimulation(DrivingSimulation):
    """One run created by CosimSimulator, following Scenic's Simulation lifecycle.

    All rendering, recording, and controller options share this execution path.
    """

    def __init__(
        self,
        scene,
        carla_client,
        metsr_client,
        timestep,
        sim_ticks_per_carla,
        sim_ticks_per_metsr,
        tm,
        render,
        record,
        network_metadata,
        metsr_render_freq=None,
        bubble_size=100,
        run_name=None,
        metsr_viz_port=8080,
        **kwargs,
    ):
        # Carla and metrs simulators
        self.carla_client = carla_client
        self.metsr_client = metsr_client
        self.timestep = timestep  # Timestep for each step
        self.sim_ticks_per_carla = sim_ticks_per_carla
        self.sim_ticks_per_metsr = sim_ticks_per_metsr
        # Initializing CARLA params
        self.tm = tm  # Carla Traffic manager
        self.carla_world = self.carla_client.get_world()
        self._pre_actor_teardown_callbacks = []
        self.map = self.carla_world.get_map()
        self.blueprintLib = self.carla_world.get_blueprint_library()
        self.render = render
        self.record = record
        self.cameraManager = None

        self.metsr_state = METSRCosimState(metadata=network_metadata)
        self._carla_waypoints_by_key = None
        self.count = 0
        self.metsr_viz_port = metsr_viz_port
        self.metsr_render_freq = runtime_utils.render_step_interval(
            timestep, metsr_render_freq
        )
        self.stream_url = f"ws://127.0.0.1:{self.metsr_viz_port}"

        # CoSim related params
        self.bubble_size = bubble_size
        self.bubble_roads = []
        self.workspace = scene.workspace
        self.bubble_spawn_queue = set()
        self.metsr_actors = []
        self.carla_actors = []
        # Native METS-R vehicles represented as static CARLA obstacles. These
        # actors never participate in Scenic behaviors or ownership handoffs.
        # Private and public vehicle IDs occupy separate namespaces.
        self.boundary_actors = {}
        self._carla_spawn_property_seeds = {}
        self.spawn_points = self.carla_world.get_map().get_spawn_points()
        self.completed_route = {}
        self._carla_destination_plans = {}
        self.road_pop_density = {}
        self.grp = None
        self.run_name = run_name

        # For tracking / data collection
        self.bubble_sizes = []
        self.total_active_vehicles = []

        super().__init__(scene, timestep=timestep, **kwargs)

    def setup(self) -> None:
        """Reset METS-R, configure the world, and create the Scenic scene."""
        self.metsr_client.reset()
        # Start the visualization once
        verbosePrint("Initializing METS-R visualization server")

        self.metsr_state.valid_roads = self.metsr_client.query_road()["roadIds"]

        self.bubble_topology = bubble_utils.BubbleTopology.from_network(
            self.workspace.network
        )

        self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2.0)

        weather = self.scene.params.get("weather")
        if weather is not None:
            if isinstance(weather, str):
                self.carla_world.set_weather(getattr(carla.WeatherParameters, weather))
            elif isinstance(weather, dict):
                self.carla_world.set_weather(carla.WeatherParameters(**weather))

        if self.render:
            runtime_utils.initialize_display(self)

        if self.record:
            print("starting recording")
            if not os.path.exists(self.record):
                os.mkdir(self.record)
            name = "{}/scenario{}.log".format(self.record, self.scenario_number)
            # Carla is looking for an absolute path, so convert it if necessary.
            name = os.path.abspath(name)
            self.carla_client.start_recorder(name)

        # Create objects.
        super().setup()
        self._synchronize_boundary_vehicles()

        # TODO create a dict mapping for maps and IDS
        # Build METSR Visualization url
        self.map_id = 12

        world_map_name = self.map.name
        print(f"World map: {world_map_name}")

        if self.render:
            print("Creating METS-R visualization server")
            build_metsr_vis_url(
                viz_url="https://engineering.purdue.edu/HSEES/METSRVis/",
                stream_url=self.stream_url,
                map_id=self.map_id,
                vehicle_id=self.getMetsrPrivateVehId(self.ego),
                vehicle_type=1,
            )
            print(f"Connect to Server at: {self.stream_url}")

        for obj in self.objects:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.apply_control(
                    carla.VehicleControl(manual_gear_shift=False)
                )
        self.carla_world.tick()

        if self.render:
            runtime_utils.initialize_camera(self)

        self.carla_world.tick()  ## allowing manualgearshift to take effect

        # The core reads every object's properties immediately after setup,
        # including native NPCs which have not reached the first step yet.
        self.obj_data_cache = bridge_utils.collect_metsr_vehicle_data(
            self.metsr_client,
            {
                obj: self.getMetsrPrivateVehId(obj)
                for obj in self.metsr_actors + self.carla_actors
            },
        )

        for obj in self.scene.objects:
            if obj.carla_actor_flag:
                if obj.speed is not None and obj.speed != 0:
                    raise RuntimeError(
                        f"object {obj} cannot have a nonzero initial speed "
                        "(this is not yet possible in CARLA)"
                    )

    def createObjectInMetsr(
        self, obj: Object, route_generation_attempts=100, origin: str = None
    ) -> None:
        """Choose a reachable destination and create a native or CARLA-owned trip."""
        requested = routing_utils.select_spawn_placement(self, obj, origin)
        obj.route = routing_utils.choose_destination_route(
            self, obj, requested, route_generation_attempts
        )
        vehicle_id = self.getMetsrPrivateVehId(obj)
        if not requested["is_cosim"]:
            bridge_utils.require_metsr_response(
                self.metsr_client.generate_trip_between_roads(
                    vehID=vehicle_id, origin=obj.route[0], destination=obj.route[-1]
                ),
                "generateTripsByRoad",
            )
            return

        orientation = obj.parentOrientation * Orientation.fromEuler(
            obj.yaw, obj.pitch, obj.roll
        )
        pose = dict(
            vehID=vehicle_id,
            x=obj.position.x,
            y=obj.position.y,
            z=obj.position.z,
            bearing=-math.degrees(orientation.eulerAngles[0]) % 360,
            speed=0.0,
            private_veh=True,
            transform_coords=True,
            destination_road_id=obj.route[-1],
        )
        try:
            placement = bridge_utils.require_metsr_response(
                self.metsr_client.initialize_cosim_vehicle(
                    **pose,
                    segment_id=requested["segment_id"],
                    connector_path_id=requested["connector_path_id"],
                ),
                "initializeCoSimVeh",
            )
        except METSRControlError as error:
            if (
                error.record is None
                or error.record.get("errorCode") != "NO_MAP_MATCH"
                or requested["connector_path_id"] is not None
            ):
                raise
            # Adjacent road geometry can reject a road hint at a junction.
            # Retry road placement once, preserving exact connector selections.
            placement = bridge_utils.require_metsr_response(
                self.metsr_client.initialize_cosim_vehicle(**pose, segment_id=None),
                "initializeCoSimVeh",
            )
        state = self.metsr_client.query_vehicle(
            vehicle_id, private_veh=True, transform_coords=True
        )["data"][0]
        routing_utils.validate_spawn_placement(
            obj, requested, placement["data"][0], state
        )

    def createObjectInCarla(
        self, obj: Object, update_velocity: bool = False, *, native_state=None
    ) -> bool:
        """Spawn or queue a CARLA actor, then initialize its motion state."""
        blueprint, transform, native_velocity = runtime_utils.prepare_carla_spawn(
            self, obj, native_state
        )
        carlaActor = None
        spawn_error = None
        try:
            carlaActor = self.carla_world.try_spawn_actor(blueprint, transform)
        except Exception as exc:
            spawn_error = exc

        if carlaActor is None:
            runtime_utils.clear_carla_route_state(self, obj)
            first_failure = obj not in self.bubble_spawn_queue
            self.bubble_spawn_queue.add(obj)
            if first_failure:
                if spawn_error is not None:
                    print(f"Exception while spawning {obj.name}: {spawn_error}")
                print(
                    f"CARLA rejected actor {obj.name} at {transform.location}; "
                    f"Scenic position {obj.position}, rotation {transform.rotation}. "
                    "The actor will remain queued and retry."
                )
                threshold = 1.2 * max(obj.length, obj.width)
                is_close, danger_veh = runtime_utils.within_threshold_to(
                    obj, self.carla_actors, threshold=threshold, verbose=True
                )
                if danger_veh is not None:
                    veh_data = bridge_utils.collect_metsr_vehicle_data(
                        self.metsr_client,
                        {
                            obj: self.getMetsrPrivateVehId(obj)
                            for obj in [obj, danger_veh]
                        },
                    )
                    print(
                        f"Nearest tracked actor: {danger_veh.name}; "
                        f"within diagnostic threshold: {is_close}"
                    )
                    for veh in (obj, danger_veh):
                        print(
                            "METS-R position:",
                            veh_data[veh]["x"],
                            veh_data[veh]["y"],
                        )
                        print(f"Scenic position: {veh.position}")
            return False

        obj.carlaActor = carlaActor
        carlaActor.set_simulate_physics(obj.physics)

        try:
            actor_utils.configure_actor(obj, self.carla_world, self.blueprintLib)
        except SimulationCreationError:
            runtime_utils.clear_carla_route_state(self, obj)
            raise

        scenic_velocity = Vector(0, 0, 0)
        if update_velocity:
            scenic_velocity = (
                native_velocity if native_velocity is not None else obj.velocity
            )
            velocity = carla.Vector3D(
                scenic_velocity.x, -scenic_velocity.y, scenic_velocity.z
            )
            carlaActor.set_target_velocity(velocity)

        runtime_utils.seed_carla_spawn_properties(self, obj, transform, scenic_velocity)

        return True

    def createObjectInSimulator(self, obj: Object) -> None:
        """Create Scenic objects and assign their initial simulator owner."""
        assert obj.origin, "All objects must have an origin"
        assert obj.destination, "All objects must have an destination"

        assert hasattr(
            obj, "carla_actor_flag"
        ), "All objects must have attribute: carla_actor_flag"
        if obj == self.objects[0]:  # Special handling for ego
            self.ego = obj
            self.spawn_ego(obj)
        else:
            self.createObjectInMetsr(obj)
            obj.finished_route = False  # Track route completion for autopilot
            if self.count == 0:
                self.metsr_client.tick()  # allow obj to enter road if possible
            obj.carla_actor_flag = False
            self.metsr_actors.append(obj)
        obj.active_autopilot = False
        obj.autopilot_action = False
        obj.trip_start = 0

    def spawn_ego(self, obj: Object) -> None:
        """Acquire the ego bubble and create the ego in both simulators."""
        obj.bubble = CircularRegion(
            center=[obj.position.x, obj.position.y], radius=self.bubble_size
        )

        bubble_roads = bubble_utils.get_bubble_roads(self)
        selection = self._select_bubble_roads(bubble_roads)
        self.freeze_roads(selection.entering)
        self.metsr_client.tick()

        self.createObjectInMetsr(obj)  # Set the METSR vehicle origin to match ego spawn
        self.metsr_client.tick()  # Allow the vehicle to spawn

        spawn_success = self.createObjectInCarla(
            obj
        )  # spawn ego in updated location and update orientation
        assert spawn_success, f"Invalid spawn selection point at : {obj.position}"
        self._commit_promotion(obj)

    def getProperties(
        self, obj: Object, properties: dict
    ) -> dict[str, float | Vector | int]:
        """Read motion from the current owner, using admission seeds until its snapshot."""
        assert hasattr(
            obj, "carla_actor_flag"
        ), "Object is not assigned properly to a simulator instance"
        if obj.carla_actor_flag:
            seeds = self._carla_spawn_property_seeds
            if obj in seeds:
                # Setup also advances CARLA directly. Its cached world snapshot
                # identifies publication without assuming a nonzero actor pose.
                snapshot = self.carla_world.get_snapshot()
                if snapshot.find(obj.carlaActor.id) is None:
                    return dict(seeds[obj])
                seeds.pop(obj)
            return actor_utils.actor_properties(obj.carlaActor, obj.parentOrientation)
        return bridge_utils.metsr_properties(
            self.obj_data_cache[obj], obj.parentOrientation
        )

    def getMetsrPrivateVehId(self, obj: Object) -> int:
        """Allocate and return the private METS-R ID for a Scenic object."""
        if obj not in self.metsr_state.private_vehicle_ids:
            self.metsr_state.private_vehicle_ids[obj] = self.metsr_state.next_vehicle_id
            self.metsr_state.next_vehicle_id += 1
        return self.metsr_state.private_vehicle_ids[obj]

    def tick_carla(self) -> None:
        """Advance CARLA and retire admission seeds after its first snapshot."""
        for _ in range(self.sim_ticks_per_carla):
            self.carla_world.tick()
            self._carla_spawn_property_seeds.clear()

    def _synchronize_boundary_vehicles(self):
        """Mirror native exit occupancy without acquiring METS-R vehicle control.

        METS-R selects vehicles near downstream lane entries.
        Its coordinateTrail contains upcoming route points, so a separate
        batched vehicle query supplies the current pose. Obstacles stay fixed
        throughout all CARLA substeps.
        """
        scenic_objects = {
            (True, str(vehicle_id)): obj
            for obj, vehicle_id in self.metsr_state.private_vehicle_ids.items()
        }
        states = bridge_utils.native_boundary_states(
            self.metsr_client,
            self.metsr_state,
            {
                vehicle_id
                for obj, vehicle_id in self.metsr_state.private_vehicle_ids.items()
                if getattr(obj, "carlaActor", None) is not None
            },
        )
        # Validate every pose before changing obstacle membership.
        for key, state in states.items():
            runtime_utils.boundary_vehicle_transform(self, state, scenic_objects.get(key))

        for key in tuple(self.boundary_actors):
            if key not in states:
                runtime_utils.remove_boundary_actor(self, key)
        for key, state in states.items():
            obj = scenic_objects.get(key)
            actor = self.boundary_actors.get(key)
            if actor is not None and not runtime_utils.carla_actor_is_alive(actor):
                runtime_utils.remove_boundary_actor(self, key)
                actor = None
            if actor is None:
                blueprint = runtime_utils.boundary_vehicle_blueprint(
                    self, key, state, obj
                )
                transform = runtime_utils.boundary_vehicle_transform(self, state, obj)
                actor = self.carla_world.try_spawn_actor(blueprint, transform)
                if actor is None:
                    raise SimulationCreationError(
                        f"CARLA could not spawn METS-R boundary vehicle {key} "
                        f"at {transform.location}"
                    )
                # Track immediately so even a configuration failure is cleaned
                # up during final teardown. Never register with Traffic Manager.
                self.boundary_actors[key] = actor
                actor.set_autopilot(False, self.tm.get_port())
                actor.set_simulate_physics(False)
            actor.set_transform(
                runtime_utils.boundary_vehicle_transform(self, state, obj, actor)
            )

    def tick_metsr(self) -> None:
        """
        Docstring for tick_metsr

        Tick Metsr client for a single step
        """
        for _ in range(self.sim_ticks_per_metsr):
            self.metsr_client.tick()

    def _select_bubble_roads(self, roads):
        """Commit the mapping result explicitly before acquiring the new bubble."""
        selection = routing_utils.classify_bubble_roads(
            self.metsr_state, roads, self.bubble_topology.roads_by_id
        )
        self.bubble_roads = selection.roads
        self.metsr_state.active_bubble_roads = set(selection.native_road_ids)
        return selection

    def step(self) -> None:
        """Update bubble ownership, advance both simulators, and publish state."""
        self.road_pop_density = {road: 0 for road in self.metsr_state.valid_roads}
        bubble_roads = bubble_utils.get_bubble_roads(
            self, CircularRegion(center=[self.ego.x, self.ego.y], radius=self.bubble_size)
        )
        selection = self._select_bubble_roads(bubble_roads)
        bubble_roads = selection.roads
        # Acquire the new roads before advancing CARLA. Old roads remain owned
        # until after CARLA's current pose has been published and all resulting
        # handbacks and completions have been applied below.
        self.freeze_roads(selection.entering)
        # Remove obsolete obstacles before promotion/queue admission, and make
        # native exit occupancy visible before any CARLA physics/sensor tick.
        self._synchronize_boundary_vehicles()
        self._service_metsr_entering_queues()
        intersections = bubble_utils.intersections_for_roads(
            self.bubble_topology, bubble_roads
        )
        # Promote newly-controlled vehicles before CARLA advances, but defer
        # demotion until after CARLA's authoritative pose has been mirrored to
        # METS-R. This removes the stale-snapshot boundary heuristic.
        # METS-R has not advanced between promotion and CARLA publication, so
        # one batched snapshot can serve promotion and synchronization.
        state_objects = list(dict.fromkeys((*self.objects[1:], *self.carla_actors)))
        step_vehicle_data = bridge_utils.collect_metsr_vehicle_data(
            self.metsr_client,
            {obj: self.getMetsrPrivateVehId(obj) for obj in state_objects},
        )
        self.update_bubble_objects(vehicle_data=step_vehicle_data)

        self._refresh_carla_destination_paths(step_vehicle_data)
        self.tick_carla()
        self.synchronize_clients(vehicle_data=step_vehicle_data)
        self._remove_completed_carla_objects()
        releasable_roads = bridge_utils.releasable_old_roads(
            self.metsr_state, selection.leaving, self.carla_actors
        )
        if releasable_roads:
            self.release_roads(releasable_roads, allow_defer=True)
        self.tick_metsr()
        self.obj_data_cache = bridge_utils.collect_metsr_vehicle_data(
            self.metsr_client,
            {
                obj: self.getMetsrPrivateVehId(obj)
                for obj in self.metsr_actors + self.carla_actors
            },
        )

        if self.render and self.count % self.metsr_render_freq == 0:
            self.cameraManager.render(self.display)
            pygame.display.flip()
            try:
                self.metsr_client.render()
            except Exception:
                if self.count % 100 == 0:
                    print("Warning no registered connection to streaming client")

        runtime_utils.record_step_statistics(self)

        self.count += 1
        # Update the Scenic region used by behaviors on the next step.
        self.ego.bubble = CircularRegion(
            center=[self.objects[0].x, self.objects[0].y], radius=self.bubble_size
        )
        for road in bubble_roads:
            self.ego.bubble = self.ego.bubble.union(road)
        for intersection in intersections:
            self.ego.bubble = self.ego.bubble.union(intersection)

    def initiate_autopilot(self, obj: Object) -> bool:
        """
        docstring for initiate_autopilot

        :param obj: Target vehicle to initiate autopilot for
        :type obj: Vehicle object

        Activates autopilot for a simulation vehicle
         (i) If the vehicle is in CARLA, enables Traffic Manager after its
             destination path has been installed during admission or executeActions
         (2) If the vehicle is in METSR updates the overwrite flag for manually controlling the vehicle
        """
        if obj.carla_actor_flag:
            self._enable_carla_autopilot(obj)
            success = True
        else:
            obj.override_autopilot = False
            success = True

        return success

    def _enable_carla_autopilot(self, obj):
        """Initialize CARLA control, including before promotion is committed."""
        obj.carlaActor.set_autopilot(True, self.tm.get_port())
        self.tm.auto_lane_change(obj.carlaActor, True)
        obj.active_autopilot = True
        obj.autopilot_action = True

    def _service_metsr_entering_queues(self):
        """Admit ready Scenic vehicles from controlled-road departure queues."""
        bridge_utils.check_queue_admission_spawn_deadlines(
            self.metsr_state,
            self.count,
            {
                vehicle_id
                for obj, vehicle_id in self.metsr_state.private_vehicle_ids.items()
                if getattr(obj, "carla_actor_flag", False)
            },
        )
        if not self.metsr_state.controlled_roads:
            return
        response = bridge_utils.require_metsr_response(
            self.metsr_client.query_cosim_entering_vehicle_queue(),
            "coSimEnteringVehicleQueue",
        )
        requests = bridge_utils.select_queue_admissions(self.metsr_state, response)
        if not requests:
            return
        response = self.metsr_client.enter_road_from_queue(
            requests=requests[0] if len(requests) == 1 else requests
        )
        for vehicle_id, road_id in bridge_utils.validate_queue_admissions(
            response, requests
        ).items():
            self.metsr_state.admitted_vehicles[vehicle_id] = {
                "roadId": road_id,
                "admittedAt": self.count,
            }

    def _publish_carla_vehicle_updates(self, updates):
        """Publish each pose once, with a native target only for handback."""
        if not updates:
            return

        def field(name):
            values = [update.get(name) for update in updates]
            return values[0] if len(values) == 1 else values

        targets = [update.get("handoff") for update in updates]
        selectors = {}
        if any(targets):
            for index, name in enumerate(
                ("segment_id", "lane_index", "connector_path_id")
            ):
                values = [target[index] if target else None for target in targets]
                if any(value is not None for value in values):
                    selectors[name] = values[0] if len(values) == 1 else values
        response = self.metsr_client.teleport_cosim_vehicle(
            field("vehicle_id"),
            field("x"),
            field("y"),
            z=field("z"),
            bearing=field("bearing"),
            speed=field("speed"),
            private_veh=field("private_veh"),
            transform_coords=field("transform_coords"),
            **selectors,
        )
        acknowledgements = bridge_utils.validate_pose_update_acknowledgements(
            response, updates, targets
        )

        for update, target in zip(updates, targets):
            obj = update["object"]
            record = acknowledgements[str(update["vehicle_id"])]
            if target:
                self.remove_bubble_object(obj)
                continue
            segment = record.get("segmentId")
            self.metsr_state.authoritative_segments[obj] = (
                str(segment) if segment is not None else None
            )
            destination = update["vehicle_state"].get("destinationRoadId")
            if destination is None:
                destination = update["cosim_record"].get("destinationRoadId")
            if (
                destination is not None
                and segment is not None
                and record.get("segmentType") == "road"
                and str(segment) == str(destination)
                and obj not in self.completed_route
            ):
                if obj is not getattr(self, "ego", None):
                    retirement = self.metsr_client.reach_dest(
                        update["vehicle_id"], private_veh=update["private_veh"]
                    )
                    bridge_utils.validate_completion_response(
                        retirement, update["vehicle_id"]
                    )
                self._commit_retirement(obj)

    def _remove_completed_carla_objects(self):
        """Remove completed NPCs once, after native retirement is acknowledged."""
        for obj in tuple(self.carla_actors):
            if obj is getattr(self, "ego", None) or obj not in self.completed_route:
                continue
            self.remove_bubble_object(obj)

    def synchronize_clients(
        self,
        obj: Object | list[Object] = None,
        *,
        vehicle_data=None,
    ):
        """Publish CARLA states without required segment hints inside the region.

        Only an outgoing NPC supplies a native handback target. METS-R's
        acknowledgment determines when CARLA ownership can end.
        """
        if obj is None:
            carla_actors = list(self.carla_actors)
        elif isinstance(obj, list):
            carla_actors = list(obj)
        else:
            carla_actors = [obj]
        if not carla_actors:
            return

        all_veh_data = (
            vehicle_data
            if vehicle_data is not None
            else bridge_utils.collect_metsr_vehicle_data(
                self.metsr_client,
                {obj: self.getMetsrPrivateVehId(obj) for obj in carla_actors},
            )
        )
        cosim_response = bridge_utils.require_metsr_response(
            self.metsr_client.query_cosim_vehicle(), "coSimVehicle"
        )
        live_private_records = {}
        for record in cosim_response.get("data", ()):
            if not isinstance(record, dict) or record.get("isPrivate") is not True:
                continue
            vehicle_id = record.get("vehicleId")
            if vehicle_id is not None:
                live_private_records.setdefault(str(vehicle_id), []).append(record)

        pending_updates = []
        for carla_object in carla_actors:
            try:
                actor = carla_object.carlaActor
                if not runtime_utils.carla_actor_is_alive(actor):
                    raise RuntimeError("CARLA actor is no longer alive")
                location = actor.get_location()
            except Exception as error:
                raise SimulationCreationError(
                    f"Cannot read CARLA actor {getattr(carla_object, 'name', carla_object)}: "
                    f"{error}. Aborting this run so METS-R can be reset."
                ) from error
            vehicle_state = all_veh_data[carla_object]
            vehicle_id = self.getMetsrPrivateVehId(carla_object)
            records = live_private_records.get(str(vehicle_id), ())
            if len(records) > 1:
                raise SimulationCreationError(
                    f"Expected at most one live private METS-R co-simulation record for "
                    f"{carla_object} (vehicle {vehicle_id}) during synchronization; "
                    f"found {len(records)}"
                )
            # CARLA is authoritative while the actor is in carla_actors. This
            # snapshot supplies optional connector/route metadata only; the
            # teleport acknowledgement determines whether METS-R accepted the
            # authoritative update or completed a native ownership transfer.
            cosim_record = records[0] if records else {}

            transform = actor.get_transform()
            bearing = coordinate_utils.carlaToCompassBearing(transform.rotation.yaw)
            velocity = actor.get_velocity()
            speed = math.sqrt(
                velocity.x * velocity.x
                + velocity.y * velocity.y
                + velocity.z * velocity.z
            )

            pending_updates.append(
                {
                    "object": carla_object,
                    "location": location,
                    "vehicle_id": vehicle_id,
                    "x": location.x,
                    "y": -location.y,
                    "z": location.z,
                    "bearing": bearing,
                    "speed": speed,
                    "private_veh": True,
                    "transform_coords": True,
                    "handoff": routing_utils.native_handoff_target(
                        self, carla_object, location
                    ),
                    "vehicle_state": vehicle_state,
                    "cosim_record": cosim_record,
                }
            )

        self._publish_carla_vehicle_updates(pending_updates)

        if runtime_utils.client_sync_diagnostic_due(self.count, bool(pending_updates)):
            runtime_utils.check_client_synchronization(self)

    def _initialize_promoted_carla_autopilot(self, obj, vehicle_state):
        """Install default NPC control before its first moving CARLA tick.

        A frame of uncontrolled movement can take an admitted actor past a
        junction fork before Traffic Manager initializes its waypoint buffer.
        The actor snapshot is not published yet, so plan from the same fresh
        native pose used to spawn it instead of querying actor.get_location().
        """
        if (
            obj is getattr(self, "ego", None)
            or getattr(obj, "pcla", None) is not None
            or not getattr(obj, "autopilot_action", False)
            or getattr(obj, "active_autopilot", False)
            or getattr(obj, "trajectory", None) is not None
        ):
            return False
        start = carla.Location(
            float(vehicle_state["x"]),
            -float(vehicle_state["y"]),
            float(vehicle_state.get("z") or 0.0),
        )
        self._install_carla_destination_path(obj, vehicle_state, target_start=start)
        self._enable_carla_autopilot(obj)
        obj._control = None
        return True

    def update_bubble_objects(self, *, vehicle_data=None) -> None:
        """Promote native vehicles in controlled segments before CARLA advances."""
        all_veh_data = (
            vehicle_data
            if vehicle_data is not None
            else bridge_utils.collect_metsr_vehicle_data(
                self.metsr_client,
                {obj: self.getMetsrPrivateVehId(obj) for obj in self.objects[1:]},
            )
        )
        for obj in self.objects[1:]:
            if obj in self.completed_route:
                continue
            veh_data = all_veh_data[obj]
            metsr_road = bridge_utils.metsr_vehicle_road(veh_data)
            if metsr_road is not None:
                self.road_pop_density[metsr_road] += 1
            if obj.carla_actor_flag:
                continue
            if metsr_road is None:
                # A new trip has no segment while it waits for native admission.
                if veh_data.get("queuedRoadId") is None:
                    self._commit_retirement(obj)
                continue
            if not bridge_utils.vehicle_record_in_cosim_region(
                self.metsr_state, veh_data
            ):
                continue
            if not self.createObjectInCarla(
                obj, update_velocity=True, native_state=veh_data
            ):
                # Already-present native vehicles need the same bounded spawn
                # retry as vehicles admitted through enterRoadFromQueue.
                self.metsr_state.admitted_vehicles.setdefault(
                    self.metsr_state.private_vehicle_ids[obj],
                    {
                        "roadId": metsr_road,
                        "admittedAt": self.count,
                    },
                )
                continue
            self._initialize_promoted_carla_autopilot(obj, veh_data)
            self._commit_promotion(obj, veh_data)

    def _commit_promotion(self, obj, vehicle_state=None):
        """Publish CARLA ownership only after actor and controller initialization."""
        if vehicle_state is not None:
            bridge_utils.initialize_carla_authoritative_segment(
                self.metsr_state, obj, vehicle_state
            )
        if obj in self.metsr_actors:
            self.metsr_actors.remove(obj)
        if obj not in self.carla_actors:
            self.carla_actors.append(obj)
        obj.carla_actor_flag = True
        self._clear_pending_admission(obj)

    def _clear_pending_admission(self, obj):
        vehicle_id = self.metsr_state.private_vehicle_ids.get(obj)
        self.metsr_state.admitted_vehicles.pop(vehicle_id, None)
        self.bubble_spawn_queue.discard(obj)

    def _commit_retirement(self, obj):
        """Record confirmed completion once; actor removal follows publication."""
        self._clear_pending_admission(obj)
        if obj not in self.completed_route:
            self.completed_route[obj] = True
            obj.finished_route = self.count

    def freeze_roads(self, keys: list[str]) -> None:
        """Acquire represented METS-R roads and commit confirmed connector ownership."""
        keys = {str(key) for key in keys}
        for key in keys:
            assert (
                key not in self.metsr_state.controlled_roads
            ), "Attempted to freeze already frozen lane"
        requested = sorted(key for key in keys if key in self.metsr_state.valid_roads)
        if not requested:
            return
        print(f"Freezing roads: {requested}")
        response = self.metsr_client.set_cosim_road(requested)
        bridge_utils.apply_road_takeover(self.metsr_state, response, requested)

    def release_roads(self, keys: list[str], *, allow_defer=False) -> None:
        """Return controlled roads to METS-R and update shared connectors."""
        keys = sorted({str(key) for key in keys})
        if not keys:
            return
        represented = set(map(str, self.metsr_state.valid_roads))
        for key in keys:
            assert (
                key in self.metsr_state.controlled_roads
            ), "Attempted to release non frozen lane"
            if key not in represented:
                raise METSRControlError(
                    f"Cannot release unrepresented METS-R road {key!r}"
                )

            # Commit local release only after native placement succeeds. The
            # server can reject a release when vehicle placement is blocked.
            response = self.metsr_client.release_cosim_road(key)
            bridge_utils.apply_road_release(
                self.metsr_state, key, response, allow_defer=allow_defer
            )

    def destroy_carla_obj(self, obj) -> None:
        """Destroy a live Scenic CARLA actor and its attached controller."""
        if pcla_utils.is_pcla_controlled_ego(self, obj):
            raise SimulationCreationError(
                "A PCLA-controlled ego cannot be destroyed during ordinary "
                "co-simulation synchronization; it is cleaned up once during "
                "final simulation teardown"
            )
        if obj.carlaActor is not None:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.set_autopilot(False, self.tm.get_port())
            if isinstance(obj.carlaActor, carla.Walker):
                actor_utils.destroy_walker_controller(obj.carlaController)
            runtime_utils.destroy_attached_carla_sensors(self, obj.carlaActor)
            obj.carlaActor.destroy()
            obj.carlaActor = None

    def remove_bubble_object(self, obj) -> None:
        """Release the CARLA actor and return the Scenic object to native control."""
        if pcla_utils.is_pcla_controlled_ego(self, obj):
            raise SimulationCreationError(
                "A PCLA-controlled ego cannot be demoted from CARLA into the "
                "METS-R background simulation"
            )
        self.destroy_carla_obj(obj)
        if obj.autopilot_action and obj.active_autopilot:
            obj.active_autopilot = False
            obj.trajectory = None
        runtime_utils.clear_carla_route_state(self, obj)
        self._carla_spawn_property_seeds.pop(obj, None)
        self._clear_pending_admission(obj)
        self.carla_actors.remove(obj)
        if obj not in self.metsr_actors:
            self.metsr_actors.append(obj)
        obj.carla_actor_flag = False

    def add_pre_actor_teardown_callback(self, callback) -> None:
        """Register cleanup which must run before Scenic destroys CARLA actors."""
        if not callable(callback):
            raise TypeError("pre-actor teardown callback must be callable")
        callbacks = getattr(self, "_pre_actor_teardown_callbacks", None)
        if callbacks is None:
            callbacks = self._pre_actor_teardown_callbacks = []
        callbacks.append(callback)

    def destroy(self) -> None:
        """
        Docstring for destroy

        Destroy both simulators instances i.e (METSR, CARLA)
        """
        if getattr(self, "_cosim_destroyed", False):
            return
        self._cosim_destroyed = True

        active_error = sys.exc_info()[0] is not None
        cleanup_error = None
        verbosePrint("Closing METS-R visualization server")
        print("Logging trip times")
        print("=" * 25)
        if getattr(self, "run_name", None) is not None:
            try:
                runtime_utils.log_trip_times(self)
            except Exception as error:
                runtime_utils.report_cleanup_failure("logging trip times", error)
        print("=" * 25)

        # Log first. Reset an aborted run (or a blocked release) using the
        # existing server lifecycle, without counting interrupted trips as arrivals.
        try:
            if active_error:
                self.metsr_client.reset()
            else:
                try:
                    self.release_roads(list(self.metsr_state.controlled_roads))
                except Exception as error:
                    runtime_utils.report_cleanup_failure(
                        "releasing METS-R roads; resetting the run", error
                    )
                    self.metsr_client.reset()
        except Exception as error:
            cleanup_error = error
            runtime_utils.report_cleanup_failure("resetting METS-R during cleanup", error)

        # METSR destroy
        try:
            if getattr(self.metsr_client, "verbose", False):
                print("Client Messages Log:")
                print("[")
                for call in self.metsr_client._messagesLog:
                    print(f"    {call},")
                print("]")
        except Exception as error:
            runtime_utils.report_cleanup_failure("printing the METS-R message log", error)

        # Each visualization owner closes its sensors before the parent vehicle.
        if getattr(self, "render", False) and getattr(self, "cameraManager", None):
            try:
                self.cameraManager.destroy_sensor()
            except Exception as error:
                runtime_utils.report_cleanup_failure("destroying the CARLA camera", error)
        runtime_utils.run_pre_actor_teardown_callbacks(self)
        for key in tuple(getattr(self, "boundary_actors", ())):
            try:
                runtime_utils.remove_boundary_actor(self, key)
            except Exception as error:
                runtime_utils.report_cleanup_failure(
                    "destroying a boundary vehicle", error
                )
        # "CARLA destroy"
        cleanup_objects = list(getattr(self, "carla_actors", ()))
        # Include actors whose initialization failed before promotion completed.
        cleanup_objects.extend(
            obj
            for obj in getattr(self, "objects", ())
            if getattr(obj, "carlaActor", None) is not None
            and not any(obj is existing for existing in cleanup_objects)
        )
        for obj in cleanup_objects:
            try:
                runtime_utils.destroy_carla_object_during_teardown(self, obj)
            except Exception as error:
                runtime_utils.report_cleanup_failure("cleaning up a CARLA object", error)
        self.metsr_state.authoritative_segments.clear()
        getattr(self, "_carla_destination_plans", {}).clear()

        try:
            self.carla_client.stop_recorder()
        except Exception as error:
            runtime_utils.report_cleanup_failure("stopping the CARLA recorder", error)

        try:
            super().destroy()
        except Exception as error:
            runtime_utils.report_cleanup_failure(
                "finalizing the Scenic simulation", error
            )

        if cleanup_error is not None and not active_error:
            raise cleanup_error

    def executeActions(self, allActions) -> None:
        """Apply Scenic actions through each vehicle's current simulator owner."""
        super().executeActions(allActions)
        for obj in self.agents:
            if obj.carla_actor_flag:  # Processing CARLA actors
                if (
                    not obj.autopilot_action and obj.active_autopilot
                ):  # Disable autopilot first to enable smooth transitions
                    obj.carlaActor.set_autopilot(False, self.tm.get_port())
                    obj.active_autopilot = False
                elif obj.autopilot_action and not obj.active_autopilot:
                    if getattr(obj, "trajectory", None) is None:
                        self._install_carla_destination_path(obj)
                    else:
                        self.tm.set_path(obj.carlaActor, obj.trajectory)
                    self.initiate_autopilot(obj)
                    obj._control = None
                else:
                    ctrl = obj._control
                    if ctrl is not None:
                        obj.carlaActor.apply_control(ctrl)
                        pcla_utils.notify_control_applied(obj, ctrl)
                        obj._control = None
            else:
                if not obj.autopilot_action:  # Default is autopilot
                    target_acc = getattr(obj, "target_acceleration", 0)
                    bridge_utils.require_metsr_response(
                        self.metsr_client.control_vehicle(
                            self.getMetsrPrivateVehId(obj), target_acc, private_veh=True
                        ),
                        "controlVeh",
                    )
                if hasattr(obj, "trajectory"):
                    if obj.trajectory and not obj.autopilot_action:
                        if not obj.trajectory_is_active:
                            metsr_trajectory = network_utils.generate_metsr_trajectory(
                                self.metsr_state, obj.trajectory
                            )
                            if metsr_trajectory:
                                bridge_utils.require_metsr_response(
                                    self.metsr_client.update_vehicle_route(
                                        self.getMetsrPrivateVehId(obj),
                                        metsr_trajectory,
                                        private_veh=True,
                                    ),
                                    "updateVehicleRoute",
                                )
                                obj.trajectory_is_active = True

    def _install_carla_destination_path(
        self, obj, vehicle_state=None, *, target_start=None
    ):
        """Plan and install one default path toward the native destination road.

        METS-R's routeRoadIds excludes the current road and does not constrain
        CARLA's choice of intermediate roads or lanes.
        """
        if vehicle_state is None:
            vehicle_state = bridge_utils.collect_metsr_vehicle_data(
                self.metsr_client, {obj: self.getMetsrPrivateVehId(obj)}
            )[obj]
        destination = routing_utils.vehicle_destination_road(obj, vehicle_state)
        path = routing_utils.generate_carla_destination_trajectory(
            self, destination, obj, target_start=target_start
        )
        self.tm.set_path(obj.carlaActor, path)
        obj.trajectory = path
        self._carla_destination_plans[obj] = (destination, path)
        return path

    def _refresh_carla_destination_paths(self, vehicle_data):
        """Refresh default TM paths when METS-R changes a vehicle's destination."""
        plans = self._carla_destination_plans
        for obj, (destination, path) in tuple(plans.items()):
            if getattr(obj, "trajectory", None) is not path:
                # A Scenic behavior supplied a different explicit trajectory.
                plans.pop(obj, None)
                continue
            if (
                not getattr(obj, "active_autopilot", False)
                or pcla_utils.is_pcla_controlled_ego(self, obj)
                or obj not in vehicle_data
            ):
                continue
            current_destination = routing_utils.vehicle_destination_road(
                obj, vehicle_data[obj]
            )
            if current_destination == destination:
                continue
            self._install_carla_destination_path(obj, vehicle_data[obj])
