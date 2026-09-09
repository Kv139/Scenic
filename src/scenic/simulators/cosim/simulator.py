from scenic.core.simulators import Simulation, Simulator
from scenic.core.vectors import Orientation, Vector
from scenic.syntax.veneer import verbosePrint
from scenic.simulators.metsr.client import METSRClient
from scenic.simulators.metsr.util import build_metsr_vis_url
from scenic.simulators.cosim.utils.utils import *
from scenic.core.regions import CircularRegion, PolygonalRegion
from scenic.core.object_types import Object
from scenic.core.simulators import SimulationCreationError, ObjectMissingInSimulation
from scenic.domains.driving.roads import Lane, Intersection, Road
from scenic.domains.driving.simulators import DrivingSimulation, DrivingSimulator

import pygame
import inspect
import sys
import warnings
import os
import math
import random
import scenic.simulators.cosim.utils.utils as _utils
import scenic.simulators.carla.utils.utils as utils
import pandas as pd
import networkx as nx

from .utils.network_helper import network_cache
from .utils.global_route_planner import GlobalRoutePlanner
import scenic.simulators.carla.utils.visuals as visuals
from scenic.simulators.carla.blueprints import oldBlueprintNames, carModels, busModels

try:
    import carla
except ImportError as e:
    raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

COSIM_ADMISSION_SPAWN_MAX_TICKS = 100
ROAD_ENTRY_HOLD_REPORT_INTERVAL = 100
FIRST_ROAD_LANE_CHANGE_DISTANCE = 10.0
FIRST_ROAD_ANCHOR_END_BUFFER = 2.0
PENDING_LANE_RECONCILIATION_MAX_SHIFT = 15.0
PENDING_LANE_RECONCILIATION_MAX_LONGITUDINAL_DELTA = 3.0
PENDING_LANE_RECONCILIATION_MAX_HEADING_DELTA = 30.0
PENDING_LANE_RECONCILIATION_MAP_TOLERANCE = 4.25
PENDING_LANE_RECONCILIATION_COLLISION_MARGIN = 0.5
CARLA_OWNED_PROJECTION_TOLERANCE = 4.25
METSR_CONNECTOR_NO_LANE = -1
ROAD_OBSERVATION_UNKNOWN = "unknown"
ROAD_OBSERVATION_SAME = "same"
ROAD_OBSERVATION_PREDECESSOR = "predecessor"
ROAD_OBSERVATION_DIRECT_SUCCESSOR = "direct_successor"
ROAD_OBSERVATION_ONE_SKIP = "one_skip"
ROAD_OBSERVATION_UNSUPPORTED = "unsupported"

def _render_step_interval(timestep, requested):
    """Resolve a render cadence, defaulting to one simulated second."""
    if requested is None:
        return max(1, int(round(1.0 / float(timestep))))
    interval = int(requested)
    if interval <= 0:
        raise ValueError("metsr_render_freq must be a positive integer")
    return interval



class METSRControlError(RuntimeError):
    """A native METS-R control response containing a failed operation."""

    def __init__(self, message, response=None, record=None):
        super().__init__(message)
        self.response = response
        self.record = record
        self.retryable = bool(
            isinstance(record, dict) and record.get("retryable", False)
        )


class _ScenicOwnedPCLAVehicle:
    """Read-only actor facade preventing legacy PCLA from destroying Scenic's ego."""

    def __init__(self, actor):
        self._actor = actor

    @property
    def id(self):
        return self._actor.id

    @property
    def is_alive(self):
        # PCLA.cleanup uses this flag to decide whether it owns destruction of
        # the vehicle. Sensors can still match this facade through ``id``.
        return False

    def destroy(self):
        return False

    def __getattr__(self, name):
        return getattr(self._actor, name)


def _require_metsr_response(response, operation):
    """Validate a native response envelope and all of its result records."""
    if not isinstance(response, dict):
        raise METSRControlError(
            f"METS-R returned no valid response for {operation}: {response!r}",
            response=response,
        )
    if response.get("messageType") != operation:
        raise METSRControlError(
            f"Expected {operation}, received {response.get('messageType')!r}",
            response=response,
        )
    records = response.get("data", ())
    if not isinstance(records, (list, tuple)):
        raise METSRControlError(
            f"METS-R returned invalid data for {operation}: {records!r}",
            response=response,
        )
    for record in records:
        if isinstance(record, dict) and record.get("status") == "error":
            error_code = record.get("errorCode")
            message = record.get("message")
            if error_code and message:
                detail = f"{error_code}: {message}"
            else:
                detail = message or error_code or "unknown error"
            vehicle_id = record.get("vehicleId")
            subject = f" vehicle {vehicle_id}" if vehicle_id is not None else ""
            raise METSRControlError(
                f"METS-R rejected{subject} during {operation}: {detail}",
                response=response,
                record=record,
            )
    status = response.get("status")
    if status not in {"ok", "partial"}:
        raise METSRControlError(
            response.get("message")
            or f"METS-R rejected {operation} with status {status!r}",
            response=response,
        )
    return response


def _require_all_success_control_response(response, operation):
    """Require a successful envelope before changing Scenic ownership state."""
    response = _require_metsr_response(response, operation)
    if response.get("status") != "ok":
        raise METSRControlError(
            f"METS-R returned non-success status {response.get('status')!r} "
            f"for {operation}",
            response=response,
        )
    return response


def _require_single_vehicle_response(response, operation, vehicle_id):
    """Require one successful acknowledgement for the requested vehicle."""
    response = _require_metsr_response(response, operation)
    if response.get("status") != "ok":
        raise METSRControlError(
            f"METS-R returned non-success status {response.get('status')!r} "
            f"for {operation}",
            response=response,
        )
    records = response.get("data", ())
    if len(records) != 1 or not isinstance(records[0], dict):
        raise METSRControlError(
            f"METS-R returned {len(records)} result records for {operation}; "
            f"expected one for vehicle {vehicle_id}",
            response=response,
        )
    record = records[0]
    returned_vehicle_id = record.get("vehicleId")
    if returned_vehicle_id is None or str(returned_vehicle_id) != str(vehicle_id):
        raise METSRControlError(
            f"METS-R acknowledged vehicle {returned_vehicle_id!r} during "
            f"{operation}; expected vehicle {vehicle_id}",
            response=response,
            record=record,
        )
    if record.get("status") not in (None, "ok"):
        raise METSRControlError(
            f"METS-R returned non-success vehicle result during {operation}",
            response=response,
            record=record,
        )
    return response, record


def _require_vehicle_batch_response(response, operation, vehicle_ids):
    """Require one successful acknowledgement per requested vehicle.

    METS-R accepts batched control records, but record order is not part of the
    protocol contract. Index acknowledgements by vehicle ID and reject missing,
    duplicate, or unexpected records before Scenic changes local ownership.
    """
    response = _require_metsr_response(response, operation)
    if response.get("status") != "ok":
        raise METSRControlError(
            f"METS-R returned non-success status {response.get('status')!r} "
            f"for {operation}",
            response=response,
        )

    requested = tuple(str(vehicle_id) for vehicle_id in vehicle_ids)
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate vehicle IDs requested for {operation}")

    acknowledgements = {}
    for record in response.get("data", ()):
        if not isinstance(record, dict):
            raise METSRControlError(
                f"METS-R returned an invalid result record for {operation}",
                response=response,
                record=record,
            )
        vehicle_id = record.get("vehicleId")
        key = None if vehicle_id is None else str(vehicle_id)
        if key not in requested:
            raise METSRControlError(
                f"METS-R acknowledged unexpected vehicle {vehicle_id!r} "
                f"during {operation}",
                response=response,
                record=record,
            )
        if key in acknowledgements:
            raise METSRControlError(
                f"METS-R acknowledged vehicle {vehicle_id!r} more than once "
                f"during {operation}",
                response=response,
                record=record,
            )
        if record.get("status") not in (None, "ok"):
            raise METSRControlError(
                f"METS-R returned non-success vehicle result during {operation}",
                response=response,
                record=record,
            )
        acknowledgements[key] = record

    missing = [
        vehicle_id for vehicle_id in requested
        if vehicle_id not in acknowledgements
    ]
    if missing:
        raise METSRControlError(
            f"METS-R omitted acknowledgements for vehicles {missing!r} "
            f"during {operation}",
            response=response,
        )
    return response, acknowledgements


def _metsr_vehicle_road(record):
    """Return the logical physical road represented by a native vehicle record."""
    if not isinstance(record, dict):
        return None
    on_connector = record.get("onConnector") or (
        str(record.get("segmentType", "")).lower() == "connector"
    )
    if on_connector:
        return (
            record.get("targetRoadId")
            or record.get("segmentId")
        )
    return record.get("segmentId")


class _PendingLaneReconciliationOccupied(SimulationCreationError):
    """A verified physical blocker makes an otherwise-safe lane move retryable."""


class _PendingLaneReconciliationUnavailable(SimulationCreationError):
    """No exact parallel target is currently safe at the observed pose."""


class CosimSimulator(DrivingSimulator):
    def __init__(self,
        carla_map,
        map_path,
        xml_map,
        bubble_size = 50, # Might be good to add some logic for what a minimal bubble size is so users cannot make it too small
        address="127.0.0.1",
        carla_port=2000,
        metsr_host="localhost", # Not sure what this actually means here
        metsr_port=4000,
        timestep=0.1, # Not entirely sure what the distinction between timestep and sim_timestep is in metsr
        sim_timestep=0.05,
        traffic_manager_port=None,
        metsr_sim_dir=None,
        timeout=20,
        verbose=False,
        render=True,
        record="",
        run_name=None,
        metsr_viz_port = 8765,
        metsr_render_freq=None,
    ):
        super().__init__()
        # TODO rename these this is terrible
        self.timestep = timestep
        self.sim_timestep = sim_timestep # This should represent the timestep recorded in the METSR config

        if self.timestep < self.sim_timestep:
            if self.sim_timestep % self.timestep == 0:
                self.sim_ticks_per_carla = int(self.sim_timestep / self.timestep)
                self.sim_ticks_per_metsr = 1
            else:
                assert False, f"Cannot correctly identify timestep with metsr tick: {self.sim_timestep} and carla tick: {self.timestep}"
        elif self.timestep == self.sim_timestep:
            self.sim_ticks_per_metsr = self.sim_ticks_per_carla = 1
        else:
            assert False, f"Invalid timestep with METSR: {sim_timestep} and CARLA: {timestep}"

        self.map_path = map_path
        self.xml_to_xodr_map = {}
        self.metsr_lane_indices = {}
        self.metsr_lane_connections = {}
        self.metsr_internal_lane_connections = {}
        self.metsr_internal_edge_connections = {}

        self.bubble_size = bubble_size
        self.render= render
        self.record = record
        self.run_name = run_name
        self.metsr_sim_dir = metsr_sim_dir
        self.metsr_render_freq = _render_step_interval(timestep, metsr_render_freq)

        # Setting up the Carla Simulator
        verbosePrint(f"Connection to CARLA on port {carla_port}")
        self.carla_client = carla.Client(address,carla_port)
        self.carla_client.set_timeout(timeout)
        """
        Need to figure out how to handle the map paths for this
        """
        if carla_map is not None:
            try:
                print(f"Loading carla world: {carla_map}")
                self.world = self.carla_client.load_world(carla_map)
                self.xml_to_xodr_map = _utils.generate_map(str(xml_map)) #convert pathlib obj to str for XML tree TODO what is best practice?
                self.metsr_lane_indices = _utils.generate_metsr_lane_index_map(
                    str(xml_map)
                )
                self.metsr_lane_connections = (
                    _utils.generate_metsr_lane_connection_map(str(xml_map))
                )
                self.metsr_internal_lane_connections = (
                    _utils.generate_metsr_internal_lane_connection_map(str(xml_map))
                )
                self.metsr_internal_edge_connections = (
                    _utils.generate_metsr_internal_edge_connection_map(str(xml_map))
                )
                self.xml_to_xodr_intersections = _utils.generate_signal_map(str(xml_map))
            except Exception as e:
                raise RuntimeError(f"CARLA could not load world '{carla_map}'") from e
        else:
            print(f"Loading carla Map: {map_path}")
            #TODO figure out how to properly do the map handling here
            if str(map_path).endswith(".xodr"):
                with open(map_path) as odr_file:
                    self.world = self.carla_client.generate_opendrive_world(odr_file.read())
            else:
                raise RuntimeError("CARLA only supports OpenDrive maps")

        if traffic_manager_port is None:
            traffic_manager_port = carla_port + 6000
            assert traffic_manager_port != metsr_port, f"Specified Traffic manager port {traffic_manager_port} is not available"
        self.tm = self.carla_client.get_trafficmanager(traffic_manager_port)
        self.tm.set_synchronous_mode(True)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        print(f"Relaxed timestep restriction for testing?")
        assert sim_timestep <= .1 , f"timestep must be less that 0.1"
        settings.fixed_delta_seconds = sim_timestep
        self.world.apply_settings(settings)
        verbosePrint("Map loaded in simulator.")

        # self.scenario_number = 0
        verbosePrint("Carla was initialized correctly proceeding to Metsr")
        self.metsr_viz_port = metsr_viz_port

        # Setting up Metsr simulator
        self.metsr_client = METSRClient(host=metsr_host,
                                port=metsr_port,
                                sim_folder=metsr_sim_dir,
                                verbose=verbose)

        if self.render:
            verbosePrint(f"Starting METS-R visualization server, client will timeout in 30 seconds")
            self.metsr_client.start_viz(server_port=self.metsr_viz_port, startup_timeout=60)
            print(f"Please connect to METS-R at port: {self.metsr_viz_port}")



        verbosePrint("Clients have successfully been initialized")

    def createSimulation(self,scene,*, timestep, **kwargs): #TODO: fix timestep
        if timestep is not None and timestep != self.timestep:
            raise RuntimeError(
                "cannot customize timestep for individual CARLA simulations; "
                "set timestep when creating the CarlaSimulator instead"
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
            mappings=self.xml_to_xodr_map,
            lane_indices=self.metsr_lane_indices,
            lane_connections=self.metsr_lane_connections,
            internal_lane_connections=self.metsr_internal_lane_connections,
            internal_edge_connections=self.metsr_internal_edge_connections,
            xml_to_xodr_intersections = self.xml_to_xodr_intersections,
            run_name= self.run_name,
            metsr_viz_port = self.metsr_viz_port,
            metsr_render_freq = self.metsr_render_freq,
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
    def __init__(self, scene, carla_client, metsr_client, timestep, sim_ticks_per_carla, sim_ticks_per_metsr, tm, render ,record, mappings, xml_to_xodr_intersections, lane_indices=None, lane_connections=None, internal_lane_connections=None, internal_edge_connections=None, metsr_render_freq=None, bubble_size=100, run_name=None, metsr_viz_port=8080, **kwargs ):

        # Carla and metrs simulators
        self.carla_client = carla_client
        self.metsr_client = metsr_client
        self.timestep = timestep    # Timestep for each step
        self.sim_ticks_per_carla = sim_ticks_per_carla
        self.sim_ticks_per_metsr = sim_ticks_per_metsr
        # Initializing CARLA params
        self.tm = tm # Carla Traffic manager
        self.carla_world = self.carla_client.get_world()
        self._pre_actor_teardown_callbacks = []
        self.map = self.carla_world.get_map()
        self.blueprintLib = self.carla_world.get_blueprint_library()
        self.carla_cameraManager = None
        self.render = render
        self.record = record
        self.cameraManager = None

        # Initializing METSR params
        self.next_pv_id = 0
        self.pv_id_map = {}
        self.frozen_vehicles = set()
        self.scenic_to_metsr_map = mappings
        # ``None`` means an older caller did not provide an authoritative XML
        # index map; an empty dict means XML parsing found no usable lanes.
        self.metsr_lane_indices = lane_indices
        self.metsr_lane_connections = lane_connections or {}
        self.metsr_internal_lane_connections = internal_lane_connections or {}
        self.metsr_internal_edge_connections = internal_edge_connections or {}
        self._sumo_lane_to_carla_keys = {}
        for carla_key, sumo_lanes in mappings.items():
            for sumo_lane in sumo_lanes:
                self._sumo_lane_to_carla_keys.setdefault(str(sumo_lane), set()).add(
                    str(carla_key)
                )
        self._carla_waypoints_by_key = None
        self._client_calls = []
        self.count = 0
        self.metsr_viz_port = metsr_viz_port
        self.metsr_render_freq = _render_step_interval(timestep, metsr_render_freq)
        self.stream_url = f"ws://127.0.0.1:{self.metsr_viz_port}"
        self.metsr_lane_cache = {}

        # CoSim related params
        self.bubble_size = bubble_size
        self.bubble_roads = []
        self.workspace = scene.workspace
        self.carla_control_roads = {}
        # Physical roads and server-returned connector internal edges which are
        # currently controlled by CARLA. Canonical connector IDs remain opaque
        # METS-R tokens and are not physical map segments.
        self.carla_control_segments = set()
        self.metsr_internal_edge_to_connector = {}
        self.metsr_connector_records = {}
        self.metsr_road_connectors = {}
        self.active_bubble_metsr_roads = set()
        self.bubble_spawn_queue = set({})
        self.admitted_queue_vehicles = {}
        self.frozen_scenic_roads = []
        self.xml_to_xodr_intersections = xml_to_xodr_intersections
        self.metsr_actors = []
        self.carla_actors = []
        # Native METS-R vehicles represented as static CARLA obstacles. These
        # actors never participate in Scenic behaviors or ownership handoffs.
        # Private and public vehicle IDs occupy separate namespaces.
        self.boundary_actors = {}
        # Last physical segment selected from each CARLA-owned actor's pose.
        # Values are ordinary controlled road IDs or opaque canonical connector
        # IDs returned by METS-R; connector IDs are never logical route roads.
        self._carla_authoritative_segments = {}
        self.metsr_road_lines = []
        self.spawn_points = self.carla_world.get_map().get_spawn_points()
        self.completed_route = {}
        self.road_entry_holds = {}
        self.road_entry_hold_diagnostics = {}
        self._pending_pcla_pose_resets = set()
        self.pending_route_refreshes = {}
        self.pending_lane_reconciliation_verifications = {}
        self.same_road_departure_lane_verifications = {}
        self._carla_destination_plans = {}
        self.metsr_road_cache = {}
        self.metsr_connector_path_cache = {}
        self.metsr_connector_centerline_cache = {}
        self.road_pop_density = {}
        self.grp = None
        self.run_name = run_name

        # For tracking / data collection
        self.bubble_sizes = []
        self.total_active_vehicles = []


        super().__init__(scene, timestep=timestep, **kwargs)


    def setup(self) -> None:
        """
        Docstring for setup

        Setup the simulation instance
        """
        # Updated version takes no arguements
        self.metsr_client.reset()
        # Start the visualization once
        verbosePrint(f"Initializing METS-R visualization server")

        self.valid_metsr_roads = self.metsr_client.query_road()["roadIds"]

        self.network_helper = network_cache(self.workspace,
                                            self.scenic_to_metsr_map,
                                            self.valid_metsr_roads)


        self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2.0)

        weather = self.scene.params.get("weather")
        if weather is not None:
            if isinstance(weather, str):
                self.carla_world.set_weather(getattr(carla.WeatherParameters, weather))
            elif isinstance(weather, dict):
                self.carla_world.set_weather(carla.WeatherParameters(**weather))

        # Setup HUD
        # self.render=False
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
            print(f"starting recording")
            if not os.path.exists(self.record):
                os.mkdir(self.record)
            name = "{}/scenario{}.log".format(self.record, self.scenario_number)
            # Carla is looking for an absolute path, so convert it if necessary.
            name = os.path.abspath(name)
            self.carla_client.start_recorder(name)

        # Create objects.
        super().setup()
        self._synchronize_boundary_vehicles()

        #TODO create a dict mapping for maps and IDS
        # Build METSR Visualization url
        self.map_id = 12

        world_map_name = self.map.name
        print(f"World map: {world_map_name}")
        # map_keys = {
        #     "Town01": 4,
        #     "Town02": 5,
        #     "Town03": 6,
        #     "Town04": 7,
        #     "Town05": 8,
        #     "Town06": 9
        # }

        if self.render:
            print(f"Creating METS-R visualization server")
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

        # Set up camera manager and collision sensor for ego
        if self.render:
            camIndex = 0
            camPosIndex = 0
            egoActor = self.objects[0].carlaActor
            self.cameraManager = visuals.CameraManager(self.carla_world, egoActor, self.hud)
            self.cameraManager._transform_index = camPosIndex
            self.cameraManager.set_sensor(camIndex)
            self.cameraManager.set_transform(self.camTransform)

        self.carla_world.tick()  ## allowing manualgearshift to take effect

        # The core reads every object's properties immediately after setup,
        # including native NPCs which have not reached the first step yet.
        self.obj_data_cache = self._collect_metsr_vehicle_data(
            self.metsr_actors + self.carla_actors
        )

        for obj in self.scene.objects:
            if obj.carla_actor_flag:
                if obj.speed is not None and obj.speed != 0:
                    raise RuntimeError(
                        f"object {obj} cannot have a nonzero initial speed "
                        "(this is not yet possible in CARLA)"
                    )

    def createObjectInMetsr(self, obj: Object, route_generation_attempts=100, origin: str = None) -> None:
        """
        Docstring for createObjectInMetsr

        :param obj: Cosimulation car object
        :type obj: Scenic Object

        Creates vehicle inside the METSR simulator
        """
        lane = self.network_helper._nearest_lane(obj)
        if lane is None:
            raise SimulationCreationError(
                f"Unable to identify a Scenic origin lane for {obj}"
            )
        lane_key = f"{lane.road.id}_{lane.id}"
        mapped_lanes = list(
            self.network_helper.scenic_to_metsr_map_lanes.get(lane_key, ())
        )
        if not mapped_lanes:
            raise SimulationCreationError(
                f"No METS-R lane mapping exists for Scenic lane {lane_key}"
            )
        physical_origin = (
            str(origin)
            if origin is not None
            else self.identify_nearest_road(obj, mapped_lanes)
        )
        physical_lane_index = None
        physical_distance = math.inf
        physical_at_terminal = False
        if physical_origin is not None:
            (
                physical_lane_index,
                physical_distance,
                physical_at_terminal,
            ) = self._nearest_mapped_lane_projection(
                obj.position, mapped_lanes, physical_origin
            )

        (
            connector_segment,
            connector_path_id,
            connector_distance,
            connector_at_terminal,
            connector_path_record,
        ) = (None, None, math.inf, False, None)
        if origin is None:
            (
                connector_segment,
                connector_path_id,
                connector_distance,
                connector_at_terminal,
                connector_path_record,
            ) = self._nearest_mapped_connector_path_projection(
                obj.position, mapped_lanes
            )

        use_connector = (
            connector_segment is not None
            and math.isfinite(connector_distance)
            and connector_distance <= physical_distance
        )
        if use_connector:
            placement_segment = str(connector_segment)
            origin_lane_index = None
            origin_connector_path_id = connector_path_id
            source_lane_distance = connector_distance
            source_at_lane_terminal = connector_at_terminal
            route_origin = connector_path_record.get("sourceRoadId")
            route_query_origin = connector_path_record.get("targetRoadId")
            if route_origin is None or route_query_origin is None:
                raise SimulationCreationError(
                    f"Connector {placement_segment!r} path "
                    f"{origin_connector_path_id} omitted its physical source or "
                    f"target road"
                )
            route_origin = str(route_origin)
            route_query_origin = str(route_query_origin)
        elif (
            physical_origin is not None
            and physical_lane_index is not None
            and math.isfinite(physical_distance)
        ):
            placement_segment = str(physical_origin)
            origin_lane_index = physical_lane_index
            origin_connector_path_id = None
            source_lane_distance = physical_distance
            source_at_lane_terminal = physical_at_terminal
            route_origin = route_query_origin = placement_segment
        else:
            raise SimulationCreationError(
                f"No queryable METS-R road lane or connector path exists for "
                f"{obj} on Scenic lane {lane_key}; mapped lanes were "
                f"{mapped_lanes} and resolved road queries were "
                f"{[self._mapped_lane_query(item) for item in mapped_lanes]}"
            )

        origin_is_cosim = (
            placement_segment in self.carla_control_roads
            or self._is_controlled_connector_id(placement_segment)
        )
        # Native departures use only the mapped origin road and enter its queue;
        # their sampled coordinates are not passed to METS-R. Only co-simulation
        # initialization projects the pose and needs this distance bound.
        if origin_is_cosim:
            max_projection_error = max(3.0, float(obj.width) * 1.5)
            projection_error_limit = self._source_projection_error_limit(
                obj, lane_key, max_projection_error, source_at_lane_terminal
            )
            if source_lane_distance > projection_error_limit:
                selector = (
                    f"connectorPathId={origin_connector_path_id}"
                    if origin_connector_path_id is not None
                    else f"laneIndex={origin_lane_index}"
                )
                raise SimulationCreationError(
                    f"Scenic spawn for {obj} is {source_lane_distance:.2f} m from "
                    f"mapped METS-R segment {placement_segment}; refusing a remote "
                    f"projection; requested=({obj.position.x:.6f}, "
                    f"{obj.position.y:.6f}), Scenic lane={lane_key}, mapped "
                    f"lanes={mapped_lanes}, selected {selector}, "
                    f"terminal={source_at_lane_terminal}, "
                    f"limit={projection_error_limit:.2f} m"
                )

        route = None
        for _ in range(route_generation_attempts):
            # Resample the destination on every attempt and map its CARLA
            # waypoint directly, avoiding METS-R's ambiguous coordinate lookup.
            dest = random.choice(self.spawn_points)
            pos = utils.carlaToScenicPosition(dest.location)
            waypoint = self.map.get_waypoint(dest.location)
            if waypoint is None:
                continue
            destination_key = f"{waypoint.road_id}_{waypoint.lane_id}"
            destination_lanes = list(
                self.network_helper.scenic_to_metsr_map_lanes.get(
                    destination_key, ()
                )
            )
            if not destination_lanes:
                continue
            destination = self.identify_nearest_road(pos, destination_lanes)
            if destination is None:
                continue
            candidate = self.metsr_client.query_route_between_roads(
                route_query_origin, destination
            )["data"][0]
            if not isinstance(candidate, dict):
                continue
            candidate_roads = [
                str(road) for road in candidate.get("roadIds", ())
            ]
            if (
                not candidate_roads
                or candidate_roads[0] != route_query_origin
            ):
                continue
            if route_origin != route_query_origin:
                candidate_roads.insert(0, route_origin)
            if len(candidate_roads) > 1:
                route = candidate_roads
                break

        if route is None:
            raise SimulationCreationError(
                f"Failed to generate a multi-road trajectory for {obj} at "
                f"{(obj.position.x, obj.position.y)} from mapped segment "
                f"{placement_segment}"
            )

        obj.route = route
        vehID = self.getMetsrPrivateVehId(obj)
        call_kwargs = {
            "vehID": vehID,
            "origin": obj.route[0],
            "destination": obj.route[-1],
        }

        if not origin_is_cosim:
            _require_metsr_response(
                self.metsr_client.generate_trip_between_roads(**call_kwargs),
                "generateTripsByRoad",
            )
            print(f"Creating obj: {vehID}")
            # Native METS-R departures enter through the road's pending queue.
            # They have no current segment until a later METS-R tick, so Scenic
            # must neither teleport them nor query/update their on-road state.
            return

        print(f"Creating obj: {vehID}")

        if origin_is_cosim:
            # Scenic never uses Digital Twin teleport. CARLA-controlled roads
            # and connector paths instead receive Scenic's sampled pose through
            # the authoritative co-simulation initialization API.
            local_orientation = Orientation.fromEuler(
                obj.yaw, obj.pitch, obj.roll
            )
            global_orientation = obj.parentOrientation * local_orientation
            bearing, _, _ = global_orientation.eulerAngles
            bearing = -math.degrees(bearing) % 360
            placement_kwargs = {
                "vehID": vehID,
                "x": obj.position.x,
                "y": obj.position.y,
                "z": getattr(obj.position, "z", 0.0),
                "bearing": bearing,
                "speed": 0.0,
                "private_veh": True,
                "transform_coords": True,
                "destination_road_id": obj.route[-1],
            }
            try:
                placement = _require_metsr_response(
                    self.metsr_client.initialize_cosim_vehicle(
                        **placement_kwargs,
                        segment_id=placement_segment,
                        connector_path_id=origin_connector_path_id,
                    ),
                    "initializeCoSimVeh",
                )
            except METSRControlError as exc:
                record = exc.record if isinstance(exc.record, dict) else {}
                reason = record.get("errorCode")
                if (
                    reason != "NO_MAP_MATCH"
                    or origin_connector_path_id is not None
                ):
                    raise
                # A physical OpenDRIVE road can be adjacent to the closest
                # METS-R segment at a junction. Retry once without that road
                # hint; an exact connector path is never weakened this way.
                placement = _require_metsr_response(
                    self.metsr_client.initialize_cosim_vehicle(
                        **placement_kwargs, segment_id=None
                    ),
                    "initializeCoSimVeh",
                )

        placed_state = self.metsr_client.query_vehicle(
            vehID, private_veh=True, transform_coords=True
        )["data"][0]
        if "x" not in placed_state or "y" not in placed_state:
            raise SimulationCreationError(
                f"METS-R did not return an on-network placement for {obj}"
            )
        placement_record = placement["data"][0]
        placed_segment_id = placement_record.get("segmentId")
        if (
            placed_segment_id is None
            or str(placed_segment_id) != placement_segment
        ):
            raise SimulationCreationError(
                f"METS-R placed {obj} on segment {placed_segment_id!r}, but "
                f"Scenic selected segment {placement_segment!r}"
            )
        placed_lane_index = placement_record.get("laneIndex")
        placed_connector_path_id = placement_record.get("connectorPathId")
        if origin_connector_path_id is not None:
            if (
                placed_lane_index not in (None, METSR_CONNECTOR_NO_LANE)
                or placed_connector_path_id != origin_connector_path_id
            ):
                raise SimulationCreationError(
                    f"METS-R placed {obj} on connector path "
                    f"{placed_connector_path_id!r} of {placement_segment}, but "
                    f"Scenic lane {lane_key} maps to connectorPathId "
                    f"{origin_connector_path_id}"
                )
        elif (
            origin_lane_index >= 0
            and placed_lane_index != origin_lane_index
        ):
            raise SimulationCreationError(
                f"METS-R placed {obj} on lane {placed_lane_index} of road "
                f"{obj.route[0]}, but Scenic lane {lane_key} maps to METS-R "
                f"lane {origin_lane_index}"
            )
        backward_shift = abs(float(placement_record.get("backwardShift", 0.0)))
        displacement = math.hypot(
            placed_state["x"] - obj.position.x,
            placed_state["y"] - obj.position.y,
        )
        placement_error_limit = source_lane_distance + backward_shift + 0.25
        if displacement > placement_error_limit:
            selector = (
                f"connectorPathId={origin_connector_path_id}"
                if origin_connector_path_id is not None
                else f"laneIndex={origin_lane_index}"
            )
            raise SimulationCreationError(
                f"METS-R projected {obj} {displacement:.2f} m from its Scenic "
                f"spawn on segment {placement_segment} (reported backward shift "
                f"{backward_shift:.2f} m); requested="
                f"({obj.position.x:.6f}, {obj.position.y:.6f}), placed="
                f"({placed_state['x']:.6f}, {placed_state['y']:.6f}), "
                f"Scenic lane={lane_key}, mapped lanes={mapped_lanes}, "
                f"requested METS-R {selector}, placement={placement_record}, "
                f"allowed displacement={placement_error_limit:.2f} m"
            )

        # initializeCoSimVeh already plans METS-R's route to the supplied
        # destination for both road and connector spawns. The route query above
        # only selects a reachable destination; do not write that road sequence
        # back or impose it on CARLA/PCLA's driving decisions.

    def _degenerate_connector_spawn_orientation(self, state, position):
        """Recover a zero-length connector's direction from its mapped lane.

        SUMO continuity connectors can contain only repeated points. Native
        displacement into such a connector then reflects lateral lane placement,
        not a driving tangent. Keep ordinary native headings; only recover this
        undefined direction when the exact path topology and 3D pose agree.
        """
        connector_id = state.get("connectorId")
        path_id = self._normalized_connector_path_id(state.get("connectorPathId"))
        if path_id is None or not self._is_controlled_connector_id(connector_id):
            return None
        connector_id = str(connector_id)
        record = self.metsr_connector_records[connector_id]
        source, target = record.get("sourceRoadId"), record.get("targetRoadId")
        if source is None or target is None:
            return None
        mapped_endpoints = set()
        connections = getattr(self, "metsr_internal_lane_connections", {})
        for via_lane in state.get("connectorPathViaLaneIds", ()):
            for source_lane, target_lane in connections.get(str(via_lane), ()):
                if (
                    self._mapped_lane_query(source_lane)[0] == str(source)
                    and self._mapped_lane_query(target_lane)[0] == str(target)
                ):
                    mapped_endpoints.update((str(source_lane), str(target_lane)))
        if not mapped_endpoints:
            return None

        centerline = self._query_connector_path_centerline(connector_id, path_id)
        point = (position.x, position.y, position.z)
        if self._point_to_polyline_distance_3d(point, centerline) > CARLA_OWNED_PROJECTION_TOLERANCE:
            return None
        points = [tuple(map(float, vertex[:3])) for vertex in centerline]
        if any(math.dist(points[0], vertex) > 1e-6 for vertex in points[1:]):
            return None
        waypoint, error = self._projected_carla_driving_waypoint(
            carla.Location(position.x, -position.y, position.z)
        )
        if waypoint is None or error > CARLA_OWNED_PROJECTION_TOLERANCE:
            return None
        if not mapped_endpoints.intersection(self._mapped_lanes_for_carla_waypoint(waypoint)):
            return None
        return utils.carlaToScenicOrientation(waypoint.transform.rotation)

    def createObjectInCarla(
        self, obj: Object, update_velocity: bool = False, *, native_state=None
    ) -> bool:
        """
        Docstring for createObjectInCarla

        :param obj: Cosimulation car object
        :type obj: Scenic Object
        """
        try:
            blueprint = self.blueprintLib.find(obj.blueprint)
        except IndexError as e:
            found = False
            if obj.blueprint in oldBlueprintNames:
                for oldName in oldBlueprintNames[obj.blueprint]:
                    try:
                        blueprint = self.blueprintLib.find(oldName)
                        found = True
                        warnings.warn(
                            f"CARLA blueprint {obj.blueprint} not found; "
                            f"using older version {oldName}"
                        )
                        obj.blueprint = oldName
                        break
                    except IndexError:
                        continue
            if not found:
                raise SimulationCreationError(
                    f"Unable to find blueprint {obj.blueprint}" f" for object {obj}"
                ) from e
        if obj.rolename is not None:
            blueprint.set_attribute("role_name", obj.rolename)

        # set walker as not invincible
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "False")
        # Queue admission can change the native pose during this step, before
        # Scenic refreshes obj's dynamic properties. Use the same current record
        # which establishes ownership for position, heading, and initial velocity.
        position = obj.position
        orientation = obj.orientation
        native_velocity = None
        if native_state is not None:
            try:
                x, y = float(native_state["x"]), float(native_state["y"])
                z = float(native_state.get("z") or 0.0)
                bearing = float(native_state["bearing"])
                speed = float(native_state["speed"])
                if not all(math.isfinite(value) for value in (x, y, z, bearing, speed)):
                    raise ValueError("non-finite pose or speed")
                if speed < 0:
                    raise ValueError("negative speed")
            except (KeyError, TypeError, ValueError) as error:
                raise SimulationCreationError(
                    f"METS-R admission has no valid current pose for {obj}"
                ) from error
            position = Vector(x, y, z)
            heading = math.radians(-bearing)
            orientation = self._degenerate_connector_spawn_orientation(
                native_state, position
            )
            if orientation is None:
                orientation = Orientation.fromEuler(heading, 0, 0)
            native_velocity = Vector(0, speed, 0).rotatedBy(orientation)

        loc = utils.scenicToCarlaLocation(
            position,
            world=self.carla_world,
            blueprint=obj.blueprint,
            snapToGround=obj.snapToGround
        )
        rot = utils.scenicToCarlaRotation(orientation)
        transform = carla.Transform(loc, rot)
        if blueprint.has_attribute("color") and obj.color is not None:
            c = obj.color
            c_str = f"{int(c.r*255)},{int(c.g*255)},{int(c.b*255)}"
            blueprint.set_attribute("color", c_str)
        carlaActor = None
        spawn_error = None
        try:
            carlaActor = self.carla_world.try_spawn_actor(blueprint, transform)
        except Exception as exc:
            spawn_error = exc

        if carlaActor is None:
            self._clear_carla_authoritative_segment(obj)
            first_failure = obj not in self.bubble_spawn_queue
            self.bubble_spawn_queue.add(obj)
            if first_failure:
                if spawn_error is not None:
                    print(f"Exception while spawning {obj.name}: {spawn_error}")
                print(
                    f"CARLA rejected actor {obj.name} at {transform.location}; "
                    f"Scenic position {obj.position}, rotation {rot}. "
                    "The actor will remain queued and retry."
                )
                threshold = 1.2 * max(obj.length, obj.width)
                is_close, danger_veh = _utils.within_threshold_to(
                    obj, self.carla_actors, threshold=threshold, verbose=True
                )
                if danger_veh is not None:
                    veh_data = self._collect_metsr_vehicle_data([obj, danger_veh])
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

        self.bubble_spawn_queue.discard(obj)

        obj.carlaActor = carlaActor
        carlaActor.set_simulate_physics(obj.physics)

        if isinstance(carlaActor, carla.Vehicle):
            # TODO should get dimensions at compile time, not simulation time
            extent = carlaActor.bounding_box.extent
            ex, ey, ez = extent.x, extent.y, extent.z
            # Ensure each extent is positive to work around CARLA issue #5841
            obj.width = ey * 2 if ey > 0 else obj.width
            obj.length = ex * 2 if ex > 0 else obj.length
            obj.height = ez * 2 if ez > 0 else obj.height
            carlaActor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))


        elif isinstance(carlaActor, carla.Walker):
            carlaActor.apply_control(carla.WalkerControl())
            # spawn walker controller
            controller_bp = self.blueprintLib.find("controller.ai.walker")
            controller = self.carla_world.try_spawn_actor(
                controller_bp, carla.Transform(), carlaActor
            )
            if controller is None:
                self._clear_carla_authoritative_segment(obj)
                raise SimulationCreationError(
                    f"Unable to spawn carla controller for object {obj}"
                )
            obj.carlaController = controller

        obj.spawn_guard = 2
        obj.carla_actor_flag = True

        scenic_velocity = Vector(0, 0, 0)
        if update_velocity:
            scenic_velocity = native_velocity if native_velocity is not None else obj.velocity
            velocity = carla.Vector3D(
                scenic_velocity.x, -scenic_velocity.y, scenic_velocity.z
            )
            carlaActor.set_target_velocity(velocity)

        # CARLA's actor proxy can report a zero transform until its first world
        # snapshot. Preserve only this exact admission seed for that interval;
        # later native queries must not supply a CARLA owner's motion state.
        global_orientation = utils.carlaToScenicOrientation(rot)
        yaw, pitch, roll = obj.parentOrientation.localAnglesFor(global_orientation)
        if not hasattr(self, "_carla_spawn_property_seeds"):
            self._carla_spawn_property_seeds = {}
        self._carla_spawn_property_seeds[obj] = dict(
            position=utils.carlaToScenicPosition(loc),
            velocity=scenic_velocity,
            speed=math.hypot(*scenic_velocity),
            angularSpeed=0,
            angularVelocity=Vector(0, 0, 0),
            yaw=yaw, pitch=pitch, roll=roll,
            elevation=utils.carlaToScenicElevation(loc),
        )

        return True



    def createObjectInSimulator(self, obj: Object) -> None:
        """
        Docstring for createObjectInSimulator

        Spawn the object in the appropriate simulator
            (i) Ego is spawned in both simulators
            (ii)Ticks metsr to allow the vehicle to enter the road if the simulation has not started

        """
        assert obj.origin,      "All objects must have an origin"
        assert obj.destination, "All objects must have an destination"

        assert hasattr(obj, "carla_actor_flag"), "All objects must have attribute: carla_actor_flag"
        if obj == self.objects[0]: # Special handling for ego
            self.ego = obj
            self.spawn_ego(obj)
            self.carla_actors.append(obj)
        else:
            self.createObjectInMetsr(obj)
            obj.finished_route = False # Track route completion for autopilot
            if self.count == 0:
                self.metsr_client.tick() # allow obj to enter road if possible
            obj.carla_actor_flag = False
            obj.spawn_guard = 0
        # Track autopilot behaviors TODO redundant?
        self.metsr_actors.append(obj)
        obj.active_autopilot = False
        obj.autopilot_action = False
        obj.trip_start = 0

    def spawn_ego(self,obj: Object) -> None:
        """
        docstring for spawn_ego

        :param ego: Simulation ego object
        :type ego: EgoCar

        Special handling for spawning the Ego vehicle
            (1) First spawn ego in METSR on the appropriate lane (set by Scenic)
            (2) Teleport ego to the precise spawn location in MESTR
            (3) Collect exact spawn location to define bubbble region
            (4) Freeze CoSimulated regions inside METSR
            (5) Spawn ego inside CARLA
        """
        obj.bubble = CircularRegion(center=[obj.position.x,
                                            obj.position.y],
                                            radius=self.bubble_size)

        bubble_roads = self._get_bubble_roads()
        new_roads, _ = self.classify_bubble_roads(bubble_roads)
        self.freeze_roads(new_roads) # Freeze lanes according to Ego Spawn
        self.metsr_client.tick()

        self.createObjectInMetsr(obj) # Set the METSR vehicle origin to match ego spawn
        self.metsr_client.tick() # Allow the vehicle to spawn

        spawn_success = self.createObjectInCarla(obj) # spawn ego in updated location and update orientation
        assert spawn_success, f"Invalid spawn selection point at : {obj.position}"

    def getCarlaProperties(self, obj : Object, properties : dict) -> dict[str, float | Vector | int]:
            """
            Docstring for getCarlaProperties

            :param obj: Cosimulation car object
            :type obj: Scenic Object

            return objects properties from the Carla simulator
            """
            # Extract Carla properties
            carlaActor = obj.carlaActor
            currTransform = carlaActor.get_transform()
            currLoc = currTransform.location
            currRot = currTransform.rotation
            currVel = carlaActor.get_velocity()
            currAngVel = carlaActor.get_angular_velocity()

            # Prepare Scenic object properties
            position = utils.carlaToScenicPosition(currLoc)
            velocity = utils.carlaToScenicPosition(currVel)
            speed = math.hypot(*velocity)
            angularSpeed = utils.carlaToScenicAngularSpeed(currAngVel)
            angularVelocity = utils.carlaToScenicAngularVel(currAngVel)
            globalOrientation = utils.carlaToScenicOrientation(currRot)
            yaw, pitch, roll = obj.parentOrientation.localAnglesFor(globalOrientation)
            elevation = utils.carlaToScenicElevation(currLoc)

            values = dict(
                position=position,
                velocity=velocity,
                speed=speed,
                angularSpeed=angularSpeed,
                angularVelocity=angularVelocity,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                elevation=elevation,
            )
            return values


    def getMetsrProperties(self, obj: object, properties : dict) -> dict[str, float | Vector | int]:
        """
        Docstring for getMetsrProperties

        :param obj: Cosimulation car object
        :type obj: Scenic Object

        return objects properties from the METSR simulator
        """
        raw_data = self.obj_data_cache[obj]
        position = Vector(raw_data["x"], raw_data["y"], raw_data["z"] if raw_data["z"] is not None else 0)
        speed = raw_data["speed"]
        bearing = math.radians(-raw_data["bearing"])
        globalOrientation = Orientation.fromEuler(bearing,0,0)
        yaw, pitch, roll = obj.parentOrientation.localAnglesFor(globalOrientation)
        # Velocity is global; yaw above is relative to parentOrientation.
        # Using local yaw here launches admitted CARLA actors sideways.
        velocity = Vector(0, speed, 0).rotatedBy(bearing)
        angularSpeed = 0
        angularVelocity = Vector(0,0,0)

        values = dict(
            position=position,
            velocity=velocity,
            speed=speed,
            angularSpeed=angularSpeed,
            angularVelocity=angularVelocity,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            elevation=position.z
        )
        return values

    def getProperties(self, obj : Object, properties : dict)-> dict[str, float | Vector | int]:
        """
        Docstring for getProperties

        :param obj: Cosimulation car object
        :type obj: Scenic Object

        return objects properties for any CoSim object
        """
        assert hasattr(obj, "carla_actor_flag"), f"Object is not assigned properly to a simulator instance"
        if obj.carla_actor_flag:
            seeds = getattr(self, "_carla_spawn_property_seeds", {})
            if obj in seeds:
                # Setup also advances CARLA directly. Its cached world snapshot
                # identifies publication without assuming a nonzero actor pose.
                snapshot = self.carla_world.get_snapshot()
                if snapshot.find(obj.carlaActor.id) is None:
                    return dict(seeds[obj])
                seeds.pop(obj)
            return self.getCarlaProperties(obj, properties)
        return self.getMetsrProperties(obj, properties)

    def getMetsrPrivateVehId(self, obj: Object) -> int:
        """
        Docstring for getMetsrPrivateVehId

        :param obj: Cosimulation car object
        :type obj: Scenic Object

        Return unique vehicle idea
            Generates a new ID if none exists for vehicle
        """
        if obj not in self.pv_id_map:
            self.pv_id_map[obj] = self.next_pv_id
            self.next_pv_id += 1
        return self.pv_id_map[obj]

    def tick_carla(self) -> None:
        """
        Docstring for tick_carla

        Tick Carla client for a single step
        """
        for _ in range(self.sim_ticks_per_carla):
            # Behaviors run before ``step`` and may have queued a fresh manual
            # control after synchronization placed an actor into a safety hold.
            # Reassert the hold immediately before every physics tick so neither
            # PCLA nor Traffic Manager can move/rotate the verification pose.
            for obj in tuple(getattr(self, "road_entry_holds", ())):
                self._enforce_road_entry_hold(obj)
            self.carla_world.tick()
            # All successful spawns are now present in CARLA's published state;
            # spawn_guard does not choose the source of an owner's properties.
            getattr(self, "_carla_spawn_property_seeds", {}).clear()


    def _boundary_vehicle_blueprint(self, key, state, scenic_object):
        """Use the Scenic model when known, or a CARLA car/bus for native fleets."""
        if scenic_object is not None:
            model = scenic_object.blueprint
            models = (model, *oldBlueprintNames.get(model, ()))
        else:
            models = busModels if state["vehicleClass"] == 2 else carModels
        for model in models:
            try:
                blueprint = self.blueprintLib.find(model)
            except IndexError:
                continue
            blueprint.set_attribute(
                "role_name", f"metsr_boundary_{'private' if key[0] else 'public'}_{key[1]}"
            )
            if scenic_object is not None and blueprint.has_attribute("color"):
                color = getattr(scenic_object, "color", None)
                if color is not None:
                    blueprint.set_attribute(
                        "color", f"{int(color.r * 255)},{int(color.g * 255)},{int(color.b * 255)}"
                    )
            return blueprint
        raise SimulationCreationError(
            f"No CARLA vehicle blueprint is available for METS-R boundary vehicle {key}"
        )

    def _boundary_vehicle_transform(self, state, scenic_object, actor=None):
        """Convert a current METS-R pose, keeping a static vehicle on the ground."""
        try:
            x, y = float(state["x"]), float(state["y"])
            z = float(state.get("z") or 0.0)
            bearing = float(state["bearing"])
            if not all(math.isfinite(value) for value in (x, y, z, bearing)):
                raise ValueError("non-finite pose")
        except (KeyError, TypeError, ValueError) as error:
            raise METSRControlError(
                "METS-R boundary vehicle has no valid current pose", record=state
            ) from error
        location = carla.Location(x, -y, z)
        if getattr(scenic_object, "snapToGround", True):
            waypoint = self.map.get_waypoint(
                location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if waypoint is None:
                raise SimulationCreationError(
                    f"No CARLA road elevation for boundary vehicle {state['vehicleId']}"
                )
            # Dynamic actors settle under gravity; a physics-disabled obstacle
            # must instead be placed with its bounding-box bottom on the road.
            height_offset = 0.5
            if actor is not None and actor.bounding_box.extent.z > 0:
                height_offset = (
                    actor.bounding_box.extent.z - actor.bounding_box.location.z + 0.05
                )
            location.z = waypoint.transform.location.z + height_offset
        return carla.Transform(location, carla.Rotation(yaw=bearing - 90.0))

    def _remove_boundary_actor(self, key):
        actor = self.boundary_actors[key]
        if self._carla_actor_is_registered(actor) and actor.destroy() is False:
            raise SimulationCreationError(
                f"CARLA could not remove METS-R boundary vehicle {key}"
            )
        del self.boundary_actors[key]

    def _synchronize_boundary_vehicles(self):
        """Mirror native exit occupancy without acquiring METS-R vehicle control.

        Boundary membership is selected by METS-R. Its coordinateTrail contains
        upcoming route points, so a separate batched vehicle query supplies the
        current pose. Obstacles stay fixed throughout all CARLA substeps.
        """
        if not hasattr(self, "boundary_actors"):
            self.boundary_actors = {}
        response = _require_all_success_control_response(
            self.metsr_client.query_boundary_vehicle(), "boundaryVehicle"
        )
        if not isinstance(response.get("data"), (list, tuple)):
            raise METSRControlError("boundaryVehicle omitted data", response=response)
        scenic_objects = {
            (True, str(vehicle_id)): obj
            for obj, vehicle_id in self.pv_id_map.items()
        }
        controlled = {
            key for key, obj in scenic_objects.items()
            if getattr(obj, "carlaActor", None) is not None
        }
        selected = {}
        for record in response["data"]:
            if (
                not isinstance(record, dict)
                or record.get("vehicleId") is None
                or not isinstance(record.get("isPrivate"), bool)
            ):
                raise METSRControlError(
                    "boundaryVehicle returned an invalid identity", record=record
                )
            key = (record["isPrivate"], str(record["vehicleId"]))
            if key in controlled or self._vehicle_record_in_cosim_region(record):
                continue
            selected[key] = record

        states = {}
        if selected:
            response = _require_all_success_control_response(
                self.metsr_client.query_vehicle(
                    [record["vehicleId"] for record in selected.values()],
                    private_veh=[key[0] for key in selected],
                    transform_coords=True,
                ),
                "vehicle",
            )
            records = response.get("data", ())
            if len(records) != len(selected):
                raise METSRControlError(
                    "METS-R omitted boundary vehicle poses", response=response
                )
            # query_vehicle returns records in request order. Check both the
            # visible ID and vehicle class so public/private ID collisions cannot
            # attach one fleet's pose to another fleet's obstacle.
            for key, state in zip(selected, records):
                if (
                    not isinstance(state, dict)
                    or str(state.get("vehicleId")) != key[1]
                    or state.get("vehicleClass") not in (0, 1, 2, 3)
                    or (state["vehicleClass"] in (0, 3)) != key[0]
                ):
                    raise METSRControlError(
                        f"METS-R returned the wrong boundary vehicle pose for {key}",
                        response=response, record=state,
                    )
                obj = scenic_objects.get(key)
                # Validate every pose before changing any obstacle membership.
                self._boundary_vehicle_transform(state, obj)
                states[key] = state

        for key in tuple(self.boundary_actors):
            if key not in states:
                self._remove_boundary_actor(key)
        for key, state in states.items():
            obj = scenic_objects.get(key)
            actor = self.boundary_actors.get(key)
            if actor is not None and not self._carla_actor_is_alive(actor):
                self._remove_boundary_actor(key)
                actor = None
            if actor is None:
                blueprint = self._boundary_vehicle_blueprint(key, state, obj)
                transform = self._boundary_vehicle_transform(state, obj)
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
            actor.set_transform(self._boundary_vehicle_transform(state, obj, actor))

    def tick_metsr(self) -> None:
        """
        Docstring for tick_metsr

        Tick Metsr client for a single step
        """
        for _ in range(self.sim_ticks_per_metsr):
            self.metsr_client.tick()

    def step(self) -> None:
        """
        Docstring for step

        Step both simulators:
            (1): Update the high fidelity region based on the ego's new locatin
            (2): Spawn and destroy objects according to region changes
            (3): Tick both clients and synchronize states
            (4): Compute new bubble region
        """
        self.road_pop_density = {road: 0 for road in self.valid_metsr_roads}
        # (1): Update the high fidelity region based on the ego's new locatin
        bubble_roads = self._get_bubble_roads(CircularRegion(center=[self.ego.x, self.ego.y], radius=self.bubble_size))
        new_roads, old_roads = self.classify_bubble_roads(bubble_roads)
        # Acquire the new roads before advancing CARLA. Old roads remain owned
        # until after CARLA's current pose has been published and all resulting
        # handbacks and completions have been applied below.
        self.freeze_roads(new_roads)
        # Remove obsolete obstacles before promotion/queue admission, and make
        # native exit occupancy visible before any CARLA physics/sensor tick.
        self._synchronize_boundary_vehicles()
        self._service_metsr_entering_queues()
        intersections = self.get_bubble_intersections(bubble_roads=bubble_roads, bubble_region=self.ego.bubble)
        # (2): Spawn and destroy objects according to region changes
        bubble_road_ids = [road.id for road in bubble_roads]
        intersection_ids = [intersection.id for intersection in intersections]
        # Promote newly-controlled vehicles before CARLA advances, but defer
        # demotion until after CARLA's authoritative pose has been mirrored to
        # METS-R. This removes the stale-snapshot boundary heuristic.
        # METS-R has not advanced between promotion and CARLA publication, so
        # one batched snapshot can serve promotion and synchronization.
        state_objects = list(dict.fromkeys((*self.objects[1:], *self.carla_actors)))
        step_vehicle_data = self._collect_metsr_vehicle_data(state_objects)
        self.update_bubble_objects(
            bubble_road_ids,
            intersection_ids,
            vehicle_data=step_vehicle_data,
        )

        # (3): Tick both clients and synchronize states
        self._refresh_carla_destination_paths(step_vehicle_data)
        self.tick_carla()
        self.synchronize_clients(vehicle_data=step_vehicle_data)
        self._remove_completed_carla_objects()
        releasable_roads = self._releasable_old_roads(old_roads)
        if releasable_roads:
            self.release_roads(releasable_roads, allow_defer=True)
        self._report_road_entry_holds()
        self.tick_metsr()
        self.obj_data_cache = self._collect_metsr_vehicle_data(self.metsr_actors + self.carla_actors)

        if self.render and self.count % self.metsr_render_freq == 0:
            self.cameraManager.render(self.display)
            pygame.display.flip()
            try:
                self.metsr_client.render()
            except Exception as e:
                if self.count % 100 == 0:
                    print(f"Warning no registered connection to streaming client")

        self.bubble_sizes.append(len(self.carla_actors))
        self.total_active_vehicles.append(len(self.objects) - (len(self.frozen_vehicles) + len(self.bubble_spawn_queue)))
        if self.count % 200 == 0:
            print(f"Step: {self.count}. Total actors: {len(self.objects)}, bubble queue:{len(self.bubble_spawn_queue)} ")
            print(f"Total active vehicles: {self.total_active_vehicles[-1]}, frozen vehicles {len(self.frozen_vehicles)}")
            print(f"Total bubble actors: {len(self.carla_actors) + len(self.bubble_spawn_queue)}")
            print(f"Completed routes: {len(list(self.completed_route.keys()))}")

        if self.count % 50 == 0:
            if self.run_name is not None:
                out_file = self.run_name + "_veh_data.csv"

                data_dict_at_i = {
                    "active_vehicles": self.total_active_vehicles[-1],
                    "bubble_actors" : len(self.carla_actors),
                    "bubble_queue": len(self.bubble_spawn_queue),
                    "completed_routes": len(list(self.completed_route.keys())),
                    **self.road_pop_density
                }
                final_df = pd.DataFrame([data_dict_at_i])
                del data_dict_at_i

                if self.count == 0:
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    final_df.to_csv(out_file,
                                    mode="w",
                                    header=True)
                else:
                    final_df.to_csv(out_file,
                                    mode="a",
                                    header=False)

        self.count += 1
        # (4): Compute new bubble region and process behavior interrupts
        self.ego.bubble = CircularRegion(center=[self.objects[0].x, self.objects[0].y], radius=self.bubble_size)
        for road in bubble_roads:
            self.ego.bubble = self.ego.bubble.union(road)
        for intersection in intersections:
            self.ego.bubble = self.ego.bubble.union(intersection)


    def get_bubble_intersections(self, bubble_roads: list[Road], bubble_region: CircularRegion) -> list[Intersection]:
        """Include intersections attached to the selected road component.

        Every incident intersection is needed for METS-R ownership consistency.
        XY overlap alone is insufficient because it can select another level.
        """
        road_ids = {str(road.id) for road in bubble_roads}
        bubble_intersections = []
        for intersection in self.network_helper.network_intersections:
            attached_roads = list(intersection.roads)
            attached_roads.extend(
                maneuver.connectingLane.road
                for maneuver in intersection.maneuvers
                if maneuver.connectingLane is not None
            )
            if any(str(road.id) in road_ids for road in attached_roads):
                bubble_intersections.append(intersection)
        return bubble_intersections

    def initiate_autopilot(self, obj : Object) -> bool:
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
            obj.carlaActor.set_autopilot(True, self.tm.get_port())
            # Traffic Manager may choose lanes along its own destination path.
            self.tm.auto_lane_change(obj.carlaActor, True)
            obj.active_autopilot = True
            obj.autopilot_action = True
            success = True
        else:
            obj.override_autopilot = False
            success = True

        return success

    @staticmethod
    def _normalized_road_route(route):
        """Return a normalized route tuple without inventing route state."""
        if not isinstance(route, (list, tuple)):
            return ()
        return tuple(str(road_id) for road_id in route if road_id is not None)

    @classmethod
    def _route_predecessor(cls, road_id, route):
        """Return a uniquely provable immediate predecessor, if present."""
        if road_id is None:
            return None
        normalized = cls._normalized_road_route(route)
        target = str(road_id)
        indices = [
            index
            for index, candidate in enumerate(normalized)
            if candidate == target
        ]
        if len(indices) != 1 or indices[0] == 0:
            return None
        return normalized[indices[0] - 1]

    @classmethod
    def _route_successor(cls, road_id, route):
        """Return the uniquely anchored immediate successor, if provable."""
        if road_id is None:
            return None
        normalized = cls._normalized_road_route(route)
        if not normalized:
            return None
        current = str(road_id)
        if normalized[0] == current:
            return normalized[1] if len(normalized) > 1 else None
        indices = [
            index
            for index, candidate in enumerate(normalized)
            if candidate == current
        ]
        if len(indices) == 1:
            index = indices[0]
            return normalized[index + 1] if index + 1 < len(normalized) else None
        return None

    @classmethod
    def _route_tail_at_direct_target(cls, current_road_id, target_road_id, route):
        """Return the live route tail after one provable direct edge."""
        normalized = cls._normalized_road_route(route)
        if current_road_id is None or target_road_id is None:
            return ()
        current = str(current_road_id)
        target = str(target_road_id)
        if len(normalized) >= 2 and normalized[:2] == (current, target):
            return normalized[1:]
        indices = [
            index for index, candidate in enumerate(normalized) if candidate == current
        ]
        if len(indices) != 1:
            return ()
        index = indices[0]
        if index + 1 >= len(normalized) or normalized[index + 1] != target:
            return ()
        return normalized[index + 1 :]

    def _direct_lane_topology(
        self,
        current_road_id,
        current_lane_id,
        target_road_id,
        target_lane_id,
    ):
        """Return ``(road_pair_exists, exact_lane_pair_exists)``."""
        if current_road_id is None or target_road_id is None:
            return False, False
        pairs = tuple(
            getattr(self, "metsr_lane_connections", {}).get(
                (str(current_road_id), str(target_road_id)), ()
            )
        )
        if not pairs:
            return False, False
        if (
            current_lane_id is None
            or target_lane_id is None
            or isinstance(current_lane_id, bool)
            or isinstance(target_lane_id, bool)
        ):
            return True, False
        try:
            current_lane_index = int(current_lane_id)
            target_lane_index = int(target_lane_id)
        except (TypeError, ValueError):
            return True, False
        for source_lane, target_lane in pairs:
            source_road, source_index = self._mapped_lane_query(source_lane)
            target_road, target_index = self._mapped_lane_query(target_lane)
            if (
                source_road == str(current_road_id)
                and target_road == str(target_road_id)
                and source_index == current_lane_index
                and target_index == target_lane_index
            ):
                return True, True
        return True, False

    def _required_skipped_successor_lane(
        self,
        pending_road_id,
        pending_lane_id,
        observed_road_id,
        remaining_route,
    ):
        """Resolve the unique route-compatible lane after a skipped short road."""
        route = self._normalized_road_route(remaining_route)
        pending_road = str(pending_road_id)
        observed_road = str(observed_road_id)
        if len(route) < 2 or route[:2] != (pending_road, observed_road):
            raise SimulationCreationError(
                "Cannot reconcile a skipped pending target without an exact "
                f"live route prefix {(pending_road, observed_road)}; route={list(route)}"
            )

        pending_raw_lane = self._resolve_required_initial_lane(
            pending_road, pending_lane_id
        )
        arrivals = {
            str(target_lane)
            for source_lane, target_lane in self.metsr_lane_connections.get(
                (pending_road, observed_road), ()
            )
            if str(source_lane) == pending_raw_lane
            and self._mapped_lane_query(target_lane)[0] == observed_road
        }
        if len(route) > 2:
            onward_sources = {
                str(source_lane)
                for source_lane, _ in self.metsr_lane_connections.get(
                    (observed_road, route[2]), ()
                )
            }
            arrivals &= onward_sources

        candidates = []
        for raw_lane in sorted(arrivals):
            road, lane_index = self._mapped_lane_query(raw_lane)
            if road == observed_road and lane_index is not None:
                candidates.append((raw_lane, int(lane_index)))
        if len(candidates) != 1:
            raise SimulationCreationError(
                "Expected one route-compatible successor lane while reconciling "
                f"pending road/lane {pending_road}/{pending_lane_id} onto "
                f"{observed_road}; found={candidates}, route={list(route)}"
            )
        raw_lane, lane_index = candidates[0]
        return raw_lane, lane_index, route[1:]

    def _required_same_road_departure_lane(
        self, current_road_id, current_lane_id, remaining_route, target_start
    ):
        """Return one adjacent lane required to serve the live next road."""
        route = self._normalized_road_route(remaining_route)
        current_road = str(current_road_id)
        if len(route) < 2 or route[0] != current_road:
            return None

        current_raw_lane = self._resolve_required_initial_lane(
            current_road, current_lane_id
        )
        direct_pairs = self.metsr_lane_connections.get(
            (current_road, route[1]), ()
        )
        departure_lanes = {
            str(source_lane) for source_lane, _ in direct_pairs
        }
        if not departure_lanes:
            raise SimulationCreationError(
                f"No direct driving-lane topology serves live route edge "
                f"{current_road}->{route[1]}"
            )
        if current_raw_lane in departure_lanes:
            return None

        lane_chains = self._candidate_metsr_lane_chains(
            route, target_start, required_initial_lane=current_raw_lane
        )
        compatible_departures = {
            str(chain[0])
            for chain in lane_chains
            if chain and str(chain[0]) in departure_lanes
        }
        _, current_index = self._mapped_lane_query(current_raw_lane)
        candidates = []
        for raw_lane in compatible_departures:
            road_id, compact_lane = self._mapped_lane_query(raw_lane)
            if (
                compact_lane is None
                or str(road_id) != current_road
                or current_index is None
            ):
                continue
            if abs(int(compact_lane) - int(current_index)) == 1:
                candidates.append((raw_lane, int(compact_lane)))

        if len(candidates) != 1:
            raise SimulationCreationError(
                "Expected one adjacent route-compatible departure lane while "
                f"preparing {current_road}->{route[1]} from "
                f"{current_raw_lane}; found={sorted(candidates)}, "
                f"compatible={sorted(compatible_departures)}"
            )
        return candidates[0]

    @classmethod
    def _classify_road_observation(
        cls,
        current_road_id,
        observed_road_id,
        remaining_route,
        predecessor_road_id=None,
    ):
        """Classify a mapped CARLA road relative to one METS-R route state.

        A live METS-R remaining route is anchored at its current road. Repeated,
        absent, or otherwise ambiguous anchors fail closed; the only forward
        classifications are the exact successor and one explicitly represented
        skipped road.
        """
        if observed_road_id is None:
            return ROAD_OBSERVATION_UNKNOWN
        if current_road_id is None:
            return ROAD_OBSERVATION_UNSUPPORTED

        current = str(current_road_id)
        observed = str(observed_road_id)
        if observed == current:
            return ROAD_OBSERVATION_SAME
        route = cls._normalized_road_route(remaining_route)
        if not route:
            return ROAD_OBSERVATION_UNSUPPORTED

        indices = [
            index for index, candidate in enumerate(route) if candidate == current
        ]
        if route[0] == current:
            # A remaining route is anchored by its first element even if a
            # later loop revisits the same road.
            current_index = 0
        elif len(indices) == 1:
            current_index = indices[0]
        elif indices:
            return ROAD_OBSERVATION_UNSUPPORTED
        else:
            return ROAD_OBSERVATION_UNSUPPORTED

        if current_index + 1 < len(route) and observed == route[current_index + 1]:
            return ROAD_OBSERVATION_DIRECT_SUCCESSOR
        if current_index + 2 < len(route) and observed == route[current_index + 2]:
            return ROAD_OBSERVATION_ONE_SKIP
        if current_index > 0 and observed == route[current_index - 1]:
            return ROAD_OBSERVATION_PREDECESSOR
        if predecessor_road_id is not None and observed == str(predecessor_road_id):
            return ROAD_OBSERVATION_PREDECESSOR
        return ROAD_OBSERVATION_UNSUPPORTED

    def _observed_road_hint(self, obj, transition_state, observed_road_id):
        """Return a server-owned connector hint for a physical observation."""
        del obj
        connector_hint = self._canonical_connector_hint(
            transition_state, observed_road_id
        )
        if connector_hint is not None:
            return connector_hint
        if self._is_controlled_internal_segment(observed_road_id):
            return self.metsr_internal_edge_to_connector[str(observed_road_id)]
        return None

    def _is_controlled_internal_segment(self, segment_id):
        """Return whether METS-R assigned this physical edge to a connector."""
        if segment_id is None:
            return False
        self._ensure_connector_ownership_state()
        segment_id = str(segment_id)
        return (
            segment_id in self.carla_control_segments
            and segment_id in self.metsr_internal_edge_to_connector
        )

    def _vehicle_record_in_cosim_region(self, record):
        """Test physical COSIM ownership without reducing connectors to roads."""
        if not isinstance(record, dict):
            return False
        self._ensure_connector_ownership_state()

        connector_ids = {
            str(value)
            for value in (record.get("connectorId"), record.get("segmentId"))
            if value is not None
        }
        if connector_ids & set(self.metsr_connector_records):
            return True

        physical_segments = {
            str(value)
            for value in (
                record.get("segmentId"),
                record.get("observedSegmentId"),
                record.get("physicalSegmentId"),
            )
            if value is not None
        }
        physical_segments.update(
            str(edge_id)
            for edge_id in (record.get("internalEdgeIds") or ())
            if edge_id is not None
        )
        if physical_segments & self.carla_control_segments:
            return True

        logical_road = _metsr_vehicle_road(record)
        return (
            logical_road is not None
            and str(logical_road) in self.carla_control_roads
        )

    def _canonical_connector_hint(self, cosim_record, observed_segment_id):
        """Resolve an observation to METS-R's canonical connector ID.

        Scenic never constructs or parses connector IDs. It accepts only a
        canonical ID returned by setCoSimRoad and proves that the live record
        and observed physical segment refer to that controlled connector
        before sending the ID back in teleportCoSimVeh.
        """
        if not isinstance(cosim_record, dict):
            return None

        connector_id = cosim_record.get("connectorId")
        if connector_id is None and (
            str(cosim_record.get("segmentType", "")).lower() == "connector"
        ):
            connector_id = cosim_record.get("segmentId")
        if connector_id is None:
            return None
        connector_id = str(connector_id)

        self._ensure_connector_ownership_state()
        ownership_record = self.metsr_connector_records.get(connector_id)
        if not isinstance(ownership_record, dict):
            return None
        referenced_connectors = {
            owned_connector
            for connector_ids in self.metsr_road_connectors.values()
            for owned_connector in connector_ids
        }
        if connector_id not in referenced_connectors:
            return None

        for field in ("sourceRoadId", "targetRoadId"):
            live_value = cosim_record.get(field)
            owned_value = ownership_record.get(field)
            if (
                live_value is not None
                and owned_value is not None
                and str(live_value) != str(owned_value)
            ):
                return None

        observed = (
            None if observed_segment_id is None else str(observed_segment_id)
        )
        if observed == connector_id:
            return connector_id
        if (
            observed is not None
            and self.metsr_internal_edge_to_connector.get(observed) == connector_id
            and observed in self.carla_control_segments
        ):
            return connector_id

        source_road_id = (
            cosim_record.get("sourceRoadId")
            or ownership_record.get("sourceRoadId")
        )
        target_road_id = (
            cosim_record.get("targetRoadId")
            or ownership_record.get("targetRoadId")
        )
        if (
            source_road_id is not None
            and observed == str(source_road_id)
            and target_road_id is not None
            and str(target_road_id) in self.carla_control_roads
        ):
            return connector_id
        return None

    def _ensure_carla_authority_state(self):
        """Initialize CARLA-owned segment continuity for lightweight fixtures."""
        if not hasattr(self, "_carla_authoritative_segments"):
            self._carla_authoritative_segments = {}

    def _owned_connector_ids(self):
        """Return opaque connector IDs currently referenced by road ownership."""
        self._ensure_connector_ownership_state()
        referenced = {
            str(connector_id)
            for connector_ids in self.metsr_road_connectors.values()
            for connector_id in connector_ids
        }
        return referenced & set(map(str, self.metsr_connector_records))

    def _is_controlled_connector_id(self, segment_id):
        if segment_id is None:
            return False
        return str(segment_id) in self._owned_connector_ids()

    def _connector_record_for_transition(
        self, source_road_id, target_road_id, cosim_record=None
    ):
        """Resolve source->target through server-returned opaque connector data."""
        if source_road_id is None or target_road_id is None:
            return None, None
        source_road_id = str(source_road_id)
        target_road_id = str(target_road_id)
        owned = self._owned_connector_ids()

        preferred = []
        if isinstance(cosim_record, dict):
            for value in (
                cosim_record.get("connectorId"),
                cosim_record.get("segmentId"),
            ):
                if value is not None and str(value) in owned:
                    preferred.append(str(value))
        preferred.extend(sorted(owned - set(preferred)))

        for connector_id in preferred:
            record = self.metsr_connector_records.get(connector_id)
            if not isinstance(record, dict):
                continue
            source = record.get("sourceRoadId")
            target = record.get("targetRoadId")
            if (
                source is not None
                and target is not None
                and str(source) == source_road_id
                and str(target) == target_road_id
            ):
                return connector_id, record
        return None, None

    def _initialize_carla_authoritative_segment(
        self, obj, vehicle_state=None, cosim_record=None
    ):
        """Seed continuity from admission metadata without parsing connector IDs."""
        self._ensure_carla_authority_state()
        if obj in self._carla_authoritative_segments:
            return self._carla_authoritative_segments[obj]

        for record in (cosim_record, vehicle_state):
            if not isinstance(record, dict):
                continue
            segment = record.get("segmentId")
            if segment is None:
                continue
            segment = str(segment)
            if segment in self.carla_control_roads:
                self._carla_authoritative_segments[obj] = segment
                return segment
            if self._is_controlled_internal_segment(segment):
                connector_id = self.metsr_internal_edge_to_connector[segment]
                self._carla_authoritative_segments[obj] = connector_id
                return connector_id
            if self._is_controlled_connector_id(segment):
                self._carla_authoritative_segments[obj] = segment
                return segment
        return None

    def _clear_carla_authoritative_segment(self, obj):
        self._ensure_carla_authority_state()
        self._carla_authoritative_segments.pop(obj, None)
        plan = getattr(self, "_carla_destination_plans", {}).pop(obj, None)
        if plan is not None and getattr(obj, "trajectory", None) is plan[1]:
            obj.trajectory = None

    def _releasable_old_roads(self, old_roads):
        """Keep native segments controlled until their CARLA occupants leave."""
        occupied = set()
        segments = getattr(self, "_carla_authoritative_segments", {})
        for obj in self.carla_actors:
            segment = segments.get(obj)
            if segment is None:
                # No accepted observation yet: releasing a road could resume
                # native motion while this actor is still controlled by CARLA.
                return []
            occupied.add(str(segment))
        held = set(occupied)
        for road, connectors in getattr(self, "metsr_road_connectors", {}).items():
            if occupied.intersection(map(str, connectors)):
                held.add(str(road))
        for admission in getattr(self, "admitted_queue_vehicles", {}).values():
            held.add(str(admission["roadId"]))
        return [str(road) for road in old_roads if str(road) not in held]

    def _native_handoff_target(self, obj, location):
        """Resolve an outgoing NPC's native placement; ordinary poses need no hint.

        CARLA's current waypoint identifies a possible boundary crossing. Only
        an ordinary road outside the controlled region triggers handback;
        junctions and ambiguous/unmapped poses use ordinary native association.
        """
        if obj is getattr(self, "ego", None):
            return None
        waypoint, distance = self._projected_carla_driving_waypoint(location)
        if (waypoint is None or waypoint.is_junction
                or distance > CARLA_OWNED_PROJECTION_TOLERANCE):
            return None
        mapped_lanes = self._mapped_lanes_for_carla_waypoint(waypoint)
        candidates = []
        for mapped_lane in mapped_lanes:
            road, index = self._mapped_lane_query(mapped_lane)
            if (road is not None and not str(road).startswith(":")
                    and index is not None and index >= 0):
                candidates.append((str(road), index, mapped_lane))
        if not candidates or all(road in self.carla_control_roads
                                 for road, _, _ in candidates):
            return None
        if len(candidates) == 1:
            road, lane, _ = candidates[0]
        else:
            position = (location.x, -location.y, location.z)
            road = self.identify_nearest_road(
                position, [entry[2] for entry in candidates]
            )
            if road is None or str(road) in self.carla_control_roads:
                return None
            road = str(road)
            lane, _ = self._nearest_mapped_lane(position, mapped_lanes, road)
            if lane is None:
                return None
        # CARLA and SUMO place the start of a junction at slightly different
        # positions. A newly admitted connector may still project onto its
        # upstream CARLA road; handing it back there causes immediate re-entry
        # and repeated promotion. This is entry, not an outgoing boundary.
        current = getattr(self, "_carla_authoritative_segments", {}).get(obj)
        connector = getattr(self, "metsr_connector_records", {}).get(current, {})
        if str(connector.get("sourceRoadId")) == road:
            return None
        return road, lane, None

    def _classify_physical_road_observation(
        self,
        current_road_id,
        observed_segment_id,
        remaining_route,
        predecessor_road_id=None,
    ):
        """Classify logical roads while preserving owned internal segments."""
        if self._is_controlled_internal_segment(observed_segment_id):
            return ROAD_OBSERVATION_UNKNOWN
        return self._classify_road_observation(
            current_road_id,
            observed_segment_id,
            remaining_route,
            predecessor_road_id=predecessor_road_id,
        )

    @classmethod
    def _is_forward_road_observation(
        cls, current_road_id, observed_road_id, remaining_route
    ):
        relation = cls._classify_road_observation(
            current_road_id, observed_road_id, remaining_route
        )
        return relation in (
            ROAD_OBSERVATION_DIRECT_SUCCESSOR,
            ROAD_OBSERVATION_ONE_SKIP,
        )

    def _fail_road_synchronization(
        self,
        obj,
        veh_id,
        veh_data,
        observed_road_id,
        observed_lane_id,
        remaining_route,
        relation,
    ):
        """Stop a divergent CARLA actor and raise a diagnostic error."""
        current = _metsr_vehicle_road(veh_data)
        self._hold_for_road_entry(
            obj,
            source=current,
            target=observed_road_id,
            reason=f"ROAD_SYNC_{str(relation).upper()}",
        )
        route = list(self._normalized_road_route(remaining_route))
        raise SimulationCreationError(
            "CARLA/METS-R road synchronization failed closed: "
            f"object={getattr(obj, 'name', obj)}, vehicleID={veh_id}, "
            f"step={getattr(self, 'count', None)}, "
            f"METS road={current}, "
            f"CARLA observed road={observed_road_id}, "
            f"observed lane={observed_lane_id}, relation={relation}, "
            f"remaining route={route}"
        )

    @staticmethod
    def _carla_waypoint_distance(location, waypoint):
        """Measure an actor pose against its projected CARLA waypoint."""
        if waypoint is None:
            return math.inf
        waypoint_location = getattr(
            getattr(waypoint, "transform", None), "location", None
        )
        if waypoint_location is None:
            return math.inf
        try:
            return math.sqrt(
                (float(location.x) - float(waypoint_location.x)) ** 2
                + (float(location.y) - float(waypoint_location.y)) ** 2
                + (float(location.z) - float(waypoint_location.z)) ** 2
            )
        except (AttributeError, TypeError, ValueError):
            return math.inf

    def _projected_carla_driving_waypoint(self, location):
        """Return CARLA's nearest driving waypoint and the projection error."""
        carla_map = getattr(self, "map", None)
        get_waypoint = getattr(carla_map, "get_waypoint", None)
        if not callable(get_waypoint):
            return None, math.inf
        waypoint = get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return None, math.inf
        return waypoint, self._carla_waypoint_distance(location, waypoint)

    def _mapped_lanes_for_carla_waypoint(self, waypoint):
        """Return raw SUMO lanes mapped from one projected CARLA waypoint."""
        if waypoint is None:
            return ()
        key = f"{waypoint.road_id}_{waypoint.lane_id}"
        network_helper = getattr(self, "network_helper", None)
        lane_mapping = getattr(
            network_helper,
            "scenic_to_metsr_map_lanes",
            getattr(self, "scenic_to_metsr_map", {}),
        )
        return tuple(str(lane) for lane in lane_mapping.get(key, ()))

    def _mapped_carla_observation(self, location):
        """Map a close CARLA driving waypoint to a METS-R road and lane."""
        waypoint, projection_error = self._projected_carla_driving_waypoint(location)
        if (
            waypoint is None
            or projection_error > CARLA_OWNED_PROJECTION_TOLERANCE
        ):
            return None, None

        mapped_lanes = list(self._mapped_lanes_for_carla_waypoint(waypoint))
        if not mapped_lanes:
            return None, None

        current_position = (location.x, -location.y, location.z)
        if len(mapped_lanes) > 1:
            metsr_road = self.identify_nearest_road(
                current_position, mapped_lanes
            )
        else:
            metsr_road, _ = self._mapped_lane_query(mapped_lanes[0])
        controlled_segments = getattr(self, "carla_control_segments", ())
        if (
            metsr_road not in self.network_helper.metsr_represented_roads
            and metsr_road not in controlled_segments
        ):
            return None, None

        observed_lane_id, _ = self._nearest_mapped_lane(
            current_position, mapped_lanes, metsr_road
        )
        if observed_lane_id is not None and observed_lane_id < 0:
            observed_lane_id = None
        return metsr_road, observed_lane_id

    @staticmethod
    def _road_entry_rejection_reason(record):
        """Return the stable reason code from a rejected road-entry record."""
        if not isinstance(record, dict):
            return None
        for field in ("errorCode", "message"):
            value = record.get(field)
            if value is not None and str(value).strip():
                return str(value)
        return None

    def _record_road_entry_hold(
        self, obj, source=None, target=None, reason=None
    ):
        """Record one rejected road-entry request without changing control state."""
        diagnostics = getattr(self, "road_entry_hold_diagnostics", None)
        if diagnostics is None:
            diagnostics = self.road_entry_hold_diagnostics = {}

        step = getattr(self, "count", 0)
        diagnostic = diagnostics.get(obj)
        if diagnostic is None:
            diagnostic = {
                "start_step": step,
                "retries": 0,
                "source": source,
                "target": target,
                "latest_reason": reason,
                "next_report_step": step + ROAD_ENTRY_HOLD_REPORT_INTERVAL,
            }
            diagnostics[obj] = diagnostic
            name = getattr(obj, "name", str(obj))
            print(
                f"Road-entry hold started: {name} at step {step}; "
                f"edge {source}->{target}; reason={reason}; retries=0"
            )
            return diagnostic

        diagnostic["retries"] += 1
        if source is not None:
            diagnostic["source"] = source
        if target is not None:
            diagnostic["target"] = target
        # Some retry replies omit errorCode. Preserve the latest explicit primary
        # code rather than replacing useful context with an empty value.
        if reason is not None:
            diagnostic["latest_reason"] = reason
        return diagnostic

    def _report_road_entry_holds(self):
        """Report each continuing hold once per fixed simulation-tick interval."""
        diagnostics = getattr(self, "road_entry_hold_diagnostics", {})
        step = getattr(self, "count", 0)
        for obj, diagnostic in tuple(diagnostics.items()):
            next_report = diagnostic["next_report_step"]
            if step < next_report:
                continue
            name = getattr(obj, "name", str(obj))
            held_ticks = step - diagnostic["start_step"]
            print(
                f"Road-entry hold waiting: {name} at step {step}; "
                f"held_ticks={held_ticks}; retries={diagnostic['retries']}; "
                f"edge {diagnostic['source']}->{diagnostic['target']}; "
                f"reason={diagnostic['latest_reason']}"
            )
            # Only one message is emitted per object per simulator step. If a
            # caller advances the counter by more than one interval, align the
            # next message with the original hold-start cadence.
            intervals = (step - next_report) // ROAD_ENTRY_HOLD_REPORT_INTERVAL + 1
            diagnostic["next_report_step"] = (
                next_report + intervals * ROAD_ENTRY_HOLD_REPORT_INTERVAL
            )

    def _finish_road_entry_hold_diagnostic(self, obj):
        """Log and clear diagnostics after an actual road-entry hold release."""
        diagnostics = getattr(self, "road_entry_hold_diagnostics", {})
        diagnostic = diagnostics.pop(obj, None)
        if diagnostic is None:
            return
        step = getattr(self, "count", 0)
        name = getattr(obj, "name", str(obj))
        print(
            f"Road-entry hold released: {name} at step {step}; "
            f"held_ticks={max(0, step - diagnostic['start_step'])}; "
            f"retries={diagnostic['retries']}; "
            f"edge {diagnostic['source']}->{diagnostic['target']}; "
            f"reason={diagnostic['latest_reason']}"
        )

    @staticmethod
    def _notify_pcla_control_applied(obj, control):
        """Tell PCLA which control CARLA actually received for this object."""
        pcla = getattr(obj, "pcla", None)
        if pcla is None:
            return

        notify = getattr(pcla, "notify_control_applied", None)
        if callable(notify):
            notify(control)
            return

        # Compatibility with PCLA versions predating the explicit integration
        # hook. SimLingo predicts its next UKF state from ``agent.control``.
        agent = getattr(pcla, "agent_instance", None)
        if agent is not None and hasattr(agent, "control"):
            agent.control = control

    @staticmethod
    def _notify_pcla_pose_changed(obj):
        """Invalidate PCLA localization after Scenic moves a CARLA actor."""
        pcla = getattr(obj, "pcla", None)
        if pcla is None:
            return

        notify = getattr(pcla, "notify_pose_changed", None)
        if callable(notify):
            notify()
            return

        # Compatibility fallback for SimLingo releases without the wrapper hook.
        # The next observation reinitializes the UKF at the authoritative pose.
        agent = getattr(pcla, "agent_instance", None)
        if agent is None:
            return
        if hasattr(agent, "filter_initialized"):
            agent.filter_initialized = False
        state_log = getattr(agent, "state_log", None)
        clear_state_log = getattr(state_log, "clear", None)
        if callable(clear_state_log):
            clear_state_log()

    def _is_pcla_controlled_ego(self, obj):
        """Return whether CARLA/PCLA has permanent motion authority for obj."""
        return (
            obj is getattr(self, "ego", None)
            and getattr(obj, "pcla", None) is not None
        )

    def _clear_pcla_ego_reconciliation_state(self, obj):
        """Discard generic bridge state which must never constrain a PCLA ego."""
        held_objects = getattr(self, "road_entry_holds", {})
        if obj in held_objects:
            actor = getattr(obj, "carlaActor", None)
            disable_constant_velocity = getattr(
                actor, "disable_constant_velocity", None
            )
            if callable(disable_constant_velocity):
                disable_constant_velocity()
        held_objects.pop(obj, None)
        getattr(self, "road_entry_hold_diagnostics", {}).pop(obj, None)
        getattr(self, "pending_route_refreshes", {}).pop(obj, None)
        getattr(self, "pending_lane_reconciliation_verifications", {}).pop(
            obj, None
        )
        getattr(self, "same_road_departure_lane_verifications", {}).pop(
            obj, None
        )
        getattr(self, "_pending_pcla_pose_resets", set()).discard(obj)

    def _defer_pcla_pose_reset(self, obj):
        """Reset PCLA only after CARLA publishes the corrected actor pose."""
        pending = getattr(self, "_pending_pcla_pose_resets", None)
        if pending is None:
            pending = self._pending_pcla_pose_resets = set()
        pending.add(obj)

    @staticmethod
    def _road_entry_vehicle_control(
        actor, *, throttle, steer, brake, hand_brake
    ):
        """Build a safety control without changing the actor's drivetrain."""
        get_control = getattr(actor, "get_control", None)
        current = get_control() if callable(get_control) else None
        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=hand_brake,
            reverse=bool(getattr(current, "reverse", False)),
            manual_gear_shift=bool(
                getattr(current, "manual_gear_shift", False)
            ),
            gear=int(getattr(current, "gear", 0)),
        )

    def _hold_for_road_entry(
        self, obj, source=None, target=None, reason=None
    ):
        """Stop a CARLA actor while METS-R reports a retryable road entry."""
        actor = getattr(obj, "carlaActor", None)
        if actor is None:
            return
        if obj not in self.road_entry_holds:
            self.road_entry_holds[obj] = bool(
                getattr(obj, "active_autopilot", False)
            )
        self._record_road_entry_hold(obj, source, target, reason)
        self._enforce_road_entry_hold(obj)

    def _enforce_road_entry_hold(self, obj):
        """Physically stop a held CARLA actor for the next simulator tick."""
        actor = getattr(obj, "carlaActor", None)
        if actor is None:
            return

        # Keep the actor registered with Traffic Manager while it waits. CARLA
        # discards the actor's waypoint buffer when autopilot is disabled; a
        # later re-registration would therefore lose the custom METS-R route.
        # A partial/failed stop is not a safe hold: retain its bookkeeping and
        # fail closed so it can never be mistaken for a stationary actor.
        try:
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            set_angular_velocity = getattr(
                actor, "set_target_angular_velocity", None
            )
            if set_angular_velocity is not None:
                set_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            hold_control = self._road_entry_vehicle_control(
                actor,
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
            )
            actor.apply_control(hold_control)
            self._notify_pcla_control_applied(obj, hold_control)
            actor.enable_constant_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except (AttributeError, RuntimeError) as exc:
            raise SimulationCreationError(
                f"Unable to place {obj} into the road-entry safety hold"
            ) from exc

    def _release_road_entry_hold(
        self,
        obj,
        path_already_installed=False,
        remaining_route=None,
    ):
        actor = getattr(obj, "carlaActor", None)
        autopilot_was_active = self.road_entry_holds.get(obj, False)
        if actor is None:
            self.road_entry_holds.pop(obj, None)
            getattr(self, "road_entry_hold_diagnostics", {}).pop(obj, None)
            getattr(self, "_pending_pcla_pose_resets", set()).discard(obj)
            return

        if autopilot_was_active and not path_already_installed:
            # Rebuild the remaining METS-R route from the actor's current pose.
            # Even though the actor stayed registered, Traffic Manager may have
            # already consumed the custom path into its finite waypoint buffer
            # before the hold began. Reinstalling the remaining route avoids a
            # random branch when that buffer runs out after release.
            if remaining_route is None:
                trajectory = self.metsr_trajectory_to_carla(obj)
            else:
                trajectory = self.generate_carla_trajectory(
                    list(remaining_route), obj
                )
            if not trajectory:
                raise SimulationCreationError(
                    f"Cannot release the road-entry hold for {obj}: METS-R "
                    "returned no remaining CARLA trajectory"
                )
            obj.trajectory = trajectory
            self.tm.set_path(actor, trajectory)
            self.tm.auto_lane_change(actor, False)

        # Do not clear the hold until route restoration and physical release
        # controls have both succeeded. Otherwise a transient CARLA failure can
        # leave constant zero velocity or full braking active indefinitely while
        # Scenic incorrectly records the hold as released.
        try:
            actor.disable_constant_velocity()
            release_control = self._road_entry_vehicle_control(
                actor,
                throttle=0.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
            )
            actor.apply_control(release_control)
            self._notify_pcla_control_applied(obj, release_control)
            pending_pose_resets = getattr(
                self, "_pending_pcla_pose_resets", set()
            )
            if obj in pending_pose_resets:
                # The verification tick has now published the authoritative pose
                # to PCLA's sensor queue. Resetting here prevents the held behavior
                # from consuming the reset with the stale pre-transform frame.
                self._notify_pcla_pose_changed(obj)
                pending_pose_resets.discard(obj)
        except (AttributeError, RuntimeError) as exc:
            raise SimulationCreationError(
                f"Unable to physically release the road-entry hold for {obj}"
            ) from exc
        self.road_entry_holds.pop(obj, None)
        self._finish_road_entry_hold_diagnostic(obj)

    def _schedule_committed_route_refresh(self, obj, road_id):
        """Schedule one Traffic Manager path refresh on a committed road."""
        if road_id is not None:
            self.pending_route_refreshes[obj] = str(road_id)

    def _install_pending_target_carla_path(
        self,
        obj,
        current_road_id,
        accepted_target,
        authoritative_lane_id,
        remaining_route,
        *,
        target_start=None,
        install=True,
    ):
        """Install or precompute a lane-constrained pending-target path."""
        route_tail = self._route_tail_at_direct_target(
            current_road_id, accepted_target, remaining_route
        )
        if not route_tail:
            raise SimulationCreationError(
                f"Cannot anchor the pending target route for {obj}: edge "
                f"{current_road_id}->{accepted_target} is not the live route prefix "
                f"{list(self._normalized_road_route(remaining_route))}"
            )
        if authoritative_lane_id is None:
            raise SimulationCreationError(
                f"METS-R accepted pending edge {current_road_id}->{accepted_target} "
                "without returning its authoritative target laneID"
            )
        trajectory = self.generate_carla_trajectory(
            list(route_tail),
            obj,
            required_first_lane_id=authoritative_lane_id,
            target_start=target_start,
        )
        if not trajectory:
            raise SimulationCreationError(
                f"Cannot install the pending target route for {obj} on road "
                f"{accepted_target}: no constrained CARLA trajectory was generated"
            )
        if install:
            self.tm.set_path(obj.carlaActor, trajectory)
            self.tm.auto_lane_change(obj.carlaActor, False)
            obj.trajectory = trajectory
        return trajectory

    @staticmethod
    def _heading_delta(first, second):
        return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _point_to_segment_distance(point, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        ratio = (
            (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
        ) / length_squared
        ratio = min(1.0, max(0.0, ratio))
        closest = (start[0] + ratio * dx, start[1] + ratio * dy)
        return math.hypot(point[0] - closest[0], point[1] - closest[1])

    @staticmethod
    def _collision_extent(actor):
        bounding_box = getattr(actor, "bounding_box", None)
        extent = getattr(bounding_box, "extent", None)
        if extent is None:
            return 2.5, 1.5
        offset = getattr(bounding_box, "location", None)
        offset_x = float(getattr(offset, "x", 0.0))
        offset_y = float(getattr(offset, "y", 0.0))
        offset_z = float(getattr(offset, "z", 0.0))
        return (
            math.hypot(float(extent.x), float(extent.y))
            + math.hypot(offset_x, offset_y),
            max(0.1, float(extent.z) + abs(offset_z)),
        )

    def _pending_lane_reconciliation_target(
        self, obj, target_road_id, authoritative_lane_id
    ):
        """Return a tightly validated lateral target for a pending transition."""
        actor = getattr(obj, "carlaActor", None)
        if actor is None:
            raise SimulationCreationError(
                f"Cannot reconcile the pending target lane for {obj}: no CARLA actor"
            )
        raw_target_lane = self._resolve_required_initial_lane(
            target_road_id, authoritative_lane_id
        )
        old_transform = actor.get_transform()
        old_location = actor.get_location()
        old_rotation = old_transform.rotation
        target_road, target_lane_index = self._mapped_lane_query(raw_target_lane)
        centerline = self._query_mapped_centerline(target_road, target_lane_index)
        old_progress, _, _ = self._polyline_progress(
            (old_location.x, -old_location.y), centerline
        )
        if old_progress is None:
            raise SimulationCreationError(
                f"Cannot project {obj} onto authoritative lane {raw_target_lane}"
            )

        self._index_carla_waypoints()
        candidates = []
        for carla_key in self._sumo_lane_to_carla_keys.get(raw_target_lane, ()):
            for waypoint in self._carla_waypoints_by_key.get(carla_key, ()):
                if getattr(waypoint, "is_junction", False):
                    continue
                waypoint_key = f"{waypoint.road_id}_{waypoint.lane_id}"
                mapped = {
                    str(lane)
                    for lane in self.scenic_to_metsr_map.get(waypoint_key, ())
                }
                if raw_target_lane not in mapped:
                    continue
                location = waypoint.transform.location
                shift = math.dist(
                    (old_location.x, old_location.y, old_location.z),
                    (location.x, location.y, location.z),
                )
                if shift > PENDING_LANE_RECONCILIATION_MAX_SHIFT + 1e-6:
                    continue
                if self._heading_delta(
                    old_rotation.yaw, waypoint.transform.rotation.yaw
                ) > PENDING_LANE_RECONCILIATION_MAX_HEADING_DELTA:
                    continue
                progress, _, lateral_error = self._polyline_progress(
                    (location.x, -location.y), centerline
                )
                if (
                    progress is None
                    or lateral_error > PENDING_LANE_RECONCILIATION_MAP_TOLERANCE
                    or abs(progress - old_progress)
                    > PENDING_LANE_RECONCILIATION_MAX_LONGITUDINAL_DELTA
                ):
                    continue
                candidates.append((abs(progress - old_progress), shift, waypoint))

        if not candidates:
            raise _PendingLaneReconciliationUnavailable(
                f"No safe parallel CARLA waypoint can reconcile {obj} to "
                f"authoritative lane {raw_target_lane} on road {target_road_id}"
            )
        _, _, target_waypoint = min(candidates, key=lambda item: item[:2])
        current_waypoint = self.map.get_waypoint(
            old_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        current_surface_z = (
            current_waypoint.transform.location.z
            if current_waypoint is not None
            else old_location.z
        )
        z_offset = old_location.z - current_surface_z
        target_location = target_waypoint.transform.location
        target_transform = carla.Transform(
            carla.Location(
                x=target_location.x,
                y=target_location.y,
                z=target_location.z + z_offset,
            ),
            target_waypoint.transform.rotation,
        )
        return raw_target_lane, old_transform, target_transform

    def _assert_pending_lane_reconciliation_clear(
        self, actor, old_transform, target_transform
    ):
        """Reject a lateral reconciliation whose swept footprint is occupied."""
        try:
            actors = list(self.carla_world.get_actors())
        except Exception as exc:
            raise SimulationCreationError(
                "Cannot inspect CARLA actors before pending-lane reconciliation"
            ) from exc
        own_radius, own_height = self._collision_extent(actor)
        start = (old_transform.location.x, old_transform.location.y)
        end = (target_transform.location.x, target_transform.location.y)
        own_id = getattr(actor, "id", None)
        for other in actors:
            if other is actor or (
                own_id is not None and getattr(other, "id", None) == own_id
            ):
                continue
            type_id = str(getattr(other, "type_id", ""))
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                continue
            try:
                other_location = other.get_location()
            except Exception as exc:
                raise SimulationCreationError(
                    f"Cannot inspect CARLA actor {getattr(other, 'id', other)} "
                    "before pending-lane reconciliation"
                ) from exc
            other_radius, other_height = self._collision_extent(other)
            if abs(other_location.z - target_transform.location.z) > (
                own_height + other_height + PENDING_LANE_RECONCILIATION_COLLISION_MARGIN
            ):
                continue
            clearance = own_radius + other_radius + PENDING_LANE_RECONCILIATION_COLLISION_MARGIN
            distance = self._point_to_segment_distance(
                (other_location.x, other_location.y), start, end
            )
            if distance <= clearance:
                raise _PendingLaneReconciliationOccupied(
                    f"Pending-lane reconciliation for actor {getattr(actor, 'id', actor)} "
                    f"has an occupied swept corridor: blocker="
                    f"{getattr(other, 'id', other)}, distance={distance:.2f} m, "
                    f"required_clearance={clearance:.2f} m"
                )

    def _apply_carla_transform_without_tick(self, actor, transform, context):
        """Apply exactly one CARLA transform and require RPC acknowledgement."""
        try:
            responses = self.carla_client.apply_batch_sync(
                [carla.command.ApplyTransform(actor.id, transform)], False
            )
        except Exception as exc:
            raise SimulationCreationError(f"CARLA failed while {context}") from exc
        if len(responses) != 1:
            raise SimulationCreationError(
                f"CARLA returned {len(responses)} responses while {context}"
            )
        response = responses[0]
        has_error = response.has_error() if callable(response.has_error) else response.has_error
        if has_error:
            raise SimulationCreationError(
                f"CARLA rejected {context}: {getattr(response, 'error', None)}"
            )

    def _refresh_committed_carla_route(
        self, obj, veh_data, observed_road_id, remaining_route
    ):
        """Install a fresh path once CARLA and METS-R agree on a new road.

        CARLA Traffic Manager imports paths as locations. At a junction with
        parallel successors it can therefore choose the correct road but the
        wrong lane. Reinstalling the remaining route after each committed edge
        makes any required lane change part of the current-road path, before
        the next junction, instead of a lateral correction after that junction.
        """
        target = self.pending_route_refreshes.get(obj)
        if target is None:
            return False
        if str(_metsr_vehicle_road(veh_data)) != target:
            return False
        if observed_road_id is None or str(observed_road_id) != target:
            return False
        if not isinstance(remaining_route, (list, tuple)):
            return False

        route = [str(road_id) for road_id in remaining_route if road_id is not None]
        if not route or route[0] != target:
            return False

        # Clear the pending marker only after path generation and installation
        # succeed. A failure therefore remains fail-closed and can be diagnosed
        # without silently allowing Traffic Manager to choose another branch.
        trajectory = self.generate_carla_trajectory(route, obj)
        if not trajectory:
            raise SimulationCreationError(
                f"Cannot refresh the committed route for {obj} on road "
                f"{target}: no CARLA trajectory was generated"
            )
        self.tm.set_path(obj.carlaActor, trajectory)
        self.tm.auto_lane_change(obj.carlaActor, False)
        obj.trajectory = trajectory
        self.pending_route_refreshes.pop(obj, None)
        return True

    def _ensure_queue_admission_state(self):
        if not hasattr(self, "admitted_queue_vehicles"):
            self.admitted_queue_vehicles = {}

    def _check_queue_admission_spawn_deadlines(self):
        """Fail if an admitted external vehicle never acquires a CARLA actor."""
        self._ensure_queue_admission_state()
        current_tick = getattr(self, "count", 0)
        objects_by_vehicle_id = {
            vehicle_id: obj for obj, vehicle_id in self.pv_id_map.items()
        }
        for vehicle_id, admission in tuple(self.admitted_queue_vehicles.items()):
            obj = objects_by_vehicle_id.get(vehicle_id)
            if obj is not None and getattr(obj, "carla_actor_flag", False):
                self.admitted_queue_vehicles.pop(vehicle_id, None)
                continue
            admitted_at = admission["admittedAt"]
            if current_tick - admitted_at >= COSIM_ADMISSION_SPAWN_MAX_TICKS:
                raise SimulationCreationError(
                    f"Vehicle {vehicle_id} was admitted from the METS-R "
                    f"departure queue onto COSIM road {admission['roadId']} "
                    f"at step {admitted_at}, but CARLA did not spawn it within "
                    f"{COSIM_ADMISSION_SPAWN_MAX_TICKS} Scenic steps"
                )

    def _service_metsr_entering_queues(self):
        """Admit ready Scenic vehicles from controlled-road departure queues."""
        self._check_queue_admission_spawn_deadlines()
        if not self.carla_control_roads:
            return

        queue_response = _require_metsr_response(
            self.metsr_client.query_cosim_entering_vehicle_queue(),
            "coSimEnteringVehicleQueue",
        )
        queue_records = queue_response.get("data")
        if not isinstance(queue_records, (list, tuple)):
            raise METSRControlError(
                "coSimEnteringVehicleQueue returned no data records",
                response=queue_response,
            )

        managed_private_ids = set(self.pv_id_map.values())
        seen_roads = set()
        selected_vehicle_ids = set()
        admission_requests = []
        for road_queue in queue_records:
            if not isinstance(road_queue, dict) or road_queue.get("status") != "ok":
                raise METSRControlError(
                    "coSimEnteringVehicleQueue returned an invalid road record",
                    response=queue_response,
                    record=road_queue,
                )
            road_id = road_queue.get("segmentId")
            if road_id is None:
                raise METSRControlError(
                    "coSimEnteringVehicleQueue omitted segmentId",
                    response=queue_response,
                    record=road_queue,
                )
            road_id = str(road_id)
            if road_id in seen_roads:
                raise METSRControlError(
                    f"coSimEnteringVehicleQueue duplicated road {road_id!r}",
                    response=queue_response,
                    record=road_queue,
                )
            seen_roads.add(road_id)
            queue = road_queue.get("queue")
            if not isinstance(queue, (list, tuple)):
                raise METSRControlError(
                    f"coSimEnteringVehicleQueue returned an invalid queue for {road_id}",
                    response=queue_response,
                    record=road_queue,
                )
            if (
                road_id not in self.carla_control_roads
                or road_id not in self.active_bubble_metsr_roads
            ):
                continue

            # ID-based admission lets Scenic skip unsupported public entries
            # without requiring them to leave the native queue first.
            for entry in queue:
                if not isinstance(entry, dict) or entry.get("ready") is not True:
                    continue
                vehicle_id = entry.get("vehicleId")
                if (
                    entry.get("isPrivate") is not True
                    or vehicle_id not in managed_private_ids
                    or vehicle_id in self.admitted_queue_vehicles
                    or vehicle_id in selected_vehicle_ids
                ):
                    continue

                request = {
                    "vehicleId": vehicle_id,
                    "internalVehicleId": entry.get("internalVehicleId"),
                    "isPrivate": True,
                    "roadId": road_id,
                }
                request = {
                    key: value
                    for key, value in request.items()
                    if value is not None
                }
                selected_vehicle_ids.add(vehicle_id)
                admission_requests.append(request)

        if not admission_requests:
            return

        request_payload = (
            admission_requests[0]
            if len(admission_requests) == 1
            else admission_requests
        )
        admission_response = self.metsr_client.enter_road_from_queue(
            requests=request_payload
        )
        if not isinstance(admission_response, dict):
            raise METSRControlError(
                "METS-R returned no valid response for enterRoadFromQueue",
                response=admission_response,
            )
        if admission_response.get("messageType") != "enterRoadFromQueue":
            raise METSRControlError(
                "METS-R returned the wrong response type for enterRoadFromQueue",
                response=admission_response,
            )
        if admission_response.get("status") not in {"ok", "partial"}:
            raise METSRControlError(
                admission_response.get("message")
                or "METS-R rejected enterRoadFromQueue",
                response=admission_response,
            )
        admission_records = admission_response.get("data")
        if (
            not isinstance(admission_records, (list, tuple))
            or len(admission_records) != len(admission_requests)
        ):
            raise METSRControlError(
                "enterRoadFromQueue did not return exactly one result per request",
                response=admission_response,
            )

        records_by_key = {}
        for admitted in admission_records:
            if not isinstance(admitted, dict):
                raise METSRControlError(
                    "enterRoadFromQueue returned an invalid result record",
                    response=admission_response,
                    record=admitted,
                )
            key = (
                str(admitted.get("vehicleId")),
                str(admitted.get("roadId")),
            )
            if key in records_by_key:
                raise METSRControlError(
                    f"enterRoadFromQueue duplicated result {key!r}",
                    response=admission_response,
                    record=admitted,
                )
            records_by_key[key] = admitted

        for request in admission_requests:
            vehicle_id = request["vehicleId"]
            road_id = str(request["roadId"])
            admitted = records_by_key.get((str(vehicle_id), road_id))
            if admitted is None:
                raise METSRControlError(
                    "enterRoadFromQueue omitted the requested "
                    f"vehicle {vehicle_id} on road {road_id}",
                    response=admission_response,
                )
            if admitted.get("status") == "error":
                error = METSRControlError(
                    admitted.get("message")
                    or admitted.get("errorCode")
                    or f"METS-R rejected vehicle {vehicle_id} during enterRoadFromQueue",
                    response=admission_response,
                    record=admitted,
                )
                if error.retryable:
                    continue
                raise error
            if (
                admitted.get("status") != "ok"
                or admitted.get("isPrivate") is not True
                or admitted.get("state") is not None
            ):
                raise METSRControlError(
                    "enterRoadFromQueue did not confirm the requested "
                    f"vehicle {vehicle_id} on road {road_id}",
                    response=admission_response,
                    record=admitted,
                )
            self.admitted_queue_vehicles[vehicle_id] = {
                "roadId": road_id,
                "admittedAt": getattr(self, "count", 0),
            }

    @staticmethod
    def _client_sync_diagnostic_due(count, synchronized_any):
        """Whether to run the expensive read-only consistency diagnostic."""
        return synchronized_any and count % 200 == 0

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
            for index, name in enumerate(("segment_id", "lane_index", "connector_path_id")):
                values = [target[index] if target else None for target in targets]
                if any(value is not None for value in values):
                    selectors[name] = values[0] if len(values) == 1 else values
        response = self.metsr_client.teleport_cosim_vehicle(
            field("vehicle_id"), field("x"), field("y"),
            z=field("z"), bearing=field("bearing"), speed=field("speed"),
            private_veh=field("private_veh"),
            transform_coords=field("transform_coords"), **selectors,
        )
        _, acknowledgements = _require_vehicle_batch_response(
            response, "teleportCoSimVeh",
            [update["vehicle_id"] for update in updates],
        )
        # Validate the entire batch before committing any local owner changes.
        # A rejected or ambiguous batch aborts the run; teardown resets METS-R.
        for update, target in zip(updates, targets):
            vehicle_id = update["vehicle_id"]
            record = acknowledgements[str(vehicle_id)]
            if target:
                segment, lane, path = target
                valid = (
                    record.get("controlMode") == "native"
                    and record.get("releasedFromCoSim") is True
                    and str(record.get("segmentId")) == segment
                    and (lane is None or record.get("laneIndex") == lane)
                    and (path is None or record.get("connectorPathId") == path)
                )
            else:
                valid = (record.get("controlMode") == "cosim"
                         and record.get("releasedFromCoSim") is False)
            if not valid:
                raise METSRControlError(
                    f"METS-R did not confirm {'native handback' if target else 'CARLA pose update'} "
                    f"for vehicle {vehicle_id}", response=response, record=record,
                )

        self._ensure_carla_authority_state()
        for update, target in zip(updates, targets):
            obj = update["object"]
            record = acknowledgements[str(update["vehicle_id"])]
            if target:
                obj.spawn_guard = 0
                self.remove_bubble_object(obj)
                continue
            segment = record.get("segmentId")
            self._carla_authoritative_segments[obj] = (
                str(segment) if segment is not None else None
            )
            destination = update["vehicle_state"].get("destinationRoadId")
            if destination is None:
                destination = update["cosim_record"].get("destinationRoadId")
            if (destination is not None and segment is not None
                    and record.get("segmentType") == "road"
                    and str(segment) == str(destination)
                    and obj not in self.completed_route):
                if obj is not getattr(self, "ego", None):
                    retirement = self.metsr_client.reach_dest(
                        update["vehicle_id"], private_veh=update["private_veh"]
                    )
                    _require_all_success_control_response(retirement, "reachDest")
                    records = retirement.get("data", ())
                    # The native reachDest handler returns one OK record but
                    # does not include vehicleId on successful scalar requests.
                    if (not isinstance(records, (list, tuple)) or len(records) != 1
                            or not isinstance(records[0], dict)
                            or records[0].get("status") != "ok"
                            or (records[0].get("vehicleId") is not None
                                and str(records[0]["vehicleId"]) != str(update["vehicle_id"]))):
                        raise METSRControlError(
                            f"reachDest did not confirm completion of vehicle {update['vehicle_id']}",
                            response=retirement,
                        )
                self.completed_route[obj] = True
                obj.finished_route = self.count

    def _remove_completed_carla_objects(self):
        """Remove completed NPCs once, after native retirement is acknowledged."""
        for obj in tuple(self.carla_actors):
            if obj is getattr(self, "ego", None) or obj not in self.completed_route:
                continue
            vehicle_id = self.pv_id_map.get(obj)
            getattr(self, "admitted_queue_vehicles", {}).pop(vehicle_id, None)
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
            else self._collect_metsr_vehicle_data(carla_actors)
        )
        cosim_response = _require_metsr_response(
            self.metsr_client.query_cosim_vehicle(), "coSimVehicle"
        )
        live_private_records = {}
        for record in cosim_response.get("data", ()):
            if not isinstance(record, dict) or record.get("isPrivate") is not True:
                continue
            vehicle_id = record.get("vehicleId")
            if vehicle_id is not None:
                live_private_records.setdefault(str(vehicle_id), []).append(record)

        synchronized_any = False
        pending_updates = []
        for carla_object in carla_actors:
            try:
                actor = carla_object.carlaActor
                if not self._carla_actor_is_alive(actor):
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

            if self._is_pcla_controlled_ego(carla_object):
                self._clear_pcla_ego_reconciliation_state(carla_object)

            transform = actor.get_transform()
            bearing = _utils.get_metsr_rotation(transform.rotation.yaw)
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
                    "handoff": self._native_handoff_target(carla_object, location),
                    "vehicle_state": vehicle_state,
                    "cosim_record": cosim_record,
                }
            )

        self._publish_carla_vehicle_updates(pending_updates)
        synchronized_any = synchronized_any or bool(pending_updates)

        if self._client_sync_diagnostic_due(self.count, synchronized_any):
            self.check_client_synchronization()


    def identify_nearest_road(self, obj: Object, roads: list[str] ) -> str:
        """Resolve an ambiguous lane mapping by distance to each full lane."""
        best_road = None
        best_dist = math.inf
        unlocated_internal_segments = set()
        position = getattr(obj, "position", obj)
        point = position[:2]
        for road_lane in roads:
            road, lane_index = self._mapped_lane_query(road_lane)
            if lane_index is None:
                if self._is_controlled_internal_segment(road):
                    unlocated_internal_segments.add(str(road))
                continue
            controlled_segments = getattr(self, "carla_control_segments", ())
            if (
                road not in self.network_helper.metsr_represented_roads
                and road not in controlled_segments
            ):
                continue

            centerline = self._query_mapped_centerline(road, lane_index)
            dist = self._point_to_polyline_distance(point, centerline)
            if dist < best_dist:
                best_dist = dist
                best_road = road

        if best_road is None and unlocated_internal_segments:
            owners = {
                self.metsr_internal_edge_to_connector[segment]
                for segment in unlocated_internal_segments
            }
            # SUMO may split one CARLA junction lane into consecutive edges.
            # They are unambiguous for ownership when every edge has one owner;
            # the exact connector path is resolved separately from the pose.
            if len(owners) == 1:
                return min(unlocated_internal_segments)
        return best_road

    def _mapped_lane_query(self, road_lane):
        """Return a road ID and METS-R's filtered index for a SUMO lane ID."""
        road_lane = str(road_lane)
        lane_indices = getattr(self, "metsr_lane_indices", None)
        if lane_indices is not None and road_lane in lane_indices:
            road, separator, _ = road_lane.rpartition("_")
            if separator:
                return road, lane_indices[road_lane]

        network_helper = getattr(self, "network_helper", None)
        represented_roads = getattr(
            network_helper, "metsr_represented_roads", ()
        )
        if road_lane in represented_roads:
            return road_lane, -1

        road, separator, lane_text = road_lane.rpartition("_")
        if not separator:
            return road_lane, -1

        if lane_indices is not None:
            return road, None

        # Backward-compatible fallback for adapters constructed without an XML
        # lane-index map (including lightweight unit-test fixtures).
        try:
            return road, int(lane_text)
        except ValueError:
            return road, None

    def _source_projection_error_limit(
        self, obj, lane_key, base_limit, at_terminal
    ):
        """Allow a bounded map-end mismatch only on the confirmed CARLA lane."""
        if not at_terminal:
            return base_limit
        waypoint = self.map.get_waypoint(
            utils.scenicToCarlaLocation(obj.position),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if (
            waypoint is not None
            and f"{waypoint.road_id}_{waypoint.lane_id}" == lane_key
        ):
            return base_limit + float(waypoint.lane_width)
        return base_limit

    def _query_mapped_centerline(self, road, lane_index):
        """Query and cache centerline data without assuming a valid record."""
        cache_key = (road, lane_index)
        if cache_key in self.metsr_road_cache:
            return self.metsr_road_cache[cache_key]

        response = self.metsr_client.query_centerline(
            road, lane_index=lane_index, transform_coords=True
        )
        centerline = []
        if isinstance(response, dict):
            records = response.get("data")
            if isinstance(records, list) and records:
                record = records[0]
                if isinstance(record, dict):
                    points = record.get("centerline")
                    if isinstance(points, list):
                        centerline = points
        self.metsr_road_cache[cache_key] = centerline
        return centerline

    @staticmethod
    def _normalized_connector_path_id(value):
        """Return a non-negative connector-local path ID, or ``None``."""
        if value is None or isinstance(value, bool):
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if numeric < 0 or str(numeric) != str(value):
            return None
        return numeric

    def _connector_path_records(self, connector_id):
        """Return cached path records for one server-owned connector ID."""
        connector_id = str(connector_id)
        cache = getattr(self, "metsr_connector_path_cache", None)
        if cache is None:
            cache = self.metsr_connector_path_cache = {}
        if connector_id in cache:
            return cache[connector_id]

        connector = getattr(self, "metsr_connector_records", {}).get(
            connector_id, {}
        )
        paths = connector.get("paths") if isinstance(connector, dict) else None
        if not isinstance(paths, (list, tuple)):
            query = getattr(self.metsr_client, "query_connector_path", None)
            paths = None
            if callable(query):
                response = query(connector_id)
                if isinstance(response, dict):
                    data = response.get("data")
                    if isinstance(data, (list, tuple)):
                        for record in data:
                            if (
                                isinstance(record, dict)
                                and str(record.get("connectorId")) == connector_id
                            ):
                                nested = record.get("paths")
                                paths = nested if isinstance(
                                    nested, (list, tuple)
                                ) else [record]
                                break
        normalized = []
        for record in paths or ():
            if not isinstance(record, dict):
                continue
            path_id = self._normalized_connector_path_id(
                record.get("connectorPathId")
            )
            if path_id is None:
                continue
            item = dict(record)
            item["connectorId"] = connector_id
            item["connectorPathId"] = path_id
            for field in ("sourceRoadId", "targetRoadId"):
                if isinstance(connector, dict) and connector.get(field) is not None:
                    item.setdefault(field, str(connector[field]))
            normalized.append(item)
        cache[connector_id] = tuple(normalized)
        return cache[connector_id]

    def _connector_path_for_mapped_lane(self, mapped_lane, connector_id=None):
        """Resolve a SUMO internal via-lane to its canonical connector path."""
        mapped_lane = str(mapped_lane)
        lane_pairs = getattr(self, "metsr_internal_lane_connections", {}).get(
            mapped_lane
        )
        if not lane_pairs:
            return None

        expected_pairs = {
            (str(source), str(target)) for source, target in lane_pairs
        }
        ownership = getattr(self, "metsr_internal_edge_to_connector", {})
        owner_ids = {str(owner) for owner in ownership.values()}
        if connector_id is not None:
            requested_owner = str(connector_id)
            if requested_owner not in owner_ids:
                return None
            owner_ids = {requested_owner}

        matches = {}
        for owner in owner_ids:
            owned_internal_edges = {
                str(edge)
                for edge, candidate_owner in ownership.items()
                if str(candidate_owner) == owner
            }
            for record in self._connector_path_records(owner):
                internal_edges = {
                    str(edge) for edge in record.get("internalEdgeIds", ())
                }
                if internal_edges and not internal_edges & owned_internal_edges:
                    continue
                via_lanes = {str(lane) for lane in record.get("viaLaneIds", ())}
                lane_pair = (
                    str(record.get("sourceLaneId")),
                    str(record.get("targetLaneId")),
                )
                if mapped_lane not in via_lanes and lane_pair not in expected_pairs:
                    continue
                key = (owner, record["connectorPathId"])
                matches[key] = record
        if len(matches) != 1:
            return None
        (owner, path_id), record = next(iter(matches.items()))
        return owner, path_id, record

    def _query_connector_path_centerline(self, connector_id, connector_path_id):
        """Query one connector path centerline by connector-local path ID."""
        connector_id = str(connector_id)
        connector_path_id = self._normalized_connector_path_id(
            connector_path_id
        )
        if connector_path_id is None:
            return []
        cache = getattr(self, "metsr_connector_centerline_cache", None)
        if cache is None:
            cache = self.metsr_connector_centerline_cache = {}
        cache_key = (connector_id, connector_path_id)
        if cache_key in cache:
            return cache[cache_key]

        response = self.metsr_client.query_centerline(
            connector_id, lane_index=-1, transform_coords=True
        )
        centerline = []
        if isinstance(response, dict):
            records = response.get("data")
            if isinstance(records, (list, tuple)):
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    if str(record.get("segmentId")) != connector_id:
                        continue
                    centerlines = record.get("centerlines")
                    if (
                        isinstance(centerlines, (list, tuple))
                        and connector_path_id < len(centerlines)
                        and isinstance(
                            centerlines[connector_path_id], (list, tuple)
                        )
                    ):
                        centerline = list(centerlines[connector_path_id])
                    break
        cache[cache_key] = centerline
        return centerline

    def _carla_connector_path_distance(self, connector_id, path_id, location):
        """Measure the actual CARLA curves mapped to one native connector path.

        Converted SUMO curves can differ from the physical CARLA lane. Only
        exact via-lane mappings for the selected connector-local path qualify;
        a nearby lane belonging to another movement is not an alternative.
        """
        path_id = self._normalized_connector_path_id(path_id)
        if path_id is None:
            return math.inf
        cache = getattr(self, "_carla_connector_path_geometry", None)
        if cache is None:
            cache = self._carla_connector_path_geometry = {}
        key = (str(connector_id), path_id)
        if key not in cache:
            paths = [record for record in self._connector_path_records(connector_id)
                     if record["connectorPathId"] == path_id]
            if len(paths) != 1:
                return math.inf
            reverse_mapping = getattr(self, "_sumo_lane_to_carla_keys", {})
            carla_keys = {
                str(carla_key)
                for via_lane in paths[0].get("viaLaneIds", ())
                for carla_key in reverse_mapping.get(str(via_lane), ())
            }
            if not carla_keys:
                return math.inf
            if getattr(self, "_carla_waypoints_by_key", None) is None:
                if not callable(getattr(getattr(self, "map", None),
                                        "generate_waypoints", None)):
                    return math.inf
                self._index_carla_waypoints()
            polylines = []
            for carla_key in sorted(carla_keys):
                sections = {}
                for waypoint in self._carla_waypoints_by_key.get(carla_key, ()):
                    if not waypoint.is_junction:
                        continue
                    sections.setdefault(waypoint.section_id, []).append(waypoint)
                # Never join separate roads, lanes, or lane sections: the
                # artificial joining chord could otherwise admit an unrelated pose.
                for waypoints in sections.values():
                    points = []
                    for waypoint in sorted(waypoints, key=lambda item: item.s):
                        position = waypoint.transform.location
                        point = (float(position.x), float(-position.y),
                                 float(position.z))
                        if not points or point != points[-1]:
                            points.append(point)
                    if len(points) >= 2:
                        polylines.append(points)
            cache[key] = polylines
        point = (float(location.x), float(-location.y), float(location.z))
        return min((self._point_to_polyline_distance_3d(point, line)
                    for line in cache[key]), default=math.inf)

    def _nearest_carla_connector_path(self, connector_id, location):
        """Rank only the mapped physical paths of an already eligible connector."""
        candidates = []
        for record in self._connector_path_records(connector_id):
            path_id = record["connectorPathId"]
            distance = self._carla_connector_path_distance(
                connector_id, path_id, location
            )
            if math.isfinite(distance):
                candidates.append((distance, path_id))
        if not candidates:
            return None, math.inf
        distance, path_id = min(candidates)
        return path_id, distance

    def _nearest_mapped_connector_path_projection(
        self, position, mapped_lanes, connector_id=None
    ):
        """Return the closest mapped connector path and projection metadata."""
        best = (None, None, math.inf, False, None)
        seen = set()
        for mapped_lane in mapped_lanes:
            resolved = self._connector_path_for_mapped_lane(
                mapped_lane, connector_id=connector_id
            )
            if resolved is None:
                continue
            candidate_connector, path_id, record = resolved
            key = (candidate_connector, path_id)
            if key in seen:
                continue
            seen.add(key)
            distance, at_terminal = self._point_to_polyline_projection(
                position[:2],
                self._query_connector_path_centerline(
                    candidate_connector, path_id
                ),
            )
            if distance < best[2]:
                best = (
                    candidate_connector,
                    path_id,
                    distance,
                    at_terminal,
                    record,
                )
        return best

    def _connector_path_id_for_pose(self, connector_id, location, *records):
        """Resolve the connector selector for one authoritative CARLA pose."""
        connector_id = str(connector_id)
        waypoint, projection_error = self._projected_carla_driving_waypoint(
            location
        )
        if (
            waypoint is not None
            and projection_error <= CARLA_OWNED_PROJECTION_TOLERANCE
        ):
            mapped_lanes = self._mapped_lanes_for_carla_waypoint(waypoint)
            candidates = {
                resolved[1]
                for mapped_lane in mapped_lanes
                if (
                    resolved := self._connector_path_for_mapped_lane(
                        mapped_lane, connector_id=connector_id
                    )
                ) is not None
            }
            if len(candidates) == 1:
                return candidates.pop()
            if len(candidates) > 1:
                _, path_id, distance, _, _ = (
                    self._nearest_mapped_connector_path_projection(
                        (location.x, -location.y, location.z),
                        mapped_lanes,
                        connector_id=connector_id,
                    )
                )
                if path_id is not None and math.isfinite(distance):
                    return path_id

        for record in records:
            if not isinstance(record, dict):
                continue
            reported_connector = record.get("connectorId")
            reported_segment = record.get("segmentId")
            if connector_id not in {
                str(value)
                for value in (reported_connector, reported_segment)
                if value is not None
            }:
                continue
            path_id = self._normalized_connector_path_id(
                record.get("connectorPathId")
            )
            if path_id is not None:
                return path_id

        paths = self._connector_path_records(connector_id)
        if len(paths) == 1:
            return paths[0]["connectorPathId"]
        # A junction branch may be selected while the nearest global waypoint
        # still belongs to a crossing. Resolve its lane path from the same
        # physical geometry used for selection, including during publication.
        path_id, distance = self._nearest_carla_connector_path(connector_id, location)
        if path_id is not None and distance <= CARLA_OWNED_PROJECTION_TOLERANCE:
            return path_id
        return None

    @staticmethod
    def _point_to_polyline_distance_3d(point, polyline):
        """Measure a pose against path geometry with an explicit elevation."""
        if not polyline or any(len(vertex) < 3 for vertex in polyline):
            return math.inf
        points = [tuple(map(float, vertex[:3])) for vertex in polyline]
        if not all(
            math.isfinite(value) for vertex in [point, *points] for value in vertex
        ):
            return math.inf
        if len(points) == 1:
            return math.dist(point, points[0])
        best = math.inf
        for start, end in zip(points, points[1:]):
            delta = tuple(b - a for a, b in zip(start, end))
            length_squared = sum(value * value for value in delta)
            ratio = (
                sum((p - a) * d for p, a, d in zip(point, start, delta))
                / length_squared
                if length_squared else 0.0
            )
            ratio = min(1.0, max(0.0, ratio))
            closest = tuple(a + ratio * d for a, d in zip(start, delta))
            best = min(best, math.dist(point, closest))
        return best

    @staticmethod
    def _point_to_polyline_distance(point, polyline):
        return CosimSimulation._point_to_polyline_projection(point, polyline)[0]

    @staticmethod
    def _point_to_polyline_projection(point, polyline):
        """Return distance to a polyline and whether its nearest point is an end."""
        if not polyline:
            return math.inf, False

        # Remove consecutive duplicate control points so terminal detection is
        # based on the first/last usable point rather than a zero-length edge.
        points = []
        for item in polyline:
            candidate = tuple(item[:2])
            if not points or candidate != points[-1]:
                points.append(candidate)
        if len(points) == 1:
            return math.dist(point, points[0]), True

        px, py = point
        best = math.inf
        closest = None
        for start, end in zip(points, points[1:]):
            sx, sy = start
            ex, ey = end
            dx, dy = ex - sx, ey - sy
            length_squared = dx * dx + dy * dy
            if length_squared == 0:
                distance = math.hypot(px - sx, py - sy)
            else:
                ratio = ((px - sx) * dx + (py - sy) * dy) / length_squared
                ratio = min(1.0, max(0.0, ratio))
                closest_x = sx + ratio * dx
                closest_y = sy + ratio * dy
                distance = math.hypot(px - closest_x, py - closest_y)
            if distance < best:
                best = distance
                closest = (closest_x, closest_y) if length_squared else start

        endpoint_epsilon = 1e-6
        at_terminal = closest is not None and (
            math.dist(closest, points[0]) <= endpoint_epsilon
            or math.dist(closest, points[-1]) <= endpoint_epsilon
        )
        return best, at_terminal

    def _distance_to_mapped_road(self, position, mapped_lanes, target_road):
        """Measure a point against mapped lanes belonging to one METS-R road."""
        return self._nearest_mapped_lane(
            position, mapped_lanes, target_road
        )[1]

    def _nearest_mapped_lane(self, position, mapped_lanes, target_road):
        """Return the closest mapped METS-R lane index and its distance."""
        lane_index, distance, _ = self._nearest_mapped_lane_projection(
            position, mapped_lanes, target_road
        )
        return lane_index, distance

    def _nearest_mapped_lane_projection(
        self, position, mapped_lanes, target_road
    ):
        """Return the closest mapped lane, distance, and terminal status."""
        best_lane_index = None
        best = math.inf
        best_at_terminal = False
        target_road = str(target_road)
        for road_lane in mapped_lanes:
            road, lane_index = self._mapped_lane_query(road_lane)
            if road != target_road:
                continue
            if lane_index is None:
                continue
            distance, at_terminal = self._point_to_polyline_projection(
                position[:2],
                self._query_mapped_centerline(road, lane_index),
            )
            if distance < best:
                best = distance
                best_lane_index = lane_index
                best_at_terminal = at_terminal
        return best_lane_index, best, best_at_terminal



    def classify_bubble_roads(self, bubble_roads : list[Road]) -> tuple[list[str], list[str]]:
        """
        Docstring for update_carla_roads

        :param bubble_regions: List of objects with a designated "bubble region" constituting the CoSim region
        :type obj: List[Object] or None

        Collects all roads which are intersecting the bubble region
            (1) Default region is defined by the ego the region can be updated by passing objects with their corresponding regions
        """
        road_ids = []
        bubble_road_ids = []
        for road in bubble_roads:
            r_id = str(road.id)
            road_ids.append(r_id)
            bubble_road_ids += self.map_scenic_to_metsr_road(road)
        bubble_roads += self.network_helper.repair_mapping(road_ids, bubble_road_ids)
        self.bubble_roads = bubble_roads # List of roads contained in the CoSim region
        self.active_bubble_metsr_roads = set(bubble_road_ids)

        self.frozen_roads = list(self.carla_control_roads.keys())
        # Collect roads into new and old for freeze/unfreezing
        new_roads = [id for id in bubble_road_ids if id not in self.frozen_roads]
        old_roads = [id for id in self.frozen_roads if id not in bubble_road_ids]
        return new_roads, old_roads



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
        destination = self._vehicle_destination_road(obj, vehicle_state)
        start = carla.Location(
            float(vehicle_state["x"]), -float(vehicle_state["y"]),
            float(vehicle_state.get("z") or 0.0),
        )
        path = self.generate_carla_destination_trajectory(
            destination, obj, target_start=start, vehicle_state=vehicle_state
        )
        self.tm.set_path(obj.carlaActor, path)
        self.initiate_autopilot(obj)
        obj.trajectory = path
        obj._control = None
        if not hasattr(self, "_carla_destination_plans"):
            self._carla_destination_plans = {}
        self._carla_destination_plans[obj] = (destination, path)
        return True

    def update_bubble_objects(
        self, bubble_roads, bubble_intersections, *, vehicle_data=None,
    ) -> None:
        """Promote native vehicles in controlled segments before CARLA advances."""
        all_veh_data = (vehicle_data if vehicle_data is not None
                        else self._collect_metsr_vehicle_data(self.objects[1:]))
        self.bubble_roads_by_id = [road.id for road in self.bubble_roads]
        for obj in self.objects[1:]:
            if obj in self.completed_route:
                continue
            veh_data = all_veh_data[obj]
            metsr_road = _metsr_vehicle_road(veh_data)
            obj.spawn_guard = max(0, obj.spawn_guard - self.sim_ticks_per_carla)
            if metsr_road is not None:
                self.road_pop_density[metsr_road] += 1
            if obj.carla_actor_flag:
                continue
            if metsr_road is None:
                # A new trip has no segment while it waits for native admission.
                if veh_data.get("queuedRoadId") is None:
                    self.completed_route[obj] = True
                    obj.finished_route = self.count
                continue
            if not self._vehicle_record_in_cosim_region(veh_data):
                continue
            if not self.createObjectInCarla(obj, update_velocity=True, native_state=veh_data):
                # Already-present native vehicles need the same bounded spawn
                # retry as vehicles admitted through enterRoadFromQueue.
                self._ensure_queue_admission_state()
                self.admitted_queue_vehicles.setdefault(self.pv_id_map[obj], {
                    "roadId": metsr_road, "admittedAt": self.count,
                })
                continue
            self._initialize_carla_authoritative_segment(obj, veh_data, veh_data)
            self._initialize_promoted_carla_autopilot(obj, veh_data)
            # Commit promotion only after velocity and controller initialization.
            self.carla_actors.append(obj)
            if obj in self.metsr_actors:
                self.metsr_actors.remove(obj)
            vehicle_id = self.pv_id_map.get(obj)
            getattr(self, "admitted_queue_vehicles", {}).pop(vehicle_id, None)

    def _ensure_connector_ownership_state(self):
        """Initialize connector ownership for lightweight/legacy instances."""
        if not hasattr(self, "carla_control_roads"):
            self.carla_control_roads = {}
        if not hasattr(self, "carla_control_segments"):
            self.carla_control_segments = set(self.carla_control_roads)
        if not hasattr(self, "metsr_internal_edge_to_connector"):
            self.metsr_internal_edge_to_connector = {}
        if not hasattr(self, "metsr_connector_records"):
            self.metsr_connector_records = {}
        if not hasattr(self, "metsr_road_connectors"):
            self.metsr_road_connectors = {}

    def _rebuild_controlled_physical_segments(self):
        """Rebuild the physical-segment set without inserting connector IDs."""
        self._ensure_connector_ownership_state()
        referenced_connectors = {
            connector_id
            for connector_ids in self.metsr_road_connectors.values()
            for connector_id in connector_ids
        }
        self.metsr_connector_records = {
            connector_id: record
            for connector_id, record in self.metsr_connector_records.items()
            if connector_id in referenced_connectors
        }
        edge_mapping = {}
        for connector_id, record in self.metsr_connector_records.items():
            for internal_edge_id in record.get("internalEdgeIds", ()):
                internal_edge_id = str(internal_edge_id)
                previous = edge_mapping.get(internal_edge_id)
                if previous is not None and previous != connector_id:
                    raise METSRControlError(
                        f"Internal edge {internal_edge_id!r} belongs to both "
                        f"{previous!r} and {connector_id!r}"
                    )
                edge_mapping[internal_edge_id] = connector_id
        self.metsr_internal_edge_to_connector = edge_mapping
        self.carla_control_segments = set(map(str, self.carla_control_roads))
        self.carla_control_segments.update(edge_mapping)

    def _expected_incident_internal_edges(self, road_ids):
        """Return XML-proven connector edges touching each requested road."""
        requested = {str(road_id) for road_id in road_ids}
        expected = {road_id: set() for road_id in requested}
        edge_connections = getattr(
            self, "metsr_internal_edge_connections", None
        )
        if edge_connections is None:
            edge_connections = {}

        # Compatibility for simulation fixtures and cached maps created before
        # the explicit edge index was added. This parses a physical SUMO lane
        # ID only; connector IDs remain opaque.
        if not edge_connections:
            edge_connections = {}
            for internal_lane, pairs in getattr(
                self, "metsr_internal_lane_connections", {}
            ).items():
                internal_edge, separator, _ = str(internal_lane).rpartition("_")
                if separator:
                    edge_connections.setdefault(internal_edge, []).extend(pairs)

        for internal_edge_id, pairs in edge_connections.items():
            for source_lane, target_lane in pairs:
                source_road_id, _ = self._mapped_lane_query(source_lane)
                target_road_id, _ = self._mapped_lane_query(target_lane)
                for incident_road_id in {
                    str(source_road_id),
                    str(target_road_id),
                }:
                    if incident_road_id in requested:
                        expected[incident_road_id].add(str(internal_edge_id))
        return expected

    @staticmethod
    def _connector_topology(record):
        return (
            record.get("connectorId"),
            record.get("sourceRoadId"),
            record.get("targetRoadId"),
            tuple(record.get("internalEdgeIds", ())),
        )

    def _normalize_incident_connector(
        self, connector, road_id, response, operation
    ):
        """Validate one opaque server connector record touching road_id."""
        if not isinstance(connector, dict):
            raise METSRControlError(
                f"{operation} returned an invalid connector for road {road_id}",
                response=response,
                record=connector,
            )
        connector_id = connector.get("connectorId")
        source_road_id = connector.get("sourceRoadId")
        target_road_id = connector.get("targetRoadId")
        internal_edge_ids = connector.get("internalEdgeIds")
        if connector_id is None or not str(connector_id):
            raise METSRControlError(
                f"{operation} omitted connectorId for road {road_id}",
                response=response,
                record=connector,
            )
        if source_road_id is None or target_road_id is None:
            raise METSRControlError(
                f"{operation} omitted connector endpoints for {connector_id!r}",
                response=response,
                record=connector,
            )
        connector_id = str(connector_id)
        source_road_id = str(source_road_id)
        target_road_id = str(target_road_id)
        if str(road_id) not in {source_road_id, target_road_id}:
            raise METSRControlError(
                f"{operation} returned connector {connector_id!r}, which does "
                f"not touch road {road_id!r}",
                response=response,
                record=connector,
            )
        if not isinstance(internal_edge_ids, (list, tuple)):
            raise METSRControlError(
                f"{operation} omitted internalEdgeIds for {connector_id!r}",
                response=response,
                record=connector,
            )
        normalized = dict(connector)
        normalized["connectorId"] = connector_id
        normalized["sourceRoadId"] = source_road_id
        normalized["targetRoadId"] = target_road_id
        normalized["internalEdgeIds"] = list(
            dict.fromkeys(
                str(edge_id)
                for edge_id in internal_edge_ids
                if edge_id is not None and str(edge_id)
            )
        )
        return normalized

    def _parse_takeover_ownership(self, response, requested):
        """Validate setCoSimRoad records and return uncommitted ownership."""
        requested = tuple(sorted({str(road_id) for road_id in requested}))
        expected_edges = self._expected_incident_internal_edges(requested)
        records_by_road = {}
        for record in response.get("data", ()):
            if not isinstance(record, dict) or record.get("status") != "ok":
                continue
            road_id = record.get("roadId")
            if road_id is None:
                raise METSRControlError(
                    "setCoSimRoad returned a successful record without roadId",
                    response=response,
                    record=record,
                )
            road_id = str(road_id)
            if road_id not in requested or road_id in records_by_road:
                raise METSRControlError(
                    f"setCoSimRoad returned an unexpected or duplicate road {road_id!r}",
                    response=response,
                    record=record,
                )
            records_by_road[road_id] = record
        missing_roads = set(requested) - set(records_by_road)
        if missing_roads:
            raise METSRControlError(
                "setCoSimRoad omitted successful records for "
                f"{sorted(missing_roads)}",
                response=response,
            )

        road_connectors = {road_id: set() for road_id in requested}
        connector_records = {}
        internal_edge_to_connector = {}
        returned_edges = {road_id: set() for road_id in requested}
        for road_id, road_record in records_by_road.items():
            connectors = road_record.get("connectors")
            if connectors is None:
                connectors = ()
            if not isinstance(connectors, (list, tuple)):
                raise METSRControlError(
                    f"setCoSimRoad returned invalid connectors for road {road_id}",
                    response=response,
                    record=road_record,
                )
            for connector in connectors:
                normalized = self._normalize_incident_connector(
                    connector, road_id, response, "setCoSimRoad"
                )
                connector_id = normalized["connectorId"]
                previous = connector_records.get(connector_id)
                if (
                    previous is not None
                    and self._connector_topology(previous)
                    != self._connector_topology(normalized)
                ):
                    raise METSRControlError(
                        f"setCoSimRoad returned conflicting records for {connector_id!r}",
                        response=response,
                        record=connector,
                    )
                connector_records[connector_id] = normalized
                road_connectors[road_id].add(connector_id)
                for internal_edge_id in normalized["internalEdgeIds"]:
                    previous_owner = internal_edge_to_connector.get(internal_edge_id)
                    if previous_owner is not None and previous_owner != connector_id:
                        raise METSRControlError(
                            f"Internal edge {internal_edge_id!r} belongs to both "
                            f"{previous_owner!r} and {connector_id!r}",
                            response=response,
                            record=connector,
                        )
                    internal_edge_to_connector[internal_edge_id] = connector_id
                    returned_edges[road_id].add(internal_edge_id)

            announced_ids = road_record.get("connectorIds")
            if not isinstance(announced_ids, (list, tuple)) or {
                str(connector_id) for connector_id in announced_ids
            } != road_connectors[road_id]:
                raise METSRControlError(
                    f"setCoSimRoad connectorIds/connectors disagree for road {road_id}",
                    response=response,
                    record=road_record,
                )

        for road_id, expected in expected_edges.items():
            missing = expected - returned_edges[road_id]
            if missing:
                raise METSRControlError(
                    f"setCoSimRoad omitted expected incident internal edges for "
                    f"road {road_id}: {sorted(missing)}",
                    response=response,
                    record=records_by_road[road_id],
                )

        self._ensure_connector_ownership_state()
        for connector_id, normalized in connector_records.items():
            previous = self.metsr_connector_records.get(connector_id)
            if (
                previous is not None
                and self._connector_topology(previous)
                != self._connector_topology(normalized)
            ):
                raise METSRControlError(
                    f"Connector {connector_id!r} changed topology during takeover",
                    response=response,
                    record=normalized,
                )
        for edge_id, connector_id in internal_edge_to_connector.items():
            previous_owner = self.metsr_internal_edge_to_connector.get(edge_id)
            if previous_owner is not None and previous_owner != connector_id:
                raise METSRControlError(
                    f"Internal edge {edge_id!r} changed connector ownership from "
                    f"{previous_owner!r} to {connector_id!r}",
                    response=response,
                )
        return road_connectors, connector_records, internal_edge_to_connector

    def _commit_takeover_ownership(
        self, road_ids, road_connectors, connector_records, edge_mapping
    ):
        self._ensure_connector_ownership_state()
        for road_id in road_ids:
            road_id = str(road_id)
            self.carla_control_roads[road_id] = True
            self.metsr_road_connectors[road_id] = set(
                road_connectors.get(road_id, ())
            )
        self.metsr_connector_records.update(connector_records)
        self.metsr_internal_edge_to_connector.update(edge_mapping)
        self._rebuild_controlled_physical_segments()

    def freeze_roads(self, keys: list[str]) -> None:
        """
        Docstring for freeze_roads

        :param keys: RoadIDs for METSR indexed roads
        :type keys: list[str]

        Query Metsr to freeze simulation and control of given lanes

        Scenic sends one request, validates that every requested road succeeded
        before committing local ownership, and propagates any failure without
        retry or compensating release.
        """
        self._ensure_connector_ownership_state()
        keys = {str(key) for key in keys}
        for key in keys:
            assert key not in self.carla_control_roads, (
                "Attempted to freeze already frozen lane"
            )
        requested = sorted(
            key
            for key in keys
            if key in self.network_helper.metsr_represented_roads
        )
        if not requested:
            return
        print(f"Freezing roads: {requested}")
        response = self.metsr_client.set_cosim_road(requested)
        _require_all_success_control_response(response, "setCoSimRoad")
        ownership = self._parse_takeover_ownership(response, requested)
        self._commit_takeover_ownership(requested, *ownership)


    def release_roads(self, keys: list[str], *, allow_defer=False) -> None:
        """Return controlled roads to METS-R and update shared connectors."""
        keys = sorted({str(key) for key in keys})
        if not keys:
            return
        self._ensure_connector_ownership_state()
        represented = set(map(str, self.network_helper.metsr_represented_roads))
        for key in keys:
            assert key in self.carla_control_roads, (
                "Attempted to release non frozen lane"
            )
            if key not in represented:
                raise METSRControlError(
                    f"Cannot release unrepresented METS-R road {key!r}"
                )

            # Commit local release only after native placement succeeds. The
            # server can reject a release when vehicle placement is blocked.
            response = self.metsr_client.release_cosim_road(key)
            if allow_defer and isinstance(response, dict):
                records = response.get("data")
                if (response.get("messageType") == "releaseCoSimRoad"
                        and response.get("status") in ("ok", "partial")
                        and isinstance(records, (list, tuple)) and len(records) == 1
                        and isinstance(records[0], dict)
                        and str(records[0].get("roadId")) == key
                        and records[0].get("status") == "error"
                        and records[0].get("errorCode") == "RELEASE_BLOCKED"
                        and records[0].get("retryable") is True):
                    # Keep the road owned and reconsider it on the next step.
                    continue
            _require_all_success_control_response(response, "releaseCoSimRoad")
            records = [
                record
                for record in response.get("data", ())
                if isinstance(record, dict)
                and record.get("status") == "ok"
                and str(record.get("roadId")) == key
            ]
            if len(records) != 1:
                raise METSRControlError(
                    "releaseCoSimRoad did not return exactly one successful "
                    f"record for {key}",
                    response=response,
                )
            record = records[0]
            released_connector_data = record.get("releasedConnectors")
            released_id_data = record.get("releasedConnectorIds")
            remaining_connector_data = record.get("connectors")
            remaining_id_data = record.get("connectorIds")
            if not all(
                isinstance(value, (list, tuple))
                for value in (
                    released_connector_data,
                    released_id_data,
                    remaining_connector_data,
                    remaining_id_data,
                )
            ):
                raise METSRControlError(
                    f"releaseCoSimRoad omitted connector metadata for road {key}",
                    response=response,
                    record=record,
                )

            released_records = {}
            for connector in released_connector_data:
                normalized = self._normalize_incident_connector(
                    connector, key, response, "releaseCoSimRoad"
                )
                connector_id = normalized["connectorId"]
                if connector_id in released_records:
                    raise METSRControlError(
                        f"releaseCoSimRoad duplicated connector {connector_id!r}",
                        response=response,
                        record=connector,
                    )
                released_records[connector_id] = normalized
            released_ids = {str(connector_id) for connector_id in released_id_data}
            if set(released_records) != released_ids:
                raise METSRControlError(
                    "releaseCoSimRoad released connector IDs and records "
                    f"disagree for road {key}",
                    response=response,
                    record=record,
                )

            remaining_records = {}
            for connector in remaining_connector_data:
                normalized = self._normalize_incident_connector(
                    connector, key, response, "releaseCoSimRoad"
                )
                connector_id = normalized["connectorId"]
                if connector_id in remaining_records:
                    raise METSRControlError(
                        f"releaseCoSimRoad duplicated connector {connector_id!r}",
                        response=response,
                        record=connector,
                    )
                remaining_records[connector_id] = normalized
            remaining_ids = {str(connector_id) for connector_id in remaining_id_data}
            if set(remaining_records) != remaining_ids:
                raise METSRControlError(
                    "releaseCoSimRoad remaining connector IDs and records "
                    f"disagree for road {key}",
                    response=response,
                    record=record,
                )

            owned_ids = set(self.metsr_road_connectors.get(key, ()))
            if (
                released_ids & remaining_ids
                or released_ids | remaining_ids != owned_ids
            ):
                raise METSRControlError(
                    f"releaseCoSimRoad connector partition for {key} does not "
                    f"match Scenic ownership: owned={sorted(owned_ids)}, "
                    f"released={sorted(released_ids)}, "
                    f"remaining={sorted(remaining_ids)}",
                    response=response,
                    record=record,
                )

            active_after_release = set(self.carla_control_roads) - {key}
            for connector_id, connector in {
                **released_records,
                **remaining_records,
            }.items():
                previous = self.metsr_connector_records.get(connector_id)
                if (
                    previous is None
                    or self._connector_topology(previous)
                    != self._connector_topology(connector)
                ):
                    raise METSRControlError(
                        f"releaseCoSimRoad changed unknown connector topology "
                        f"for {connector_id!r}",
                        response=response,
                        record=connector,
                    )

                endpoints = {
                    connector["sourceRoadId"],
                    connector["targetRoadId"],
                }
                other_references = {
                    road_id
                    for road_id, connector_ids in self.metsr_road_connectors.items()
                    if road_id != key and connector_id in connector_ids
                }
                should_remain = bool(endpoints & active_after_release)
                if connector_id in remaining_ids:
                    if not should_remain or not other_references:
                        raise METSRControlError(
                            f"releaseCoSimRoad retained connector {connector_id!r} "
                            "without another controlled endpoint",
                            response=response,
                            record=connector,
                        )
                elif should_remain or other_references:
                    raise METSRControlError(
                        f"releaseCoSimRoad released connector {connector_id!r} "
                        "which is still referenced by another controlled endpoint",
                        response=response,
                        record=connector,
                    )

            # Commit only after the complete response is validated.
            del self.carla_control_roads[key]
            self.metsr_road_connectors.pop(key, None)
            self.metsr_connector_records.update(remaining_records)
            for connector_id in released_ids:
                self.metsr_connector_records.pop(connector_id, None)
            self._rebuild_controlled_physical_segments()

    def destroy_carla_obj(self,obj) -> None:
        """
        Docstring for destroy_carla_obj

        Destroys obj from CARLA simulation

        :param obj: Carla object to be destroyed
        """
        self._clear_carla_authoritative_segment(obj)
        if self._is_pcla_controlled_ego(obj):
            raise SimulationCreationError(
                "A PCLA-controlled ego cannot be destroyed during ordinary "
                "co-simulation synchronization; it is cleaned up once during "
                "final simulation teardown"
            )
        if obj.carlaActor is not None:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.set_autopilot(False, self.tm.get_port())
            if isinstance(obj.carlaActor, carla.Walker):
                obj.cralaController.stop()
                obj.carlaController.destroy()
            self._destroy_attached_carla_sensors(obj.carlaActor)
            obj.carlaActor.destroy()
            obj.carlaActor = None # Set this to None to prevent reaccess of a previously deleted vehicle?

    def remove_bubble_object(self,obj, destroy=True) -> None:
        """
        Docstring for remove_bubble_object

        :param obj: object to be deleted
        :type obj: Car
        """
        if self._is_pcla_controlled_ego(obj):
            raise SimulationCreationError(
                "A PCLA-controlled ego cannot be demoted from CARLA into the "
                "METS-R background simulation"
            )
        self._clear_carla_authoritative_segment(obj)
        # Add a check metsr road == matches the cars current
        if obj.autopilot_action and obj.active_autopilot:
            obj.active_autopilot = not(_utils.disable_carla_autopilot(obj, self.tm))
            obj.trajectory = None
        if destroy:
            self.destroy_carla_obj(obj)
        obj.carla_actor_flag = False
        self.carla_actors.remove(obj)
        self.metsr_actors.append(obj)
        self.road_entry_holds.pop(obj, None)
        getattr(self, "road_entry_hold_diagnostics", {}).pop(
            obj, None
        )
        self.pending_route_refreshes.pop(obj, None)
        getattr(self, "pending_lane_reconciliation_verifications", {}).pop(
            obj, None
        )
        getattr(self, "same_road_departure_lane_verifications", {}).pop(
            obj, None
        )
        getattr(self, "_pending_pcla_pose_resets", set()).discard(obj)


    @staticmethod
    def _carla_actor_is_alive(actor) -> bool:
        """Return whether an actor can still be safely used during teardown."""
        try:
            return bool(actor.is_alive)
        except AttributeError:
            # Lightweight actor implementations used by compatible CARLA clients
            # may not expose ``is_alive``; preserve the traditional behavior.
            return True
        except RuntimeError:
            # CARLA raises here when the server-side actor has already gone away.
            return False

    def _carla_actor_is_registered(self, actor) -> bool:
        """Return whether CARLA still has the server-side actor.

        Actor.is_alive is cached by some CARLA releases and can remain true
        after another cleanup path has removed the actor. Query the world when
        that API is available so teardown does not issue a second destroy request
        against a stale Python proxy.
        """
        if not self._carla_actor_is_alive(actor):
            return False

        actor_id = getattr(actor, "id", None)
        world = getattr(self, "carla_world", None)
        get_actor = getattr(world, "get_actor", None)
        if actor_id is None or not callable(get_actor):
            return True
        try:
            return get_actor(actor_id) is not None
        except RuntimeError:
            # The CARLA connection or actor can disappear while teardown is in
            # progress. A further destroy request cannot improve that state.
            return False

    @staticmethod
    def _remove_scenic_actor_from_pcla_cleanup_pool(pcla, scenic_actor) -> None:
        """Keep PCLA's global provider from destroying a Scenic-owned actor.

        Unmodified PCLA calls CarlaDataProvider.cleanup(), which destroys every
        actor in its private actor pool. PCLA normally registers the ego only in
        its state maps, but some agents also place it in that pool. The provider
        is private to PCLA and clears the pool during cleanup, so remove only the
        matching Scenic actor before invoking PCLA's cleanup routine.
        """
        if scenic_actor is None:
            return

        cleanup = getattr(pcla, "cleanup", None)
        function = getattr(cleanup, "__func__", cleanup)
        namespace = getattr(function, "__globals__", {})
        provider = namespace.get("CarlaDataProvider")
        actor_pool = getattr(provider, "_carla_actor_pool", None)
        if not hasattr(actor_pool, "items") or not hasattr(actor_pool, "pop"):
            return

        scenic_id = getattr(scenic_actor, "id", None)
        matching_keys = []
        for key, pooled_actor in tuple(actor_pool.items()):
            same_actor = pooled_actor is scenic_actor
            if not same_actor and scenic_id is not None and pooled_actor is not None:
                try:
                    same_actor = pooled_actor.id == scenic_id
                except (AttributeError, RuntimeError):
                    same_actor = False
            if same_actor:
                matching_keys.append(key)
        for key in matching_keys:
            actor_pool.pop(key, None)

    @staticmethod
    def _report_cleanup_failure(action, error) -> None:
        """Report a teardown failure without replacing the simulation failure."""
        message = f"Co-simulation cleanup failed while {action}: {error}"
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        except Exception:
            # Warning filters are allowed to promote warnings to exceptions. Cleanup
            # must remain best-effort even in that configuration.
            print(message)

    def add_pre_actor_teardown_callback(self, callback) -> None:
        """Register cleanup which must run before Scenic destroys CARLA actors."""
        if not callable(callback):
            raise TypeError("pre-actor teardown callback must be callable")
        callbacks = getattr(self, "_pre_actor_teardown_callbacks", None)
        if callbacks is None:
            callbacks = self._pre_actor_teardown_callbacks = []
        callbacks.append(callback)

    def _run_pre_actor_teardown_callbacks(self) -> None:
        callbacks = getattr(self, "_pre_actor_teardown_callbacks", ())
        while callbacks:
            callback = callbacks.pop()
            try:
                callback()
            except Exception as error:
                self._report_cleanup_failure(
                    "running a pre-actor teardown callback", error
                )

    def _destroy_attached_carla_sensors(self, actor) -> None:
        """Stop and destroy every live sensor attached to one Scenic actor."""
        actor_id = getattr(actor, "id", None)
        world = getattr(self, "carla_world", None)
        get_actors = getattr(world, "get_actors", None)
        if actor_id is None or not callable(get_actors):
            return
        try:
            actors = get_actors()
            sensors = actors.filter("sensor.*") if hasattr(actors, "filter") else ()
        except Exception as error:
            self._report_cleanup_failure("enumerating attached CARLA sensors", error)
            return

        for sensor in tuple(sensors):
            try:
                parent = getattr(sensor, "parent", None)
                if parent is None or getattr(parent, "id", None) != actor_id:
                    continue
                listening = getattr(sensor, "is_listening", False)
                listening = listening() if callable(listening) else bool(listening)
                if listening:
                    sensor.stop()
                if self._carla_actor_is_registered(sensor):
                    sensor.destroy()
            except Exception as error:
                sensor_id = getattr(sensor, "id", "unknown")
                self._report_cleanup_failure(
                    f"destroying CARLA sensor {sensor_id} attached to actor "
                    f"{actor_id}",
                    error,
                )

    @staticmethod
    def _cleanup_pcla(pcla, scenic_actor=None) -> None:
        """Clean up PCLA while retaining Scenic ownership when PCLA supports it."""
        CosimSimulation._remove_scenic_actor_from_pcla_cleanup_pool(
            pcla, scenic_actor
        )
        cleanup = pcla.cleanup
        try:
            parameters = inspect.signature(cleanup).parameters.values()
        except (TypeError, ValueError):
            parameters = ()

        supports_non_owning_cleanup = any(
            parameter.name == "destroy_vehicle"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if supports_non_owning_cleanup:
            cleanup(destroy_vehicle=False)
        else:
            # Unmodified legacy PCLA has no destroy_vehicle argument. It needs the
            # vehicle ID while deleting attached sensors, then destroys
            # ``self.vehicle`` itself. Substitute a facade which preserves sensor
            # matching but reports non-live and makes destroy a no-op.
            pcla_vehicle = getattr(pcla, "vehicle", None)
            same_vehicle = pcla_vehicle is scenic_actor
            if (
                not same_vehicle
                and pcla_vehicle is not None
                and scenic_actor is not None
            ):
                try:
                    same_vehicle = pcla_vehicle.id == scenic_actor.id
                except (AttributeError, RuntimeError):
                    same_vehicle = False

            if not same_vehicle:
                cleanup()
                return

            facade = _ScenicOwnedPCLAVehicle(scenic_actor)
            pcla.vehicle = facade
            try:
                cleanup()
            finally:
                # A failed legacy cleanup must not retain the facade or regain an
                # owning reference to Scenic's actor.
                pcla.vehicle = None

    def _destroy_carla_object_during_teardown(self, obj) -> None:
        """Best-effort cleanup for one CARLA object, safe to call only once."""
        actor = getattr(obj, "carlaActor", None)
        if actor is None:
            return

        if isinstance(actor, carla.Vehicle) and self._carla_actor_is_registered(
            actor
        ):
            try:
                actor.set_autopilot(False, self.tm.get_port())
            except Exception as error:
                self._report_cleanup_failure("disabling CARLA autopilot", error)

        if isinstance(actor, carla.Walker) and self._carla_actor_is_registered(
            actor
        ):
            controller = getattr(obj, "carlaController", None)
            if controller is not None:
                try:
                    controller.stop()
                    if self._carla_actor_is_registered(controller):
                        controller.destroy()
                except Exception as error:
                    self._report_cleanup_failure(
                        "destroying a CARLA walker controller", error
                    )

        pcla = getattr(obj, "pcla", None)
        pcla_cleaned_sensors = False
        if pcla is not None:
            try:
                self._cleanup_pcla(pcla, actor)
                pcla_cleaned_sensors = True
            except Exception as error:
                self._report_cleanup_failure("cleaning up PCLA", error)

        # PCLA already destroys its sensors through their original handles.
        # CARLA's last snapshot can still enumerate those IDs until the next
        # tick, so sweeping them again would issue duplicate destroy requests.
        if not pcla_cleaned_sensors:
            self._destroy_attached_carla_sensors(actor)

        try:
            if self._carla_actor_is_registered(actor):
                actor.destroy()
        except Exception as error:
            self._report_cleanup_failure("destroying a CARLA actor", error)
        finally:
            # Make teardown idempotent and prevent later code from reusing a stale
            # Python handle even when the actor was destroyed by legacy PCLA.
            obj.carlaActor = None
            self._clear_carla_authoritative_segment(obj)

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
                self._log_trip_times()
            except Exception as error:
                self._report_cleanup_failure("logging trip times", error)
        print("=" * 25)

        # Log first. Reset an aborted run (or a blocked release) using the
        # existing server lifecycle, without counting interrupted trips as arrivals.
        try:
            if active_error:
                self.metsr_client.reset()
            else:
                try:
                    self.release_roads(list(getattr(self, "carla_control_roads", ())))
                except Exception as error:
                    self._report_cleanup_failure("releasing METS-R roads; resetting the run", error)
                    self.metsr_client.reset()
        except Exception as error:
            cleanup_error = error
            self._report_cleanup_failure("resetting METS-R during cleanup", error)

        # METSR destroy
        try:
            if getattr(self.metsr_client, "verbose", False):
                print("Client Messages Log:")
                print("[")
                for call in self.metsr_client._messagesLog:
                    print(f"    {call},")
                print("]")
        except Exception as error:
            self._report_cleanup_failure("printing the METS-R message log", error)

        # Each visualization owner closes its sensors before the parent vehicle.
        if getattr(self, "render", False) and getattr(self, "cameraManager", None):
            try:
                self.cameraManager.destroy_sensor()
            except Exception as error:
                self._report_cleanup_failure("destroying the CARLA camera", error)
        self._run_pre_actor_teardown_callbacks()
        for key in tuple(getattr(self, "boundary_actors", ())):
            try:
                self._remove_boundary_actor(key)
            except Exception as error:
                self._report_cleanup_failure("destroying a boundary vehicle", error)
        # "CARLA destroy"
        cleanup_objects = list(getattr(self, "carla_actors", ()))
        # Include actors whose initialization failed before promotion completed.
        cleanup_objects.extend(
            obj for obj in getattr(self, "objects", ())
            if getattr(obj, "carlaActor", None) is not None
            and not any(obj is existing for existing in cleanup_objects)
        )
        for obj in cleanup_objects:
            try:
                self._destroy_carla_object_during_teardown(obj)
            except Exception as error:
                self._report_cleanup_failure("cleaning up a CARLA object", error)
        getattr(self, "_carla_authoritative_segments", {}).clear()
        getattr(self, "_carla_destination_plans", {}).clear()

        try:
            self.carla_client.stop_recorder()
        except Exception as error:
            self._report_cleanup_failure("stopping the CARLA recorder", error)

        try:
            super().destroy()
        except Exception as error:
            self._report_cleanup_failure("finalizing the Scenic simulation", error)

        if cleanup_error is not None and not active_error:
            raise cleanup_error

    def map_scenic_to_metsr_road(self, road: Road) -> list[str]:
        """Maps Scenic road to equvialent METSR roads, 1->M mapping"""
        return self.network_helper.map_scenic_to_metsr_road(road)

    def map_scenic_to_metsr_lanes(self, lane: Lane) -> set[str]:
        """Map Scenic lane to equivalent METSR road"""
        return self.network_helper.map_scenic_to_metsr_lanes(lane)

    def generate_metsr_trajectory(self, trajectory: list[Lane]) -> list[str] | None:
        """Convert a scenic specified trajectory to an equivalent metsr route"""
        return self.network_helper.generate_metsr_trajctory(trajectory)

    def _nearest_road(self, obj: Object, allow_offroad: bool = True, radius_size: int = 30) -> tuple[Road, str]:
        """Collect the nearest road to obj location"""
        return self.network_helper._nearest_road(obj, allow_offroad, radius_size)

    def _nearest_lane(self,obj : Object, allow_offlane : bool = True, radius_size : int = 50, allow_intersection_links : bool = True) -> Lane:
        """Collect the nearest lane to obj location"""
        return self.network_helper._nearest_lane(obj, allow_offlane, radius_size, allow_intersection_links)

    def _get_intersection(self, obj: Object, road: Road ) -> Intersection | None:
        """Returns the intersection the obj is on if any"""
        return self.network_helper._get_intersection(obj, road)

    def _get_bubble_roads(self, bubble_region: CircularRegion | None = None) -> list[Road]:
        """Select nearby roads connected to the ego's current CARLA road layer."""
        if bubble_region is None:
            bubble_region = self.ego.bubble
        actor = getattr(self.ego, "carlaActor", None)
        location = (
            actor.get_location() if actor is not None
            else utils.scenicToCarlaLocation(self.ego.position)
        )
        waypoint, _ = self._projected_carla_driving_waypoint(location)
        if waypoint is None:
            raise SimulationCreationError(
                "Cannot determine the ego's CARLA road for bubble selection"
            )
        roads = self.network_helper._get_bubble_roads(
            bubble_region, anchor_road_id=waypoint.road_id
        )
        if not roads:
            raise SimulationCreationError(
                "The ego's CARLA road is absent from the Scenic bubble: "
                f"road={waypoint.road_id}, location={location}"
            )
        return roads


    def executeActions(self, allActions) -> None:
        """
        Docstring for executeActions

        Apply control updates which were accumulated while executing the actions
        Filters out actions for Carla only objects

        :param allActions: ?
        TODO: Clean this up and make it more robust -- right now it is generally optimized for the speccific scenario
        """
        held_objects = getattr(self, "road_entry_holds", {})
        carla_actions = {}
        for obj in self.agents:
            if obj in held_objects:
                # The behavior still advances and can observe the simulation, but
                # its control for this tick is intentionally discarded. Replaying
                # it after release would apply stale PCLA steering at the boundary.
                carla_actions[obj] = ()
                obj._control = None
            else:
                carla_actions[obj] = allActions[obj]
        super().executeActions(carla_actions)
        for obj in self.agents:
            if obj.carla_actor_flag: # Processing CARLA actors
                if obj in held_objects:
                    # Also clear controls accumulated by an earlier action or a
                    # custom action implementation. ``tick_carla`` reasserts the
                    # physical hold immediately before advancing the world.
                    obj._control = None
                    continue
                if not obj.autopilot_action and obj.active_autopilot: # Disable autopilot first to enable smooth transitions
                    obj.active_autopilot = not(_utils.disable_carla_autopilot(obj, self.tm))
                elif obj.autopilot_action and not obj.active_autopilot:
                    if getattr(obj, "trajectory", None) is None:
                        obj.trajectory = self.metsr_trajectory_to_carla(obj)
                    self.tm.set_path(obj.carlaActor, obj.trajectory)
                    self.initiate_autopilot(obj)
                    obj._control = None
                else:
                    ctrl = obj._control
                    if ctrl is not None:
                        obj.carlaActor.apply_control(ctrl)
                        self._notify_pcla_control_applied(obj, ctrl)
                        obj._control = None
            else:
                if not obj.autopilot_action: # Default is autopilot
                    target_acc = obj.target_acceleration if hasattr(obj, "target_accleration") else 0 # apply, if no action is taken no movement
                    _require_metsr_response(
                        self.metsr_client.control_vehicle(
                            self.getMetsrPrivateVehId(obj), target_acc, private_veh=True
                        ),
                        "controlVeh",
                    )
                if hasattr(obj, "trajectory"):
                    if obj.trajectory and not obj.autopilot_action:
                        if not self.trajectory_is_active:
                            metsr_trajectory = self.generate_metsr_trajectory(obj.trajectory)
                            if metsr_trajectory:
                                _require_metsr_response(
                                    self.metsr_client.update_vehicle_route(
                                        self.getMetsrPrivateVehId(obj),
                                        metsr_trajectory,
                                        private_veh=True,
                                    ),
                                    "updateVehicleRoute",
                                )
                                self.trajectory_is_active = True


    @staticmethod
    def _vehicle_destination_road(obj, vehicle_state):
        destination = vehicle_state.get("destinationRoadId")
        if destination is None or isinstance(destination, bool) or str(destination) == "":
            raise SimulationCreationError(
                f"METS-R vehicle query omitted destinationRoadId for {obj}"
            )
        return str(destination)

    def metsr_trajectory_to_carla(self, obj):
        """Plan with CARLA toward the vehicle query's destination road.

        METS-R's routeRoadIds lists upcoming roads, excluding the current road.
        It is not a constraint on CARLA's choice of intermediate roads or lanes.
        """
        vehicle_id = self.getMetsrPrivateVehId(obj)
        _, state = _require_single_vehicle_response(
            self.metsr_client.query_vehicle(
                vehicle_id, private_veh=True, transform_coords=True
            ),
            "vehicle", vehicle_id,
        )
        destination = self._vehicle_destination_road(obj, state)
        path = self.generate_carla_destination_trajectory(
            destination, obj, vehicle_state=state
        )
        if not hasattr(self, "_carla_destination_plans"):
            self._carla_destination_plans = {}
        self._carla_destination_plans[obj] = (destination, path)
        return path

    def _refresh_carla_destination_paths(self, vehicle_data):
        """Refresh default TM paths when METS-R changes a vehicle's destination."""
        plans = getattr(self, "_carla_destination_plans", {})
        for obj, (destination, path) in tuple(plans.items()):
            if getattr(obj, "trajectory", None) is not path:
                # A Scenic behavior supplied a different explicit trajectory.
                plans.pop(obj, None)
                continue
            if (
                not getattr(obj, "active_autopilot", False)
                or self._is_pcla_controlled_ego(obj)
                or obj not in vehicle_data
            ):
                continue
            current_destination = self._vehicle_destination_road(obj, vehicle_data[obj])
            if current_destination == destination:
                continue
            new_path = self.generate_carla_destination_trajectory(
                current_destination, obj, vehicle_state=vehicle_data[obj]
            )
            self.tm.set_path(obj.carlaActor, new_path)
            obj.trajectory = new_path
            plans[obj] = (current_destination, new_path)

    def generate_carla_destination_trajectory(
        self, destination_road_id, obj, *, target_start=None, vehicle_state=None
    ):
        """Choose a CARLA route to any reachable driving lane on the target road."""
        destination = str(destination_road_id)
        if target_start is None:
            target_start = obj.carlaActor.get_location()

        def on_destination(waypoint, location=None):
            if waypoint is None:
                return False
            key = f"{waypoint.road_id}_{waypoint.lane_id}"
            lanes = [
                lane for lane in self.scenic_to_metsr_map.get(key, ())
                if self._mapped_lane_query(lane)[1] is not None
            ]
            roads = {self._mapped_lane_query(lane)[0] for lane in lanes}
            if destination not in roads:
                return False
            if len(roads) == 1:
                return True
            # A single OpenDRIVE lane can cover several SUMO road fragments.
            # Its mapping alone must not imply arrival on a distant fragment.
            location = waypoint.transform.location if location is None else location
            point = (location.x, -location.y)
            distances = {
                road: self._distance_to_mapped_road(point, lanes, road)
                for road in roads
            }
            return distances[destination] + 1e-6 < min(
                distance for road, distance in distances.items() if road != destination
            )

        start_waypoint = self.map.get_waypoint(
            target_start, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if on_destination(start_waypoint, target_start):
            # Arrival is recorded by normal pose synchronization. Do not plan a
            # loop to an anchor behind a car which already reached its target.
            return [target_start]

        target_lanes = sorted(
            lane for lane in self._sumo_lane_to_carla_keys
            if self._mapped_lane_query(lane)[0] == destination
            and self._mapped_lane_query(lane)[1] is not None
        )
        best_path, best_length = None, math.inf
        failures = []
        for lane in target_lanes:
            anchor = self._lane_anchor_waypoint(lane, final=True)
            if anchor is None:
                continue
            try:
                trace = list(self.grp.trace_route(target_start, anchor.transform.location))
            except (nx.NetworkXException, KeyError, IndexError) as error:
                failures.append(f"{lane}: {error}")
                continue
            if not trace or not on_destination(trace[-1][0]):
                failures.append(f"{lane}: CARLA did not reach the destination road")
                continue
            # Only the destination is constrained. CARLA selects all intermediate
            # roads; no SUMO macro-route matching or lane-chain stitching runs.
            locations = [waypoint.transform.location for waypoint, _ in trace]
            previous = target_start
            length = 0.0
            for location in locations:
                length += math.dist(
                    (previous.x, previous.y, previous.z),
                    (location.x, location.y, location.z),
                )
                previous = location
            if length < best_length:
                # Traffic Manager's ImportPath selects junction branches by
                # their exit road relative to the next imported coordinate.
                # Dense points inside a junction instead make it choose the
                # exit nearest that interior point, which can be another turn.
                # Keep the CARLA-planned exit anchors and ordinary road curves;
                # use the complete trace above when comparing route lengths.
                anchors = []
                for waypoint, _ in trace:
                    if waypoint.is_junction:
                        continue
                    projected = self.map.get_waypoint(waypoint.transform.location)
                    if projected is None or projected.is_junction:
                        # A road entry may share its exact coordinates with the
                        # junction endpoint. Use the next point inside the road.
                        continue
                    anchors.append(waypoint)
                if not anchors or not on_destination(anchors[-1]):
                    failures.append(f"{lane}: no unambiguous destination road anchor")
                    continue
                best_path = [waypoint.transform.location for waypoint in anchors]
                best_length = length
        if best_path is not None:
            return best_path
        detail = failures[0] if failures else "no mapped destination lane has a CARLA anchor"
        raise SimulationCreationError(
            f"CARLA cannot route {obj} from {target_start} to METS-R destination "
            f"road {destination}: {detail}"
        )

    def _route_candidate_sequence_matches(self, candidate_sets, requested_route):
        """Check that candidate road IDs represent every requested road in order.

        A CARLA trace can begin on the immediate physical predecessor while
        METS-R already reports the target road as its first remaining macro
        road. Permit one repeated leading candidate run only when at least one
        candidate has a direct SUMO lane connection to that first requested
        road. Once the requested route begins, matching remains strict.
        """
        requested = tuple(str(road) for road in requested_route)
        if not requested or not candidate_sets:
            return False

        states = set()
        leading_run = None
        for raw_candidates in candidate_sets:
            candidates = frozenset(str(road) for road in raw_candidates)
            if not states:
                if requested[0] in candidates:
                    states = {0}
                    continue
                if leading_run is not None and candidates != leading_run:
                    return False
                if not any(
                    (candidate, requested[0]) in self.metsr_lane_connections
                    for candidate in candidates
                ):
                    return False
                leading_run = candidates
                continue

            next_states = set()
            for index in states:
                if requested[index] in candidates:
                    next_states.add(index)
                if (
                    index + 1 < len(requested)
                    and requested[index + 1] in candidates
                ):
                    next_states.add(index + 1)
            states = next_states
            if not states:
                return False
        return len(requested) - 1 in states

    @staticmethod
    def _compressed_route_candidates(candidate_sets):
        compressed = []
        for candidates in candidate_sets:
            value = tuple(sorted(str(item) for item in candidates))
            if not compressed or value != compressed[-1]:
                compressed.append(value)
        return compressed

    def _waypoint_route_candidates(self, waypoint):
        key = f"{waypoint.road_id}_{waypoint.lane_id}"
        mapped_lanes = self.scenic_to_metsr_map.get(key)
        if not mapped_lanes:
            return None, key

        candidates = set()
        for mapped_lane in mapped_lanes:
            road, _, _ = str(mapped_lane).rpartition("_")
            if road and not road.startswith(":"):
                candidates.add(road)
        return candidates, key

    def _align_carla_route_start(self, requested_route, target_start):
        """Align one stale METS-R prefix with CARLA's authoritative pose.

        A vehicle synchronized at a road boundary can already be on the direct
        arrival lane of the second reported road while METS-R still includes
        the short predecessor as the first remaining road.  That predecessor
        is physically behind the actor and cannot be used as a route anchor.
        Trim exactly one road only when the current mapped SUMO lane proves the
        requested transition has already happened.  All other mismatches fail
        closed, except the existing inverse case where CARLA is still on a
        direct predecessor of METS-R's first reported road.
        """
        requested = tuple(str(road) for road in requested_route)
        waypoint = self.map.get_waypoint(
            target_start,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None or not requested:
            return requested

        candidates, key = self._waypoint_route_candidates(waypoint)
        if not candidates or requested[0] in candidates:
            return requested

        mapped_lanes = {
            str(lane) for lane in self.scenic_to_metsr_map.get(key, ())
        }
        if len(requested) > 1 and requested[1] in candidates:
            arrival_lanes = {
                str(target_lane)
                for _, target_lane in self.metsr_lane_connections.get(
                    (requested[0], requested[1]), ()
                )
            }
            confirmed_arrivals = mapped_lanes & arrival_lanes
            if confirmed_arrivals:
                verbosePrint(
                    "CARLA pose is already on direct arrival lane(s) "
                    f"{sorted(confirmed_arrivals)} of {requested[1]}; "
                    f"dropping stale METS-R route prefix {requested[0]}"
                )
                return requested[1:]
            raise SimulationCreationError(
                f"CARLA pose on {key} maps to road {requested[1]}, but mapped "
                f"lanes {sorted(mapped_lanes)} are not direct arrivals from "
                f"reported road {requested[0]}"
            )

        for candidate in candidates:
            predecessor_sources = {
                str(source_lane)
                for source_lane, _ in self.metsr_lane_connections.get(
                    (candidate, requested[0]), ()
                )
            }
            if mapped_lanes & predecessor_sources:
                return requested

        raise SimulationCreationError(
            f"CARLA pose on {key} maps to roads {sorted(candidates)}, which "
            f"are neither the reported route start {requested[0]} nor a "
            "confirmed direct predecessor/arrival"
        )

    def _validate_carla_route_trace(self, trace, requested_route, context):
        candidate_sets = []
        unmapped_nonjunction = []
        for waypoint, _ in trace:
            candidates, key = self._waypoint_route_candidates(waypoint)
            if candidates:
                candidate_sets.append(frozenset(candidates))
            elif candidates is None and not getattr(waypoint, "is_junction", False):
                unmapped_nonjunction.append(key)

        matches = self._route_candidate_sequence_matches(
            candidate_sets, requested_route
        )
        if matches and not unmapped_nonjunction:
            return

        observed = self._compressed_route_candidates(candidate_sets)
        detail = ""
        if unmapped_nonjunction:
            detail = (
                "; unmapped non-junction CARLA lanes="
                f"{list(dict.fromkeys(unmapped_nonjunction))[:8]}"
            )
        raise SimulationCreationError(
            "CARLA route planner diverged from the METS-R macro route while "
            f"{context}: requested={list(map(str, requested_route))}, "
            f"observed candidate runs={observed}{detail}"
        )

    def _resolve_required_initial_lane(self, road_id, compact_lane_id):
        """Resolve one authoritative compact lane to its raw SUMO lane ID."""
        if compact_lane_id is None:
            return None
        if isinstance(compact_lane_id, bool):
            raise SimulationCreationError(
                "A boolean is not a valid authoritative compact lane"
            )
        try:
            compact_lane = int(compact_lane_id)
        except (TypeError, ValueError) as exc:
            raise SimulationCreationError(
                f"Invalid authoritative compact lane {compact_lane_id!r}"
            ) from exc
        if compact_lane < 0 or str(compact_lane) != str(compact_lane_id):
            raise SimulationCreationError(
                f"Invalid authoritative compact lane {compact_lane_id!r}"
            )
        road = str(road_id)
        matches = [
            str(raw_lane)
            for raw_lane, lane_index in (self.metsr_lane_indices or {}).items()
            if lane_index == compact_lane
            and self._mapped_lane_query(raw_lane)[0] == road
        ]
        if len(matches) != 1:
            raise SimulationCreationError(
                f"Expected one raw SUMO lane for authoritative compact lane "
                f"{compact_lane} of road {road}; found {matches}"
            )
        return matches[0]

    def _candidate_metsr_lane_chains(
        self, requested_route, target_start, required_initial_lane=None
    ):
        """Return low-lane-change direct SUMO lane chains for a macro route."""
        requested = tuple(str(road) for road in requested_route)
        if required_initial_lane is not None:
            required_initial_lane = str(required_initial_lane)
            if self._mapped_lane_query(required_initial_lane)[0] != requested[0]:
                return []
        if len(requested) == 1:
            if required_initial_lane is not None:
                return [[required_initial_lane]]
            waypoint = self.map.get_waypoint(
                target_start,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                return []
            key = f"{waypoint.road_id}_{waypoint.lane_id}"
            mapped_lanes = list(map(str, self.scenic_to_metsr_map.get(key, ())))
            current_lanes = [
                lane for lane in mapped_lanes
                if self._mapped_lane_query(lane)[0] == requested[0]
                and self._mapped_lane_query(lane)[1] is not None
            ]
            if current_lanes:
                return [[lane] for lane in current_lanes]

            # The remaining route can contain only the destination road while
            # CARLA is still on its direct predecessor or incoming connector.
            # Select only arrivals reached by the actual mapped source lane;
            # enumerating every lane on the destination would allow wrong turns.
            arrivals = []
            for mapped_lane in mapped_lanes:
                source_road, source_index = self._mapped_lane_query(mapped_lane)
                if source_index is not None:
                    arrivals.extend(
                        target for source, target in self.metsr_lane_connections.get(
                            (source_road, requested[0]), ()
                        )
                        if str(source) == mapped_lane
                    )
                else:
                    arrivals.extend(
                        target for _, target in getattr(
                            self, "metsr_internal_lane_connections", {}
                        ).get(mapped_lane, ())
                    )
            return [
                [lane] for lane in dict.fromkeys(map(str, arrivals))
                if self._mapped_lane_query(lane)[0] == requested[0]
                and self._mapped_lane_query(lane)[1] is not None
            ]

        start_waypoint = self.map.get_waypoint(
            target_start,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        start_lanes = []
        if start_waypoint is not None:
            start_key = f"{start_waypoint.road_id}_{start_waypoint.lane_id}"
            start_lanes = [
                lane
                for lane in self.scenic_to_metsr_map.get(start_key, ())
                if self._mapped_lane_query(lane)[0] == requested[0]
                and self._mapped_lane_query(lane)[1] is not None
            ]

        first_pairs = self.metsr_lane_connections.get(
            (requested[0], requested[1]), ()
        )
        if required_initial_lane is not None:
            start_lanes = [required_initial_lane]
        elif not start_lanes:
            start_lanes = list(dict.fromkeys(pair[0] for pair in first_pairs))

        # State: (lane-change cost, lane on current road, departure lanes).
        states = [(0, lane, []) for lane in start_lanes]
        for index, (source_road, target_road) in enumerate(
            zip(requested, requested[1:])
        ):
            pairs = self.metsr_lane_connections.get(
                (source_road, target_road), ()
            )
            if not pairs:
                return []
            next_states = {}
            for cost, arrival_lane, departures in states:
                _, arrival_index = self._mapped_lane_query(arrival_lane)
                for departure_lane, target_lane in pairs:
                    _, departure_index = self._mapped_lane_query(departure_lane)
                    if arrival_index is None or departure_index is None:
                        continue
                    candidate = (
                        cost + abs(arrival_index - departure_index),
                        target_lane,
                        departures + [departure_lane],
                    )
                    key = (target_lane, tuple(candidate[2]))
                    previous = next_states.get(key)
                    if previous is None or candidate[0] < previous[0]:
                        next_states[key] = candidate
            states = sorted(next_states.values(), key=lambda item: item[0])[:64]
            if not states:
                return []

        chains = []
        seen = set()
        for _, final_lane, departures in sorted(states, key=lambda item: item[0]):
            chain = tuple(departures + [final_lane])
            if chain not in seen:
                seen.add(chain)
                chains.append(list(chain))
        return chains

    @staticmethod
    def _polyline_fraction(polyline, fraction):
        points = [tuple(point[:3]) for point in polyline]
        if not points:
            return None
        if len(points) == 1:
            return points[0]
        lengths = [math.dist(start, end) for start, end in zip(points, points[1:])]
        target = sum(lengths) * fraction
        traversed = 0.0
        for start, end, length in zip(points, points[1:], lengths):
            if length == 0:
                continue
            if traversed + length >= target:
                ratio = (target - traversed) / length
                return tuple(
                    start[axis] + ratio * (end[axis] - start[axis])
                    for axis in range(3)
                )
            traversed += length
        return points[-1]

    @staticmethod
    def _polyline_progress(point, polyline):
        """Return along-line progress, total length, and lateral distance."""
        points = []
        for item in polyline:
            candidate = tuple(item[:2])
            if not points or candidate != points[-1]:
                points.append(candidate)
        if not points:
            return None, 0.0, math.inf
        if len(points) == 1:
            return 0.0, 0.0, math.dist(tuple(point[:2]), points[0])

        px, py = point[:2]
        progress = 0.0
        best_progress = 0.0
        best_distance = math.inf
        for start, end in zip(points, points[1:]):
            sx, sy = start
            ex, ey = end
            dx, dy = ex - sx, ey - sy
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            ratio = ((px - sx) * dx + (py - sy) * dy) / (length * length)
            ratio = min(1.0, max(0.0, ratio))
            closest = (sx + ratio * dx, sy + ratio * dy)
            distance = math.hypot(px - closest[0], py - closest[1])
            candidate_progress = progress + ratio * length
            if distance < best_distance:
                best_distance = distance
                best_progress = candidate_progress
            progress += length
        return best_progress, progress, best_distance

    def _index_carla_waypoints(self):
        if self._carla_waypoints_by_key is not None:
            return
        resolution = max(0.5, float(getattr(self.grp, "_sampling_resolution", 2.0)))
        waypoints = list(self.map.generate_waypoints(resolution))
        for segment in getattr(self.grp, "_topology", ()):
            waypoints.extend((segment["entry"], *segment["path"], segment["exit"]))
        index = {}
        seen = set()
        for waypoint in waypoints:
            identity = (
                waypoint.road_id,
                waypoint.section_id,
                waypoint.lane_id,
                round(float(waypoint.s), 3),
            )
            if identity in seen:
                continue
            seen.add(identity)
            key = f"{waypoint.road_id}_{waypoint.lane_id}"
            index.setdefault(key, []).append(waypoint)
        self._carla_waypoints_by_key = index

    def _lane_anchor_waypoint(self, sumo_lane, final=False):
        self._index_carla_waypoints()
        road, lane_index = self._mapped_lane_query(sumo_lane)
        if lane_index is None:
            return None
        centerline = self._query_mapped_centerline(road, lane_index)
        target = self._polyline_fraction(centerline, 0.9 if final else 0.7)
        if target is None:
            return None
        target = (target[0], -target[1], target[2])

        candidates = []
        for carla_key in self._sumo_lane_to_carla_keys.get(str(sumo_lane), ()):
            candidates.extend(self._carla_waypoints_by_key.get(carla_key, ()))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda waypoint: math.dist(
                (
                    waypoint.transform.location.x,
                    waypoint.transform.location.y,
                    waypoint.transform.location.z,
                ),
                target,
            ),
        )

    def _forward_lane_anchor_trace(
        self,
        sumo_lane,
        start_location,
        current_road,
        required_lane_changes,
        prefer_early=False,
    ):
        """Find a current-road lane anchor which is provably ahead.

        Progress is measured on the target SUMO lane, whose centerline follows
        traffic direction, instead of comparing CARLA ``s`` values across road
        fragments.  Candidate traces still pass the normal macro-road validator
        so a planner loop cannot masquerade as a forward anchor.
        """
        self._index_carla_waypoints()
        road, lane_index = self._mapped_lane_query(sumo_lane)
        if lane_index is None or str(road) != str(current_road):
            raise SimulationCreationError(
                f"Cannot anchor SUMO lane {sumo_lane} on current road "
                f"{current_road}"
            )
        start_waypoint = self.map.get_waypoint(
            start_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_waypoint is not None:
            start_key = f"{start_waypoint.road_id}_{start_waypoint.lane_id}"
            mapped_start_lanes = {
                str(mapped_lane)
                for mapped_lane in self.scenic_to_metsr_map.get(start_key, ())
            }
            source_lanes = []
            for mapped_lane in mapped_start_lanes:
                source_lane = str(mapped_lane)
                source_road, source_index = self._mapped_lane_query(source_lane)
                if source_index is None or str(source_road) == str(current_road):
                    continue
                source_lanes.append((source_lane, str(source_road)))
            direct_sources = [
                source_lane
                for source_lane, source_road in source_lanes
                if (source_lane, str(sumo_lane))
                in self.metsr_lane_connections.get(
                    (source_road, str(current_road)), ()
                )
            ]
            internal_pairs = [
                tuple(map(str, pair))
                for internal_lane in mapped_start_lanes
                for pair in getattr(
                    self, "metsr_internal_lane_connections", {}
                ).get(internal_lane, ())
            ]
            exact_internal_arrival = bool(internal_pairs) and all(
                target_lane == str(sumo_lane)
                for _, target_lane in internal_pairs
            )
            exact_predecessor_arrival = (
                source_lanes and len(direct_sources) == len(source_lanes)
            )
            if exact_predecessor_arrival or exact_internal_arrival:
                anchor = self._lane_anchor_waypoint(sumo_lane)
                if anchor is None:
                    raise SimulationCreationError(
                        f"No CARLA waypoint represents authoritative target "
                        f"lane {sumo_lane} on road {current_road}"
                    )
                trace = self._trace_route_segment(
                    start_location,
                    anchor,
                    (str(current_road),),
                    required_terminal_lane=sumo_lane,
                    use_planner_terminal_as_anchor=True,
                )
                anchor = trace[-1][0]
                anchor_key = f"{anchor.road_id}_{anchor.lane_id}"
                if str(sumo_lane) not in {
                    str(lane)
                    for lane in self.scenic_to_metsr_map.get(anchor_key, ())
                }:
                    raise SimulationCreationError(
                        f"CARLA anchor {anchor_key} does not map to exact target "
                        f"lane {sumo_lane}"
                    )
                return anchor, trace

        centerline = self._query_mapped_centerline(road, lane_index)
        start_progress, total_length, _ = self._polyline_progress(
            (start_location.x, -start_location.y), centerline
        )
        if start_progress is None:
            raise SimulationCreationError(
                f"No mapped centerline is available for SUMO lane {sumo_lane}"
            )

        sampling = max(
            0.5, float(getattr(self.grp, "_sampling_resolution", 2.0))
        )
        lane_changes = max(0, int(required_lane_changes))
        minimum_forward = max(
            sampling, FIRST_ROAD_LANE_CHANGE_DISTANCE * lane_changes
        )
        terminal_buffer = max(FIRST_ROAD_ANCHOR_END_BUFFER, sampling)
        usable_forward = total_length - start_progress - terminal_buffer
        if usable_forward + 1e-6 < minimum_forward:
            raise SimulationCreationError(
                f"Insufficient road remains to anchor {sumo_lane} ahead of "
                f"the CARLA pose on road {current_road}: usable "
                f"{max(0.0, usable_forward):.2f} m, required "
                f"{minimum_forward:.2f} m for {lane_changes} lane change(s)"
            )

        candidates = []
        for carla_key in self._sumo_lane_to_carla_keys.get(str(sumo_lane), ()):
            candidates.extend(self._carla_waypoints_by_key.get(carla_key, ()))
        desired_progress = start_progress + minimum_forward if prefer_early else min(
            total_length - terminal_buffer,
            max(0.9 * total_length, start_progress + minimum_forward),
        )
        viable = []
        for waypoint in candidates:
            location = waypoint.transform.location
            candidate_progress, _, _ = self._polyline_progress(
                (location.x, -location.y), centerline
            )
            if candidate_progress is None:
                continue
            if candidate_progress + 1e-6 < start_progress + minimum_forward:
                continue
            if candidate_progress > total_length - terminal_buffer + 1e-6:
                continue
            viable.append(
                (abs(candidate_progress - desired_progress), waypoint)
            )

        if not viable:
            raise SimulationCreationError(
                f"No indexed CARLA waypoint can anchor {sumo_lane} at least "
                f"{minimum_forward:.2f} m ahead on road {current_road}"
            )

        failures = []
        for _, waypoint in sorted(viable, key=lambda item: item[0]):
            try:
                trace = self._trace_route_segment(
                    start_location,
                    waypoint,
                    (str(current_road),),
                    required_terminal_lane=sumo_lane,
                    use_planner_terminal_as_anchor=True,
                )
            except SimulationCreationError as exc:
                failures.append(str(exc))
                continue
            effective_anchor = trace[-1][0]
            effective_location = effective_anchor.transform.location
            effective_progress, _, effective_error = self._polyline_progress(
                (effective_location.x, -effective_location.y), centerline
            )
            if (
                effective_progress is None
                or effective_error > 3.0
                or effective_progress + 1e-6 < start_progress + minimum_forward
                or effective_progress > total_length - terminal_buffer + 1e-6
            ):
                failures.append(
                    f"CARLA planner terminal for {sumo_lane} is not a usable "
                    f"forward anchor: progress={effective_progress}, "
                    f"lateral_error={effective_error}"
                )
                continue
            return effective_anchor, trace

        detail = f"; {failures[0]}" if failures else ""
        raise SimulationCreationError(
            f"No forward-reachable CARLA waypoint anchors {sumo_lane} on "
            f"road {current_road}{detail}"
        )

    @staticmethod
    def _same_carla_waypoint(first, second):
        return (
            first.road_id == second.road_id
            and first.section_id == second.section_id
            and first.lane_id == second.lane_id
            and abs(float(first.s) - float(second.s)) < 1e-3
        )

    def _trace_route_segment(
        self,
        start_location,
        target_waypoint,
        road_pair,
        required_terminal_lane=None,
        use_planner_terminal_as_anchor=False,
    ):
        trace = list(
            self.grp.trace_route(start_location, target_waypoint.transform.location)
        )
        if required_terminal_lane is not None:
            terminal = trace[-1][0] if trace else None
            terminal_key = (
                f"{terminal.road_id}_{terminal.lane_id}"
                if terminal is not None
                else None
            )
            mapped_terminal_lanes = {
                str(lane)
                for lane in self.scenic_to_metsr_map.get(terminal_key, ())
            }
            if str(required_terminal_lane) not in mapped_terminal_lanes:
                raise SimulationCreationError(
                    f"CARLA planner did not reach exact SUMO target lane "
                    f"{required_terminal_lane}; terminal={terminal_key}, "
                    f"mapped={sorted(mapped_terminal_lanes)}"
                )
        if use_planner_terminal_as_anchor:
            target_waypoint = trace[-1][0]
        start_waypoint = self.map.get_waypoint(
            start_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_waypoint is not None and (
            not trace or not self._same_carla_waypoint(start_waypoint, trace[0][0])
        ):
            trace.insert(0, (start_waypoint, None))
        if not trace or not self._same_carla_waypoint(trace[-1][0], target_waypoint):
            trace.append((target_waypoint, None))
        self._validate_carla_route_trace(
            trace,
            road_pair,
            context=f"stitching {road_pair[0]} -> {road_pair[-1]}",
        )
        return trace

    def _stitch_carla_route(
        self,
        requested_route,
        target_start,
        lane_chain,
        required_initial_lane=None,
    ):
        requested = tuple(str(road) for road in requested_route)
        if len(lane_chain) != len(requested):
            raise SimulationCreationError(
                "Internal route-stitch lane-chain length mismatch: "
                f"route={requested}, lanes={lane_chain}"
            )

        start_waypoint = self.map.get_waypoint(
            target_start,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        start_lanes = set()
        if start_waypoint is not None:
            start_key = f"{start_waypoint.road_id}_{start_waypoint.lane_id}"
            start_lanes = {
                str(lane)
                for lane in self.scenic_to_metsr_map.get(start_key, ())
                if self._mapped_lane_query(lane)[0] == requested[0]
            }

        def lane_change_count(source_lanes, target_lane):
            target_lane = str(target_lane)
            if target_lane in source_lanes:
                return 0
            _, target_index = self._mapped_lane_query(target_lane)
            source_indices = [
                self._mapped_lane_query(source_lane)[1]
                for source_lane in source_lanes
            ]
            source_indices = [
                index for index in source_indices if index is not None
            ]
            if target_index is None or not source_indices:
                return 1
            return max(
                1,
                min(abs(index - target_index) for index in source_indices),
            )

        # An authoritative pending target lane is an arrival constraint. It is
        # independent of lane_chain[0], the lane used later to depart this road.
        # Traffic Manager may therefore need two ordered same-road anchors:
        # actual lane -> authoritative arrival -> route-selected departure.
        anchor_specs = []
        first_departure_lane = str(lane_chain[0])
        segment_start = target_start
        required_initial_lane = (
            str(required_initial_lane)
            if required_initial_lane is not None
            else None
        )
        if required_initial_lane is not None:
            arrival_anchor_needed = (
                len(requested) == 1 or required_initial_lane not in start_lanes
            )
            if arrival_anchor_needed:
                arrival_anchor, arrival_trace = self._forward_lane_anchor_trace(
                    required_initial_lane,
                    segment_start,
                    requested[0],
                    lane_change_count(start_lanes, required_initial_lane),
                    prefer_early=True,
                )
                anchor_specs.append((0, arrival_anchor, arrival_trace))
                segment_start = arrival_anchor.transform.location
            if (
                len(requested) > 1
                and first_departure_lane != required_initial_lane
            ):
                departure_anchor, departure_trace = self._forward_lane_anchor_trace(
                    first_departure_lane,
                    segment_start,
                    requested[0],
                    lane_change_count(
                        {required_initial_lane}, first_departure_lane
                    ),
                )
                anchor_specs.append((0, departure_anchor, departure_trace))
                segment_start = departure_anchor.transform.location
        elif len(requested) == 1 or first_departure_lane not in start_lanes:
            departure_anchor, departure_trace = self._forward_lane_anchor_trace(
                first_departure_lane,
                segment_start,
                requested[0],
                lane_change_count(start_lanes, first_departure_lane),
            )
            anchor_specs.append((0, departure_anchor, departure_trace))
            segment_start = departure_anchor.transform.location

        for index in range(1, len(requested)):
            anchor = self._lane_anchor_waypoint(
                lane_chain[index], final=index == len(requested) - 1
            )
            if anchor is None:
                return None
            anchor_specs.append((index, anchor, None))

        stitched = []
        segment_start = target_start
        for index, anchor, prepared_trace in anchor_specs:
            if index == 0:
                road_pair = requested[:1]
            else:
                road_pair = requested[index - 1 : index + 1]
            trace = prepared_trace
            if trace is None:
                trace = self._trace_route_segment(
                    segment_start, anchor, road_pair
                )
            for item in trace:
                if (
                    stitched
                    and self._same_carla_waypoint(stitched[-1][0], item[0])
                ):
                    continue
                stitched.append(item)
            segment_start = anchor.transform.location

        self._validate_carla_route_trace(
            stitched, requested, context="validating the stitched trajectory"
        )
        return stitched

    def generate_carla_trajectory(
        self,
        route: list[str],
        obj: Object,
        required_first_lane_id=None,
        *,
        target_start=None,
    ) -> list[Lane]:
        """Stitch an explicitly requested macro-road path (not default routing).

        Default vehicle routing uses generate_carla_destination_trajectory and
        does not pass METS-R's remaining route to this exact-route helper.
        """
        requested = tuple(str(road) for road in route if road is not None)
        if not requested:
            raise SimulationCreationError(
                f"METS-R returned an empty route for {obj}"
            )

        if target_start is None:
            try:
                target_start = obj.carlaActor.get_location()
            except Exception:
                target_start = carla.Location(obj.position.x, -obj.position.y, 0)

        reported = requested
        requested = self._align_carla_route_start(requested, target_start)
        required_initial_lane = self._resolve_required_initial_lane(
            requested[0], required_first_lane_id
        )
        lane_chains = self._candidate_metsr_lane_chains(
            requested, target_start, required_initial_lane
        )
        if not lane_chains:
            start_waypoint = self.map.get_waypoint(
                target_start, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            start_key = (
                f"{start_waypoint.road_id}_{start_waypoint.lane_id}"
                if start_waypoint is not None else None
            )
            mapped_lanes = list(self.scenic_to_metsr_map.get(start_key, ()))
            raise SimulationCreationError(
                "No direct SUMO driving-lane chain represents the METS-R "
                f"macro route for {obj}: reported={list(reported)}, "
                f"effective={list(requested)}, carla_lane={start_key!r}, "
                f"is_junction={getattr(start_waypoint, 'is_junction', None)}, "
                f"mapped_sumo_lanes={mapped_lanes}, "
                f"required_initial_lane={required_initial_lane!r}"
            )

        failures = []
        for lane_chain in lane_chains:
            try:
                trace = self._stitch_carla_route(
                    requested,
                    target_start,
                    lane_chain,
                    required_initial_lane=required_initial_lane,
                )
                if trace:
                    return [waypoint.transform.location for waypoint, _ in trace]
            except SimulationCreationError as exc:
                failures.append(str(exc))

        detail = failures[0] if failures else "no CARLA lane anchor was available"
        raise SimulationCreationError(
            "Unable to stitch a CARLA path through every METS-R macro road "
            f"for {obj}: reported={list(reported)}, "
            f"effective={list(requested)}; {detail}"
        )

    def _collect_metsr_vehicle_data(self, objects: list[Object] | None = None):
        """
        Docstring for _collect_metsr_vehicle_data

        :param objects: List of objects which data should be queried
        :rtype objects: Scenic vehicle object

        Query metsr for state infomation on vehicle's default is all vehicles
        """
        objects = list(self.objects if objects is None else objects)
        if not objects:
            return {}

        obj_veh_ids = [self.getMetsrPrivateVehId(obj) for obj in objects]
        raw_veh_data = _require_metsr_response(
            self.metsr_client.query_vehicle(obj_veh_ids, True, True),
            "vehicle",
        )
        assert len(raw_veh_data["data"]) == len(obj_veh_ids), (
            "Inconsistent query result from METS-R client"
        )
        return {
            obj: raw_veh_data["data"][index]
            for index, obj in enumerate(objects)
        }


    def _save_metsr_state(self, file_name=None) -> None:
        """
        docstring for _save_metsr_state

        Saves metsr state to a file to allow for reproducible replay
        """
        if file_name == None:
            save_file = f"metsr_state_at_{self.count}.bin"
        else:
            save_file = file_name
        self.metsr_client.save(save_file)

    def _log_trip_times(self, file_name=None):
        """
        docstring for _log_trip_times

        :param file_name: target location and name for logs
        :rtype file_name: str

        Generate csv file containing total time to route completion for each vehicle
        """
        out_file = file_name if file_name else f"{self.run_name}_trip_logs.csv"
        trip_dict = {obj.name: obj.finished_route  - obj.trip_start  if hasattr(obj, "finished_route") else None for obj in self.objects[1:]}
        trip_df = pd.DataFrame([trip_dict])

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        trip_df.to_csv(out_file)


    def check_client_synchronization(self, objs=None, expected=None):
        """
        Docstring for synchronize_clients

        :param obj: Cosimulation car object
        :type obj: Scenic Object[s]

        Default : updates all CoSimulated object states in the METSR simulator
            (1) Can choose to specify which objects should be updated with obj arguement
        """
        all_actors = self.objects if not objs else objs
        all_veh_data = self._collect_metsr_vehicle_data()
        for i,obj in enumerate(all_actors):

            veh_data = all_veh_data[obj]
            if expected:
                expected_pos = expected[i]
            else:
                expected_pos = [veh_data["x"], veh_data["y"]]

            if not math.isclose(obj.position.x, expected_pos[0]) or not math.isclose(obj.position.y, expected_pos[1]):
                print(f"Obj: {obj.name} on road: {_metsr_vehicle_road(veh_data)} is out of sync discrepancy is:")
                print(f'(METSR : SCENIC) : x: {veh_data["x"], obj.position.x} y: {veh_data["y"], obj.position.y} z {veh_data["y"], obj.position.y}')
