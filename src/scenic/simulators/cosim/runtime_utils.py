"""CARLA actor support, visualization, and co-simulation run diagnostics."""

import math
import os
import warnings

import carla
import numpy as np
import pandas as pd
import pygame

from scenic.core.simulators import SimulationCreationError
from scenic.core.vectors import Orientation, Vector
from scenic.simulators.carla import actor_utils
from scenic.simulators.carla.blueprints import busModels, carModels
import scenic.simulators.carla.coordinate_utils as utils
import scenic.simulators.carla.visualization_utils as visuals
from scenic.simulators.metsr import bridge_utils, routing_utils

from . import pcla_utils


def render_step_interval(timestep, requested):
    """Resolve a render cadence, defaulting to one simulated second."""
    if requested is None:
        return max(1, int(round(1.0 / float(timestep))))
    interval = int(requested)
    if interval <= 0:
        raise ValueError("metsr_render_freq must be a positive integer")
    return interval


def boundary_vehicle_blueprint(simulation, key, state, scenic_object):
    """Use the Scenic model when known, or a CARLA car/bus for native fleets."""
    if scenic_object is not None:
        models = (scenic_object.blueprint,)
    else:
        models = busModels if state["vehicleClass"] == 2 else carModels
    for model in models:
        try:
            if scenic_object is not None:
                blueprint, _ = actor_utils.resolve_blueprint(
                    simulation.blueprintLib, model
                )
            else:
                blueprint = simulation.blueprintLib.find(model)
        except IndexError:
            continue
        blueprint.set_attribute(
            "role_name",
            f"metsr_boundary_{'private' if key[0] else 'public'}_{key[1]}",
        )
        if scenic_object is not None and blueprint.has_attribute("color"):
            color = getattr(scenic_object, "color", None)
            if color is not None:
                blueprint.set_attribute(
                    "color",
                    f"{int(color.r * 255)},{int(color.g * 255)},{int(color.b * 255)}",
                )
        return blueprint
    raise SimulationCreationError(
        f"No CARLA vehicle blueprint is available for METS-R boundary vehicle {key}"
    )


def boundary_vehicle_transform(simulation, state, scenic_object, actor=None):
    """Convert a current METS-R pose, keeping a static vehicle on the ground."""
    try:
        x, y = float(state["x"]), float(state["y"])
        z = float(state.get("z") or 0.0)
        bearing = float(state["bearing"])
        if not all(math.isfinite(value) for value in (x, y, z, bearing)):
            raise ValueError("non-finite pose")
    except (KeyError, TypeError, ValueError) as error:
        raise bridge_utils.METSRControlError(
            "METS-R boundary vehicle has no valid current pose", record=state
        ) from error
    location = carla.Location(x, -y, z)
    if getattr(scenic_object, "snapToGround", True):
        waypoint = simulation.map.get_waypoint(
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


def remove_boundary_actor(simulation, key):
    actor = simulation.boundary_actors[key]
    if carla_actor_is_registered(simulation, actor) and actor.destroy() is False:
        raise SimulationCreationError(
            f"CARLA could not remove METS-R boundary vehicle {key}"
        )
    del simulation.boundary_actors[key]


def record_step_statistics(simulation):
    """Record actor counts and periodically append run statistics."""
    simulation.bubble_sizes.append(len(simulation.carla_actors))
    simulation.total_active_vehicles.append(
        len(simulation.objects) - len(simulation.bubble_spawn_queue)
    )
    if simulation.count % 200 == 0:
        print(
            f"Step: {simulation.count}. Total actors: {len(simulation.objects)}, bubble queue:{len(simulation.bubble_spawn_queue)} "
        )
        print(f"Total active vehicles: {simulation.total_active_vehicles[-1]}")
        print(
            f"Total bubble actors: {len(simulation.carla_actors) + len(simulation.bubble_spawn_queue)}"
        )
        print(f"Completed routes: {len(simulation.completed_route)}")

    if simulation.run_name is None or simulation.count % 50 != 0:
        return
    out_file = simulation.run_name + "_veh_data.csv"
    data = {
        "active_vehicles": simulation.total_active_vehicles[-1],
        "bubble_actors": len(simulation.carla_actors),
        "bubble_queue": len(simulation.bubble_spawn_queue),
        "completed_routes": len(simulation.completed_route),
        **simulation.road_pop_density,
    }
    if simulation.count == 0:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    pd.DataFrame([data]).to_csv(
        out_file,
        mode="w" if simulation.count == 0 else "a",
        header=simulation.count == 0,
    )


def carla_actor_is_alive(actor) -> bool:
    """Return whether an actor can still be safely used during teardown."""
    try:
        return bool(actor.is_alive)
    except RuntimeError:
        # CARLA raises here when the server-side actor has already gone away.
        return False


def carla_actor_is_registered(simulation, actor) -> bool:
    """Check server membership because CARLA may cache actor.is_alive."""
    if not carla_actor_is_alive(actor):
        return False

    try:
        return simulation.carla_world.get_actor(actor.id) is not None
    except RuntimeError:
        # The CARLA connection or actor can disappear while teardown is in
        # progress. A further destroy request cannot improve that state.
        return False


def report_cleanup_failure(action, error) -> None:
    """Report a teardown failure without replacing the simulation failure."""
    message = f"Co-simulation cleanup failed while {action}: {error}"
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except Exception:
        # Warning filters are allowed to promote warnings to exceptions. Cleanup
        # must remain best-effort even in that configuration.
        print(message)


def run_pre_actor_teardown_callbacks(simulation) -> None:
    callbacks = getattr(simulation, "_pre_actor_teardown_callbacks", ())
    while callbacks:
        callback = callbacks.pop()
        try:
            callback()
        except Exception as error:
            report_cleanup_failure("running a pre-actor teardown callback", error)


def destroy_attached_carla_sensors(simulation, actor) -> None:
    """Stop and destroy every live sensor attached to one Scenic actor."""
    actor_id = actor.id
    try:
        sensors = simulation.carla_world.get_actors().filter("sensor.*")
    except Exception as error:
        report_cleanup_failure("enumerating attached CARLA sensors", error)
        return

    for sensor in tuple(sensors):
        try:
            parent = getattr(sensor, "parent", None)
            if parent is None or getattr(parent, "id", None) != actor_id:
                continue
            if sensor.is_listening:
                sensor.stop()
            if carla_actor_is_registered(simulation, sensor):
                sensor.destroy()
        except Exception as error:
            sensor_id = getattr(sensor, "id", "unknown")
            report_cleanup_failure(
                f"destroying CARLA sensor {sensor_id} attached to actor " f"{actor_id}",
                error,
            )


def destroy_carla_object_during_teardown(simulation, obj) -> None:
    """Best-effort cleanup for one CARLA object, safe to call only once."""
    actor = getattr(obj, "carlaActor", None)
    if actor is None:
        return

    if isinstance(actor, carla.Vehicle) and carla_actor_is_registered(simulation, actor):
        try:
            actor.set_autopilot(False, simulation.tm.get_port())
        except Exception as error:
            report_cleanup_failure("disabling CARLA autopilot", error)

    if isinstance(actor, carla.Walker) and carla_actor_is_registered(simulation, actor):
        controller = getattr(obj, "carlaController", None)
        if controller is not None:
            try:
                actor_utils.destroy_walker_controller(
                    controller,
                    is_registered=lambda actor: carla_actor_is_registered(
                        simulation, actor
                    ),
                )
            except Exception as error:
                report_cleanup_failure("destroying a CARLA walker controller", error)

    pcla = getattr(obj, "pcla", None)
    pcla_cleaned_sensors = False
    if pcla is not None:
        try:
            pcla_utils.cleanup(pcla, actor)
            pcla_cleaned_sensors = True
        except Exception as error:
            report_cleanup_failure("cleaning up PCLA", error)

    # PCLA already destroys its sensors through their original handles.
    # CARLA's last snapshot can still enumerate those IDs until the next
    # tick, so sweeping them again would issue duplicate destroy requests.
    if not pcla_cleaned_sensors:
        destroy_attached_carla_sensors(simulation, actor)

    try:
        if carla_actor_is_registered(simulation, actor):
            actor.destroy()
    except Exception as error:
        report_cleanup_failure("destroying a CARLA actor", error)
    finally:
        # Make teardown idempotent and prevent later code from reusing a stale
        # Python handle even when the actor was destroyed by legacy PCLA.
        obj.carlaActor = None
        clear_carla_route_state(simulation, obj)


def log_trip_times(simulation, file_name=None):
    """
    docstring for log_trip_times

    :param file_name: target location and name for logs
    :rtype file_name: str

    Generate csv file containing total time to route completion for each vehicle
    """
    out_file = file_name if file_name else f"{simulation.run_name}_trip_logs.csv"
    trip_dict = {
        obj.name: (
            obj.finished_route - obj.trip_start
            if hasattr(obj, "finished_route")
            else None
        )
        for obj in simulation.objects[1:]
    }
    trip_df = pd.DataFrame([trip_dict])

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    trip_df.to_csv(out_file)


def check_client_synchronization(simulation, objs=None, expected=None):
    """
    Docstring for synchronize_clients

    :param obj: Cosimulation car object
    :type obj: Scenic Object[s]

    Default : updates all CoSimulated object states in the METSR simulator
        (1) Can choose to specify which objects should be updated with obj arguement
    """
    all_actors = simulation.objects if not objs else objs
    all_veh_data = bridge_utils.collect_metsr_vehicle_data(
        simulation.metsr_client,
        {obj: simulation.getMetsrPrivateVehId(obj) for obj in simulation.objects},
    )
    for i, obj in enumerate(all_actors):
        veh_data = all_veh_data[obj]
        if expected:
            expected_pos = expected[i]
        else:
            expected_pos = [veh_data["x"], veh_data["y"]]

        if not math.isclose(obj.position.x, expected_pos[0]) or not math.isclose(
            obj.position.y, expected_pos[1]
        ):
            print(
                f"Obj: {obj.name} on road: {bridge_utils.metsr_vehicle_road(veh_data)} is out of sync discrepancy is:"
            )
            print(
                f'(METSR : SCENIC) : x: {veh_data["x"], obj.position.x} y: {veh_data["y"], obj.position.y} z {veh_data["y"], obj.position.y}'
            )


def prepare_carla_spawn(simulation, obj, native_state=None):
    """Select a blueprint and convert the current admission pose."""
    blueprint = actor_utils.prepare_actor_blueprint(obj, simulation.blueprintLib)
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
        orientation = routing_utils.degenerate_connector_spawn_orientation(
            simulation, native_state, position
        )
        if orientation is None:
            orientation = Orientation.fromEuler(heading, 0, 0)
        native_velocity = Vector(0, speed, 0).rotatedBy(orientation)

    loc = utils.scenicToCarlaLocation(
        position,
        world=simulation.carla_world,
        blueprint=obj.blueprint,
        snapToGround=obj.snapToGround,
    )
    rot = utils.scenicToCarlaRotation(orientation)
    transform = carla.Transform(loc, rot)
    return blueprint, transform, native_velocity


def seed_carla_spawn_properties(simulation, obj, transform, velocity):
    """Keep admission properties available until CARLA publishes the actor."""
    global_orientation = utils.carlaToScenicOrientation(transform.rotation)
    yaw, pitch, roll = obj.parentOrientation.localAnglesFor(global_orientation)
    simulation._carla_spawn_property_seeds[obj] = dict(
        position=utils.carlaToScenicPosition(transform.location),
        velocity=velocity,
        speed=math.hypot(*velocity),
        angularSpeed=0,
        angularVelocity=Vector(0, 0, 0),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        elevation=utils.carlaToScenicElevation(transform.location),
    )


def initialize_display(simulation):
    """Create the CARLA preview window and HUD before Scenic spawns actors."""
    simulation.displayDim = (1280, 720)
    simulation.displayClock = pygame.time.Clock()
    simulation.camTransform = 0
    pygame.init()
    pygame.font.init()
    simulation.hud = visuals.HUD(*simulation.displayDim)
    simulation.display = pygame.display.set_mode(
        simulation.displayDim, pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    simulation.cameraManager = None


def initialize_camera(simulation):
    """Attach the rendered preview camera after the ego actor exists."""
    simulation.cameraManager = visuals.CameraManager(
        simulation.carla_world, simulation.ego.carlaActor, simulation.hud
    )
    simulation.cameraManager._transform_index = 0
    simulation.cameraManager.set_sensor(0)
    simulation.cameraManager.set_transform(simulation.camTransform)


def client_sync_diagnostic_due(count, synchronized_any):
    """Run the read-only consistency diagnostic once per 200 steps."""
    return synchronized_any and count % 200 == 0


def clear_carla_route_state(simulation, obj):
    """Forget native ownership and a CARLA route installed by co-simulation."""
    simulation.metsr_state.authoritative_segments.pop(obj, None)
    plan = simulation._carla_destination_plans.pop(obj, None)
    if plan is not None and getattr(obj, "trajectory", None) is plan[1]:
        obj.trajectory = None


def within_threshold_to(object, cars, threshold=None, verbose=False):
    """Return whether an actor is within the threshold and the nearest other actor."""
    is_close = False
    object_pos = np.array(object.position)
    obj_distances = {}
    danger_veh = None
    min_dist = math.inf
    for car in cars:
        if car != object:
            dist = np.linalg.norm(np.array(car.position) - object_pos)
            if dist < min_dist:
                danger_veh = car
                min_dist = dist
            obj_distances[car.name] = dist
    if danger_veh is not None:
        is_close = threshold is None or min_dist <= threshold
    if verbose:
        if obj_distances:
            print(f"Min Distance for {object} was: {min_dist}")
        else:
            print(f"No other CARLA actors were available near {object}")
    return is_close, danger_veh
