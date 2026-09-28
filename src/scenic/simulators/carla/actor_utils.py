"""CARLA actor mechanics shared by standalone and co-simulation backends."""

import math
import warnings

import carla

from scenic.core.simulators import SimulationCreationError
from scenic.simulators.carla import coordinate_utils as utils
from scenic.simulators.carla.blueprints import oldBlueprintNames


def resolve_blueprint(blueprint_library, model):
    """Find a model or its legacy name, returning the blueprint and chosen name."""
    try:
        return blueprint_library.find(model), model
    except IndexError as error:
        for old_name in oldBlueprintNames.get(model, ()):
            try:
                return blueprint_library.find(old_name), old_name
            except IndexError:
                continue
        raise error


def prepare_actor_blueprint(obj, blueprint_library):
    """Resolve a Scenic actor's model and apply its shared blueprint attributes."""
    try:
        blueprint, model = resolve_blueprint(blueprint_library, obj.blueprint)
    except IndexError as error:
        raise SimulationCreationError(
            f"Unable to find blueprint {obj.blueprint} for object {obj}"
        ) from error
    if model != obj.blueprint:
        warnings.warn(
            f"CARLA blueprint {obj.blueprint} not found; using older version {model}"
        )
        obj.blueprint = model
    if obj.rolename is not None:
        blueprint.set_attribute("role_name", obj.rolename)
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "False")
    if blueprint.has_attribute("color") and obj.color is not None:
        color = obj.color
        blueprint.set_attribute(
            "color", f"{int(color.r * 255)},{int(color.g * 255)},{int(color.b * 255)}"
        )
    return blueprint


def configure_actor(obj, world, blueprint_library):
    """Initialize vehicle dimensions and gear, or attach a walker controller."""
    actor = obj.carlaActor
    if isinstance(actor, carla.Vehicle):
        extent = actor.bounding_box.extent
        # Preserve sampled dimensions for invalid extents (CARLA issue #5841).
        obj.width = extent.y * 2 if extent.y > 0 else obj.width
        obj.length = extent.x * 2 if extent.x > 0 else obj.length
        obj.height = extent.z * 2 if extent.z > 0 else obj.height
        actor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
    elif isinstance(actor, carla.Walker):
        actor.apply_control(carla.WalkerControl())
        blueprint = blueprint_library.find("controller.ai.walker")
        controller = world.try_spawn_actor(blueprint, carla.Transform(), actor)
        if controller is None:
            raise SimulationCreationError(
                f"Unable to spawn carla controller for object {obj}"
            )
        obj.carlaController = controller


def destroy_walker_controller(controller, *, is_registered=None):
    """Stop a walker controller before destroying its server-side actor."""
    controller.stop()
    if is_registered is None or is_registered(controller):
        controller.destroy()


def actor_properties(actor, parent_orientation):
    """Convert a CARLA actor snapshot to Scenic dynamic properties."""
    transform = actor.get_transform()
    position = utils.carlaToScenicPosition(transform.location)
    velocity = utils.carlaToScenicPosition(actor.get_velocity())
    angular_velocity = actor.get_angular_velocity()
    orientation = utils.carlaToScenicOrientation(transform.rotation)
    yaw, pitch, roll = parent_orientation.localAnglesFor(orientation)
    return dict(
        position=position,
        velocity=velocity,
        speed=math.hypot(*velocity),
        angularSpeed=utils.carlaToScenicAngularSpeed(angular_velocity),
        angularVelocity=utils.carlaToScenicAngularVel(angular_velocity),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        elevation=utils.carlaToScenicElevation(transform.location),
    )
