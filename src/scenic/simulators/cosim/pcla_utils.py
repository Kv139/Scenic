"""PCLA integration hooks and cleanup of Scenic-owned CARLA vehicles.

The hooks support both current PCLA and older SimLingo integrations. Simulation
timing and actor ownership remain the responsibility of ``CosimSimulation``.
"""

import inspect


def notify_control_applied(obj, control):
    """Tell PCLA which control CARLA actually received for a Scenic object."""
    pcla = getattr(obj, "pcla", None)
    notify = getattr(pcla, "notify_control_applied", None)
    if callable(notify):
        notify(control)
        return

    # Legacy SimLingo predicts its next UKF state from ``agent.control``.
    agent = getattr(pcla, "agent_instance", None)
    if hasattr(agent, "control"):
        agent.control = control


def _same_actor(first, second):
    """Match actor handles, including different proxies for the same actor."""
    if first is None or second is None:
        return False
    if first is second:
        return True
    try:
        return first.id == second.id
    except (AttributeError, RuntimeError):
        # CARLA handles may have become invalid during teardown.
        return False


class _ScenicOwnedVehicle:
    """Let legacy PCLA identify sensors without owning the ego's destruction."""

    def __init__(self, actor):
        self._actor = actor

    @property
    def is_alive(self):
        return False

    def destroy(self):
        return False

    def __getattr__(self, name):
        return getattr(self._actor, name)


def remove_scenic_actor_from_cleanup_pool(pcla, scenic_actor):
    """Exclude Scenic's actor from legacy PCLA's global provider cleanup.

    Some agents register the ego in CarlaDataProvider's destruction pool as well
    as its state maps. Remove only that actor before PCLA clears the pool.
    """
    if scenic_actor is None:
        return
    cleanup_method = pcla.cleanup
    function = getattr(cleanup_method, "__func__", cleanup_method)
    provider = getattr(function, "__globals__", {}).get("CarlaDataProvider")
    actor_pool = getattr(provider, "_carla_actor_pool", None)
    if actor_pool is not None:
        for key, actor in tuple(actor_pool.items()):
            if _same_actor(actor, scenic_actor):
                actor_pool.pop(key)


def cleanup(pcla, scenic_actor=None):
    """Clean PCLA's sensors while leaving Scenic responsible for its vehicle."""
    remove_scenic_actor_from_cleanup_pool(pcla, scenic_actor)
    cleanup_method = pcla.cleanup
    try:
        parameters = inspect.signature(cleanup_method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    if any(
        parameter.name == "destroy_vehicle"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        cleanup_method(destroy_vehicle=False)
        return

    if not _same_actor(getattr(pcla, "vehicle", None), scenic_actor):
        cleanup_method()
        return

    # Legacy cleanup needs the vehicle ID to find sensors before destroying its
    # vehicle. The facade preserves that ID and makes destruction a no-op.
    pcla.vehicle = _ScenicOwnedVehicle(scenic_actor)
    try:
        cleanup_method()
    finally:
        pcla.vehicle = None


def is_pcla_controlled_ego(simulation, obj):
    """Whether this object is the ego whose motion remains owned by PCLA."""
    return obj is simulation.ego and getattr(obj, "pcla", None) is not None
