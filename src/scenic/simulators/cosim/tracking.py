"""Refresh nearby actors every step and optionally refresh distant traffic.

The registry keeps identities for return trips through the neighborhood. A vehicle
absent from an index keeps its last observation until it becomes nearby or a
configured full refresh occurs; it is not considered completed. Unmarked
objects keep ordinary Scenic semantics.
"""

import math
import numbers

from scenic.simulators.metsr.bridge_utils import METSRControlError, require_metsr_response


def neighborhood_keys(response, message_type):
    """Validate a neighborhood index, preserving public/private ID namespaces."""
    response = require_metsr_response(response, message_type, require_all=True)
    keys = set()
    for record in response["data"]:
        if (
            not isinstance(record, dict)
            or type(record.get("vehicleId")) not in (int, str)
            or str(record["vehicleId"]).strip() == ""
            or not isinstance(record.get("isPrivate"), bool)
        ):
            raise METSRControlError(
                f"{message_type} returned an invalid identity", record=record
            )
        keys.add((record["isPrivate"], str(record["vehicleId"])))
    return keys


class NeighborhoodTracker:
    """Incremental Scenic identity registry and deterministic current selection."""

    def __init__(self):
        self.objects_by_key = {}
        self._order = {}
        self._pinned = {}
        self.selected = ()
        self.has_background = False
        self.observation_times_s = {}
        self._unobserved = {}

    def register(self, obj, vehicle_id, *, ego=False):
        if obj in self._order:
            return
        self._order[obj] = len(self._order)
        self.objects_by_key[(True, str(vehicle_id))] = obj
        self.observation_times_s[obj] = None
        self._unobserved[obj] = None
        # Sensors and ego remain live even if a scenario accidentally marks them
        # as passive. The marker is a creation-time opt-in, not a dynamic switch.
        background = bool(getattr(obj, "cosim_background", False))
        if ego or getattr(obj, "sensors", None) or not background:
            self._pinned[obj] = None
            self.selected = (*self.selected, obj)
        else:
            self.has_background = True

    def select(self, keys, *, owned=(), pending_ids=(), completed=()):
        selected = dict(self._pinned)
        # A newly spawned vehicle needs its first native observation even when
        # it is distant. Until that succeeds its timestamp remains explicitly None.
        selected.update(self._unobserved)
        for key in keys:
            obj = self.objects_by_key.get(key)
            if obj is not None and obj not in completed:
                selected[obj] = None
        for vehicle_id in pending_ids:
            obj = self.objects_by_key.get((True, str(vehicle_id)))
            if obj is not None and obj not in completed:
                selected[obj] = None
        # Never drop a CARLA-owned actor before its authoritative final handoff.
        for obj in owned:
            selected[obj] = None
        self.selected = tuple(sorted(selected, key=self._order.__getitem__))
        return self.selected

    def mark_updated(self, objects, time_s):
        """Commit freshness only after a complete dynamic-property update succeeds."""
        for obj in objects:
            self.observation_times_s[obj] = time_s
            self._unobserved.pop(obj, None)


class BackgroundRefreshSchedule:
    """Simulation-time schedule; -1 disables periodic distant observations."""

    def __init__(self, hz, timestep):
        if isinstance(hz, bool) or not isinstance(hz, numbers.Real):
            raise ValueError("background_tracking_hz must be -1 or a finite positive number")
        if not math.isfinite(hz) or (hz <= 0 and hz != -1):
            raise ValueError("background_tracking_hz must be -1 or a finite positive number")
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError("background tracking requires a finite positive timestep")
        if hz > 1.0 / timestep + 1e-12:
            raise ValueError("background_tracking_hz cannot exceed 1 / simulation timestep")
        self.hz = float(hz)
        self.enabled = hz != -1
        self.next_refresh_s = 0.0 if self.enabled else math.inf
        self.last_refresh_s = None

    def due(self, time_s):
        return self.enabled and time_s + 1e-10 >= self.next_refresh_s

    def commit(self, time_s):
        self.last_refresh_s = time_s
        self.next_refresh_s = (
            (math.floor(time_s * self.hz + 1e-10) + 1) / self.hz
            if self.enabled else math.inf
        )
