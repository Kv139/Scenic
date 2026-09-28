from scenic.core.object_types import Object
from scenic.core.regions import CircularRegion
from scenic.domains.driving.roads import Lane, Road


class network_cache:
    def __init__(self, workspace):
        self.network_roads = list(workspace.network.allRoads)
        self.roads_by_id = {str(road.id): road for road in self.network_roads}
        self.network_intersections = list(workspace.network.intersections)
        self.obj_lane_cache = {}

    def _nearest_lane(self, obj: Object) -> Lane | None:
        if obj in self.obj_lane_cache:
            lane = self.obj_lane_cache[obj]
            if lane.containsPoint(obj.position):
                return lane

        nearest_lane = obj._lane
        if nearest_lane is not None:
            self.obj_lane_cache[obj] = nearest_lane

        return nearest_lane

    def _get_bubble_roads(
        self, bubble_region: CircularRegion, anchor_road_id=None
    ) -> list[Road]:
        """Collect the local road component containing the ego's actual road.

        Scenic road polygons are two-dimensional, so polygon overlap alone also
        selects overpasses. Explicit lane links distinguish those road layers
        and keep connecting ramps without imposing an elevation cutoff. A
        caller without an anchor retains the geometric selection behavior.
        """
        candidates = [
            road for road in self.network_roads if road.intersects(bubble_region)
        ]
        if anchor_road_id is None:
            return candidates

        adjacency = getattr(self, "_bubble_road_adjacency", None)
        if adjacency is None:
            adjacency = {str(road.id): set() for road in self.network_roads}
            for road in self.network_roads:
                road_id = str(road.id)
                for lane in road.lanes:
                    for neighbor in (lane._predecessor, lane._successor):
                        neighbor_road = getattr(neighbor, "road", None)
                        if neighbor_road is None:
                            continue
                        neighbor_id = str(neighbor_road.id)
                        if neighbor_id in adjacency:
                            # Incoming and opposing traffic can interact with
                            # the ego even when it cannot drive their direction.
                            adjacency[road_id].add(neighbor_id)
                            adjacency[neighbor_id].add(road_id)
            self._bubble_road_adjacency = adjacency

        candidate_ids = {str(road.id) for road in candidates}
        selected = set()
        pending = [str(anchor_road_id)]
        while pending:
            road_id = pending.pop()
            if road_id in selected or road_id not in candidate_ids:
                continue
            selected.add(road_id)
            pending.extend(adjacency[road_id] - selected)
        return [road for road in candidates if str(road.id) in selected]
