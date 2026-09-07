[<img src="https://docs.scenic-lang.org/en/latest/_static/logo-full.svg" alt="Scenic Logo" height="100">](https://scenic-lang.org/)

[![Documentation Status](https://readthedocs.org/projects/scenic-lang/badge/?version=latest)](https://docs.scenic-lang.org/en/latest/?badge=latest)
[![Tests Status](https://github.com/BerkeleyLearnVerify/Scenic/actions/workflows/run-tests.yml/badge.svg)](https://github.com/BerkeleyLearnVerify/Scenic/actions/workflows/run-tests.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A compiler and scenario generator for Scenic, a domain-specific probabilistic programming language for modeling the environments of cyber-physical systems.
Please see the [documentation](https://docs.scenic-lang.org/) for installation instructions, as well as tutorials and other information about the Scenic language, its implementation, and its interfaces to various simulators.

For an overview of the language and some of its applications, see our [2022 journal paper](https://link.springer.com/article/10.1007/s10994-021-06120-5) on Scenic 2, which extends our [PLDI 2019 paper](https://arxiv.org/abs/1809.09310) on Scenic 1.
The new syntax and features of Scenic 3 are described in our [CAV 2023 paper](https://arxiv.org/abs/2307.03325).
Our [Publications](https://docs.scenic-lang.org/en/latest/publications.html) page lists additional relevant publications.

Scenic was initially designed and implemented at UC Berkeley by Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L. Sangiovanni-Vincentelli, and Sanjit A. Seshia.
Subsequent work has been done primarily at UC Berkeley and UC Santa Cruz: in particular, Edward Kim made major contributions to Scenic 2, and Eric Vin, Shun Kashiwa, Matthew Rhea, and Ellen Kalvan to Scenic 3.
Please see our [Credits](https://docs.scenic-lang.org/en/latest/credits.html) page for details and more contributors.

If you have any problems using Scenic, please submit an issue to [our GitHub repository](https://github.com/BerkeleyLearnVerify/Scenic) or start a conversation on our [community forum](https://forum.scenic-lang.org/).

The repository is organized as follows:

* the _src/scenic_ directory contains the package proper;
* the _examples_ directory has many examples of Scenic programs;
* the _assets_ directory contains meshes and other resources used by the examples and tests;
* the _docs_ directory contains the sources for the documentation;
* the _tests_ directory contains tests for the Scenic tool.

## PCLA configuration for METS-R/CARLA co-simulation

Run Scenic and PCLA in the PCLA Conda environment because both libraries execute
inside the dashboard process. For the WSL checkout layout used by the METS-R demo:

```bash
conda activate PCLA
python -m pip install -e /mnt/d/Git/Scenic --no-deps
export PCLA_HOME=/mnt/d/Git/PCLA
export PYTHONPATH=/mnt/d/Git/Scenic/src:$PCLA_HOME:$PYTHONPATH
```

The PCLA environment pins `antlr4-python3-runtime==4.9.3` through Hydra/OmegaConf.
Scenic's ANTLR 4.11 dependency is used by its Webots parser, so the CARLA/METS-R
configuration above retains 4.9.3 and installs Scenic without resolving optional
Webots dependencies. Use a separate Scenic environment for Webots scenarios.

Before starting the dashboard, confirm that the editable Scenic checkout and the
expected PCLA checkout are imported:

```bash
python - <<'PY'
import PCLA
import scenic.simulators.cosim.simulator as simulator

print(PCLA.__file__)
print(simulator.__file__)
PY
```

Set `pcla_agent` and, optionally, `pcla_route` in the Scenic scenario. The updated
co-simulation adapter treats the PCLA ego as CARLA-authoritative for its complete
runtime: Scenic copies its pose into a METS-R shadow but does not lane-teleport it,
install a Traffic Manager path, place it in a road-entry hold for route preparation,
or demote it into METS-R. METS-R mirrors the authoritative pose without requiring
Scenic to query or rewrite the shadow vehicle's route.

For the METS-R HPC dashboard demo, the default PCLA agent is `lav_fast`, PCLA's
explicitly inference-speed-optimized learned agent. It is substantially cheaper
than the 647-million-parameter `simlingo_simlingo` agent. Install only PCLA's
LAV pretrained folder before using it:

```bash
cd /mnt/d/Git/PCLA
git pull --ff-only
python -m pip install -U huggingface_hub
python pcla_functions/download_weights.py --agents lav
```

PCLA checkouts older than commit `a7c12c5` still reference a removed 14.5-GB
Hugging Face ZIP and will fail with HTTP 404. Update PCLA before using the
per-agent command above; Scenic does not modify the PCLA checkout.

LAV requires CARLA to be started with `-vulkan`. You can still select SimLingo
explicitly with `--pcla-agent simlingo_simlingo`. PCLA agent choice remains a
scenario parameter; the Scenic adapter does not import or modify PCLA internals.

Scenic creates and therefore owns the CARLA actor's lifecycle. At final simulation
teardown only, it calls `pcla.cleanup(destroy_vehicle=False)` when that argument is
supported. For an unmodified legacy PCLA without the argument, Scenic supplies a
non-owning actor facade and removes the Scenic ego from PCLA's private
`CarlaDataProvider` destruction pool while `cleanup()` stops the agent and removes
its sensors. Scenic then checks CARLA's world registry and destroys the real ego
actor exactly once if it is still present. This final cleanup is separate from the
per-step synchronization loop.

Some SimLingo versions print `No cfg or encoder to delete processor` during this
cleanup. That line only reports that an optional LLaVA processor was not allocated;
it is not an actor-lifecycle or simulation failure.

## Co-simulation performance defaults

Scenic vehicle state in METS-R is queried once before the CARLA tick and shared
by bubble promotion, CARLA synchronization, and post-sync demotion. Boundary
obstacle poses are fetched separately in one batch. Ordinary CARLA vehicle
teleports and ready departure-queue admissions are sent as batches. A fresh
vehicle snapshot is still taken after the METS-R tick. METS-R visualization
defaults to once per simulated second (10 steps with a 0.1-second timestep).

## METS-R/CARLA ownership handoff

Scenic NPCs starting on native METS-R roads enter the mapped road's departure
queue through `generateTripsByRoad`. Their sampled coordinates select the origin
road but are not used for exact placement. The spawn projection-distance check
applies to co-simulation initialization on CARLA-controlled roads and connectors,
where Scenic supplies an exact pose; it does not reject native queue departures
because an OpenDRIVE lane extends past its mapped SUMO lane endpoint.

This adapter targets the current native METS-R SIM and METS-R HPC protocol. A
CARLA-controlled vehicle stays in CARLA while its authoritative pose is on a
controlled physical road or a server-returned connector. Scenic sends the exact
``segmentId`` and, for physical roads, ``laneIndex`` on every
``teleportCoSimVeh`` update. Connector IDs remain opaque: Scenic learns their
endpoints and internal-edge aliases from ``setCoSimRoad`` and never constructs or
parses a connector ID.

If CARLA temporarily reports a genuinely unmapped driving lane inside a junction,
Scenic retains the actor's previously accepted connector only when the actor is
within the existing 4.25 m CARLA-waypoint projection tolerance. A mapped target
road replaces the connector on the next observation. An unrelated mapped road, an
unmapped non-junction lane, or a farther projected pose fails closed; Scenic never
substitutes METS-R's pose for CARLA's authoritative pose.

At a junction merge or fork, CARLA's nearest waypoint can switch between
overlapping movements. Scenic retains the occupied connector when the movements
share an incoming or outgoing road, their reported intersection IDs agree, and
the pose is within 4.25 m of both a junction waypoint and the occupied path. At a
fork, CARLA can select another movement from the same incoming road: once its
pose leaves the first path's tolerance and fits the observed path, Scenic accepts
that connector. A downstream-road observation advances the segment normally.
This disambiguation changes neither CARLA's pose nor either simulator's route.

When CARLA reaches a physically connected native successor, Scenic sends that native road
and lane in one ``teleportCoSimVeh`` request. METS-R atomically installs the
vehicle in native simulation and replies with ``controlMode="native"`` and
``releasedFromCoSim=true``. Scenic validates those fields, the returned
``segmentId``, and the selected lane before destroying the CARLA actor. There is
no Scenic pending-transition or draining phase; a failed or incomplete
acknowledgement raises an error and leaves the CARLA actor owned by Scenic.

Road ownership is updated after vehicle synchronization. An old bubble road is
released once no remaining CARLA actor occupies that road or one of its incident
controlled connectors. This is an occupancy safety check, not a handoff state or
retry loop.

The native ``teleportDigitalTwinVeh`` API has two mutually exclusive position
modes. Coordinate mode sends ``x``/``y`` (and optional ``z``) and lets METS-R
match a native segment. Segment mode sends ``segmentId`` plus
``distanceToSegmentEnd``, with either ``laneIndex`` for a road or
``connectorPathId`` for a connector. Scenic uses coordinate mode when placing a
new native background vehicle and never sends the removed ``roadId`` wire field.


## Boundary vehicles at co-simulation exits

Scenic calls `query_boundary_vehicle()` (`queryBoundaryVeh` in METS-R,
`boundaryVehicle` on the wire) after acquiring bubble roads and before queue
admission, vehicle promotion, and the CARLA tick. METS-R selects vehicles on native
roads immediately downstream of controlled connectors, within strictly less than
1.2 vehicle lengths of their lane entry. No connector reservation is required.

Each returned vehicle gets a separate CARLA `vehicle.*` actor with autopilot and
physics disabled. Traffic Manager and sensor-based agents can see these stationary
vehicles and decide whether the intersection exit is clear. METS-R continues
moving them; Scenic refreshes the obstacle poses before the next CARLA step and
keeps them fixed throughout that step's CARLA substeps. A batched `query_vehicle`
with `transform_coords=True` supplies current positions and headings, since the
bridge's `coordinateTrail` contains upcoming route points rather than current
vehicle poses.

Scenic uses the original blueprint for vehicles belonging to the Scenic scenario,
and an available CARLA car or bus blueprint for other METS-R vehicles. Private and
public IDs are tracked separately. Boundary actors are removed when they leave the
query, before promotion into a newly controlled road, and during teardown. They
are excluded from Scenic vehicle control, route planning, and METS-R teleport
updates. Invalid pose responses or a rejected obstacle spawn stop the simulation
before the next CARLA tick instead of presenting an occupied exit as clear.


## Destination-based CARLA routing

At co-simulation spawn, Scenic queries METS-R to select a reachable destination
and passes that destination together with the sampled pose to `initializeCoSimVeh`.
METS-R creates its own route during initialization. Scenic does not send an
`updateVehicleRoute` command for either physical-road or connector spawns.
PCLA follows its configured or scenario-generated route in CARLA, while METS-R
mirrors the resulting ego pose.

METS-R's `routeRoadIds` contains upcoming roads and excludes the current road.
Default CARLA Traffic Manager vehicles use `destinationRoadId` from the ordinary
vehicle query as their goal. CARLA's route planner chooses a path from the actual
CARLA pose to a reachable driving lane on that road, including its own intermediate
roads and lane changes. The adapter does not require matching METS-R's remaining
road sequence or finding a SUMO lane chain for it. An explicit Scenic trajectory
continues to be honored.

If METS-R changes a default vehicle's destination, Scenic replaces its CARLA path
before the next CARLA tick. Controlled movement and outbound handoffs are checked
against physical road/connector topology, independently of METS-R's suggested
route. METS-R discards route assertions while mirroring CARLA-owned poses and
rebuilds its native route to the existing destination when control returns to it.
Boundary vehicles remain static CARLA obstacles under METS-R motion control.
