"""
Generate a room for the i-roomba create vacuum
"""
from scenic.core.external_params import VerifaiRange

from vacuum_lib import *
from itertools import combinations
no_verifai = True
param verifaiSamplerType = 'mab'
dnev = 1.59576912 # double normal ev, |[-dnev, dnev]| has the same ev as |N(0, 1)|
dpi = 6.28318531

## Scene Layout ##

# Create room region and set it as the workspace
room_region = RectangularRegion(0 @ 0, 0, 5.09, 5.09)
workspace = Workspace(room_region)

# Create floor and walls
floor = new Floor
wall_offset = floor.width/2 + 0.04/2 + 1e-4
right_wall = new Wall at (wall_offset, 0, 0.25), facing toward floor
left_wall = new Wall at (-wall_offset, 0, 0.25), facing toward floor
front_wall = new Wall at (0, wall_offset, 0.25), facing toward floor
back_wall = new Wall at (0, -wall_offset, 0.25), facing toward floor

if no_verifai:
    # Place vacuum on floor
    ego = new Vacuum on floor
    record (ego.x, ego.y, ego.z) as EgoPosition

    # Create a "safe zone" around the vacuum so that it does not start stuck
    safe_zone = CircularRegion(ego.position, radius=1)

    # # Create a dining room region where we will place dining room furniture
    dining_room_region = RectangularRegion(1.25 @ 0, 0, 5, 5).difference(safe_zone)

    dining_table = new DiningTable on floor, facing Range(0, 360 deg), with size .1

    chair_1 = new DiningChair on floor, behind dining_table by -.1, 
                    facing toward dining_table, with regionContainedIn dining_room_region
    chair_3 = new DiningChair on floor, left of dining_table by -.1, 
                    facing toward dining_table, with regionContainedIn dining_room_region
    #Add some noise to the positions and yaw of the chairs around the table
    mutate chair_1, chair_3

    # Create a living room region where we will place living room furniture
    living_room_region = RectangularRegion(-1.25 @ 0, 0, 5, 5).difference(safe_zone)

    couch = new Couch on floor, ahead of left_wall by 0.335, 
            facing away from left_wall, with regionContainedIn living_room_region

    coffee_table = new CoffeeTable on floor, ahead of couch by 0.336, 
            facing away from couch, with regionContainedIn living_room_region

    # # Add some noise to the positions of the couch and coffee table
    mutate couch, coffee_table

    # Spawn some toys
    for _ in range(3):
        new Toy on floor
else:
    # Place vacuum on floor
    ego = new Vacuum at (VerifaiRange(-2.5, 2.5) @ VerifaiRange(-2.5, 2.5)), on floor
    record (ego.x, ego.y, ego.z) as EgoPosition

    # Create a "safe zone" around the vacuum so that it does not start stuck
    safe_zone = CircularRegion(ego.position, radius=1)

    # # Create a dining room region where we will place dining room furniture
    dining_room_region = RectangularRegion(1.25 @ 0, 0, 5, 5).difference(safe_zone)

    dining_table = new DiningTable on floor, at (VerifaiRange(-2.5, 2.5) @ VerifaiRange(-2.5, 2.5)), with size .1
    ideal_pos = new OrientedPoint behind dining_table by -0.1, facing toward dining_table
    chair_1 = new DiningChair on floor, 
                facing VerifaiRange((-dnev*10) deg, (dnev*10) deg) relative to ideal_pos,
                at ideal_pos offset by (VerifaiRange(-dnev*.05, dnev*.05) @ VerifaiRange(-dnev*.05, dnev*.05)),
                with regionContainedIn dining_room_region     
    ideal_pos = new OrientedPoint left of dining_table by -0.1, facing toward dining_table
    chair_3 = new DiningChair on floor, 
                facing VerifaiRange((-dnev*10) deg, (dnev*10) deg) relative to ideal_pos,
                at ideal_pos offset by (VerifaiRange(-dnev*.05, dnev*.05) @ VerifaiRange(-dnev*.05, dnev*.05)), 
                with regionContainedIn dining_room_region

    # # Create a living room region where we will place living room furniture
    living_room_region = RectangularRegion(-1.25 @ 0, 0, 5, 5).difference(safe_zone)
    ideal_pos = new OrientedPoint ahead of left_wall by 0.335, facing away from left_wall
    couch = new Couch on floor, facing VerifaiRange((-dnev*5) deg, (dnev*5) deg) relative to ideal_pos, with regionContainedIn living_room_region,
                at ideal_pos offset by (VerifaiRange(-dnev*.05, dnev*.05) @ VerifaiRange(-dnev*.5, dnev*.5))
    ideal_pos = new OrientedPoint ahead of couch by 0.336, facing away from couch
    coffee_table = new CoffeeTable on floor, facing VerifaiRange((-dnev*5) deg, (dnev*5) deg) relative to ideal_pos, with regionContainedIn living_room_region,
                at ideal_pos offset by (VerifaiRange(-dnev*.05, dnev*.05) @ VerifaiRange(-dnev*.05, dnev*.05))

    new Toy on floor, at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10)
    new Toy on floor, at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10)
    new Toy on floor, at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10)

# import random

# big_objects = [
#     lambda: new DiningTable at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#             with size 0.1,
#             facing Range(0, 360 deg),

#     lambda: new Couch at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),

#     lambda: new CoffeeTable at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),
# ]
# small_objects = [
#     lambda: new DiningChair at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),

#     lambda: new DiningChair
#                 at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#                 facing Range(0, 360 deg),

#     lambda: new BlockToy at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#             facing Range(0, 360 deg),

#     lambda: new BlockToy at (VerifaiRange(-2.5, 2.5), VerifaiRange(-2.5, 2.5), 10), on Floor,
#             facing Range(0, 360 deg),
# ]

# chosen_big_objects = random.sample(big_objects, 2)
# chosen_small_objects =  random.sample(small_objects, 2)

# walls = [right_wall, left_wall, front_wall, back_wall]
# objects = [ego]
# for obj in chosen_big_objects:
#     objects.append(obj())
# for obj in chosen_small_objects:
#     objects.append(obj())

# wallDist = 0.335
# objDist = 0.1
# require (
#   all(not (a.occupiedSpace._bufferOverapproximate(wallDist,1).intersects(b.occupiedSpace)) for a in walls for b in objects)
#   and
#   all(not (a.occupiedSpace._bufferOverapproximate(objDist,1).intersects(b.occupiedSpace)) for a, b in combinations(objects, 2))
# )


