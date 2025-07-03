from scenic.simulators.mujoco.model import *


arena = RectangularRegion(position = (0,0,0),heading=1, width=5, length=5)


ego = new Pusher in arena


cone = new MujocoBody in arena,
    with color (0.75, 0.5, 0.5, 1),
    with width .25,
    with length .5,
    with height .5,
    with shape Uniform(SpheroidShape(), BoxShape(), CylinderShape())