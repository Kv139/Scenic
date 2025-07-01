from scenic.simulators.mujoco.model import *

ego = new Pusher


cone = new MujocoBody at (2, 0, 20),
    with color (0.75, 0.5, 0.5, 1),
    with width 1,
    with length 1,
    with height 1,
    with shape Uniform(SpheroidShape(), BoxShape(), CylinderShape())
