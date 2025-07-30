import numpy as np
import random
""" 
    Takes the current state of the simulation and maps to the corresponding motor velocityes

    state: dict containing 
        - current motor velocity (left & right)
        - sensor data (7,) - 
            4 front
            2 side
            1 back
"""
class generic_controller():
    def __init__(self):

        self.visited = []

        self.threshold = random.random()

        self.FAST = 15
        self.MED  = 8
        self.SLOW = 4

    def forward(self,rate):
        return [rate,rate]

    def turn_right(self,rate):
        return [rate,0]

    def turn_left(self,rate):
        return [0,rate]

    def reverse(self, rate):
        return [-rate,-rate]

    def front_collision(self,front_sensors):
        if np.any(front_sensors < .35):
            return True
        
    def rear_collsion(self,back_sensor):
        if back_sensor < .1:
            return True

    def left_collsion(self,left_sensor):
        if left_sensor < .1:
            return True
        
    def right_collsion(self,right_sensor):
        if right_sensor < .2:
            return True
        
    def upcoming_object(self,front_sensors):
        if np.any(front_sensors < .7):
            return True

    def apply_control(self,state):
        front_sensors = np.array(state["sensor"][:3])
        back_sensor   = state["sensor"][4]
        left_sensor   = state["sensor"][5]
        right_sensor  = state["sensor"][6]
       
        pos = list(state["position"])
 
        if pos not in self.visited:
            self.visited.append(pos)
            if not self.front_collision(front_sensors):
                if self.right_collsion(right_sensor):
                    return self.turn_right(self.MED)
                elif self.left_collsion(left_sensor):
                    return self.turn_left(self.MED)
                elif not self.upcoming_object(front_sensors):
                    return self.forward(self.FAST)
                else:
                    return self.forward(self.SLOW)
            elif self.front_collision(front_sensors):
                if not self.right_collsion(right_sensor):
                    return self.turn_left(self.FAST)
                elif not self.left_collsion(left_sensor):
                    return self.turn_right(self.FAST)
                else:
                    if back_sensor > .5:
                        return self.reverse(self.FAST)
                    else: 
                        return self.reverse(self.SLOW)
        else: # if the location is not new try turning to a safe area
              # adding some randomness to this to preven the model from infintely spinning
            choice1 = random.random()
            choice12= random.random()
            if not self.right_collsion(right_sensor) and not self.left_collsion(left_sensor) and not self.front_collision(front_sensors):
                if choice1 > .5:
                    if choice12 > self.threshold:
                        return self.turn_left(self.FAST)
                    else:
                        return self.turn_right(self.FAST)
                else:
                    return self.forward(self.MED)
            elif not self.left_collsion(left_sensor):
                return self.turn_right(self.FAST)
            elif not self.right_collsion(right_sensor):
                return self.turn_left(self.FAST)
            else:
                return(self.reverse(self.MED))
            