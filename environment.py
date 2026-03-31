import math
from config import *
from ctrnn import CTRNN

class VisualObject:
    def __init__(self, cx=0.0, cy=275.0, vy=-3.0, size=30.0):
        self.cx = cx
        self.cy = cy
        self.vy = vy
        self.size = size
        self.type = "unknown"

    def reset(self):
        self.cx = 0.0
        self.cy = 275.0

    def step(self, stepsize):
        self.cy += stepsize * self.vy

    def ray_intersection(self, ray):
        pass # To be implemented by subclasses

class Circle(VisualObject):
    def __init__(self, cx=0.0, cy=275.0, vy=-3.0, size=30.0):
        super().__init__(cx, cy, vy, size)
        self.type = "circle"

    def ray_intersection(self, ray):
        if ray['m'] == float('inf'):
            if abs(ray['startX'] - self.cx) > self.size / 2:
                return
            A = 1
            B = -2 * self.cy
            C = self.cy**2 - (self.size/2)**2 + (ray['startX'] - self.cx)**2
            disc = B**2 - 4*A*C
            if disc >= 0:
                y_int = (-B - math.sqrt(disc)) / (2*A)
                new_len = abs(y_int - ray['startY'])
                if new_len < ray['length']:
                    ray['length'] = new_len
            return

        A = 1 + ray['m']**2
        B = 2 * (ray['m'] * (ray['b'] - self.cy) - self.cx)
        C = self.cx**2 + ray['b']**2 - 2*ray['b']*self.cy + self.cy**2 - (self.size/2)**2
        disc = B**2 - 4*A*C

        if disc < 0:
            return

        x1 = (-B + math.sqrt(disc)) / (2*A)
        y1 = ray['m'] * x1 + ray['b']
        d1 = math.sqrt((x1 - ray['startX'])**2 + (y1 - ray['startY'])**2)

        x2 = (-B - math.sqrt(disc)) / (2*A)
        y2 = ray['m'] * x2 + ray['b']
        d2 = math.sqrt((x2 - ray['startX'])**2 + (y2 - ray['startY'])**2)

        new_len = min(d1, d2)
        if new_len < ray['length']:
            ray['length'] = new_len

class Line(VisualObject):
    def __init__(self, cx=0.0, cy=275.0, vy=-3.0, size=30.0):
        super().__init__(cx, cy, vy, size)
        self.type = "line"

    def ray_intersection(self, ray):
        # A horizontal line segment
        if ray['m'] == float('inf'):
            if abs(ray['startX'] - self.cx) <= self.size / 2:
                dist = abs(self.cy - ray['startY'])
                if dist < ray['length']:
                    ray['length'] = dist
            return

        x_int = (self.cy - ray['b']) / ray['m']
        if abs(x_int - self.cx) <= self.size / 2:
            dist = math.sqrt((x_int - ray['startX'])**2 + (self.cy - ray['startY'])**2)
            if dist < ray['length']:
                ray['length'] = dist

class VisualAgent:
    def __init__(self):
        self.cx = -50.0
        self.cy = 0.0
        self.vx = 0.0
        self.ctrnn = CTRNN()
        self.rays = []

    def reset(self):
        self.cx = -50.0
        self.cy = 0.0
        self.vx = 0.0
        if self.ctrnn.size > 0:
            self.ctrnn.states.fill(0.0)
            self.ctrnn.outputs = self.ctrnn.gains * self.ctrnn.biases # reset calculation

    def calculate_rays(self):
        self.rays = []
        theta = -VISUAL_ANGLE / 2
        for _ in range(NUM_RAYS):
            ray = {'length': MAX_RAY_LENGTH}
            if abs(theta) < 1e-7:
                ray['m'] = float('inf')
                ray['startX'] = self.cx
                ray['startY'] = self.cy + BODY_SIZE / 2
                ray['b'] = 0 
            else:
                ray['m'] = 1.0 / math.tan(theta)
                ray['startX'] = self.cx + (BODY_SIZE / 2) * math.sin(theta)
                ray['startY'] = self.cy + (BODY_SIZE / 2) * math.cos(theta)
                ray['b'] = self.cy - ray['m'] * self.cx
            self.rays.append(ray)
            theta += VISUAL_ANGLE / (NUM_RAYS - 1)

    def step(self, stepsize, obj):
        self.calculate_rays()
        
        for i, ray in enumerate(self.rays):
            obj.ray_intersection(ray)
            ext_in = INPUT_GAIN * (MAX_RAY_LENGTH - ray['length']) / MAX_RAY_LENGTH
            if self.ctrnn.size > i:
                self.ctrnn.externalinputs[i] = ext_in

        if self.ctrnn.size > 0:
            self.ctrnn.euler_step(stepsize)
            if self.ctrnn.size > 13:
                self.vx = VEL_GAIN * (self.ctrnn.outputs[12] - self.ctrnn.outputs[13])
        
        self.cx += stepsize * self.vx
        if self.cx < -ENV_WIDTH / 2:
            self.cx = -ENV_WIDTH / 2
        elif self.cx > ENV_WIDTH / 2:
            self.cx = ENV_WIDTH / 2
