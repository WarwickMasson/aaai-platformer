'''
This file implements a soccer simulator based on the robocup soccer simulator,
but without sensory perceptions, networking, and real-time operation.
'''
import numpy as np
from numpy.random import uniform
from numpy.linalg import norm
from util import vector

def bound(value, lower, upper):
    ''' Clips off a value which exceeds the lower or upper bounds. '''
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value

def bound_vector(vect, maximum):
    ''' Bounds a vector between a negative and positive maximum range. '''
    xval = bound(vect[0], -maximum, maximum)
    yval = bound(vect[1], -maximum, maximum)
    return vector(xval, yval)

MIN_PLATWIDTH = 100
MAX_PLATWIDTH = 300
PLATHEIGHT = 40
MIN_GAP = 100
MAX_GAP = 200
HEIGHT_DIFF = 100
MIN_SPIKES = 200
MAX_SPIKES = 300
SPIKES_HEIGHT = 10
MAX_ACCEL = 5000
MAX_SPEED = 5000
MAX_TIME = 10000
MAX_WIDTH = 100000


class Platform:

    def __init__(self, position):
        self.position = position
        self.size = vector(uniform(MIN_PLATWIDTH, MAX_PLATWIDTH), PLATHEIGHT)

class Simulator:
    ''' This class represents the environment. '''

    dt = 0.01

    def __init__(self):
        ''' The entities are set up and added to a space. '''
        self.player = Player()
        self.platform1 = Platform(vector(0.0, 0.0))
        self.gap = uniform(MIN_GAP, MAX_GAP)
        self.platform2 = Platform(vector(self.gap + self.platform1.size[0], uniform(-HEIGHT_DIFF, HEIGHT_DIFF)))
        self.enemy1 = Enemy(self.platform1, 0)
        self.enemy2 = Enemy(self.platform2, 0)
        self.spikes = Platform(vector(self.platform1.size[0], self.platform1.position[1] + uniform(MIN_SPIKES, MAX_SPIKES)))
        self.spikes.size = vector(self.gap, SPIKES_HEIGHT)
        self.floor = Platform(vector(0, -2*HEIGHT_DIFF))
        self.floor.size = vector(MAX_WIDTH, SPIKES_HEIGHT)
        self.states = []
        self.time = 0.0

    def regenerate_platforms(self):
        self.platform1 = self.platform2
        self.enemy1 = self.enemy2
        self.gap = uniform(MIN_GAP, MAX_GAP)
        self.platform2 = Platform(vector(self.gap + self.platform1.size[0] + self.platform1.position[0],
            self.platform1.position[1] + uniform(-HEIGHT_DIFF, HEIGHT_DIFF)))
        self.enemy2 = Enemy(self.platform2, 0)
        self.spikes = Platform(vector(self.platform1.position[0] + self.platform1.size[0], 
            self.platform1.position[1] + uniform(MIN_SPIKES, MAX_SPIKES)))
        self.spikes.size = vector(self.gap, SPIKES_HEIGHT)
        self.floor = Platform(vector(self.platform1.position[0], self.platform1.position[1] - 2*HEIGHT_DIFF))
        self.floor.size = vector(MAX_WIDTH, SPIKES_HEIGHT)

    def get_state(self):
        ''' Returns the representation of the current state. '''
        centre = self.platform1.position
        state = np.array([
            self.player.position[0] - centre[0],
            self.player.velocity[0],
            self.platform1.size[0],
            self.gap,
            self.spikes.position[1] - centre[1],
            self.platform2.position[1] - centre[1],
            self.platform2.size[1],
            self.time])
        return state

    def on_platforms(self):
        return self.player.on_platform(self.platform1) or self.player.on_platform(self.platform2)

    def perform_action(self, action, agent):
        ''' Applies for selected action for the given agent. '''
        if self.on_platforms():
            if action:
                act, parameters = action
                if act == 'jump':
                    self.player.jump(parameters)
                elif act == 'run':
                    self.player.run(parameters)
        else:
            self.player.fall()

    def terminal_check(self):
        ''' Determines if the episode is ended, and the reward. '''
        end_episode = False
        reward = 1.0
        for entity in [self.enemy1, self.enemy2, self.spikes, self.floor]:
            if self.player.colliding(entity):
                end_episode = True
                reward = -10.0
        return reward, end_episode

    def update(self, action):
        ''' Performs a single transition with the given action,
            then returns the new state and a reward. '''
        self.time += self.dt
        print self.time
        self.states.append([self.player.position])
        self.perform_action(action, self.player)
        for entity in [self.player, self.enemy1, self.enemy2]:
            entity.update(self.time)
        for platform in [self.platform1, self.platform2]:
            if self.player.colliding(platform):
                self.player.decollide(platform)
        if self.player.above_platform(self.platform2):
            self.regenerate_platforms()
        return self.terminal_check()

    def take_action(self, action):
        ''' Take a full, stabilised update. '''
        end_episode = False
        run = True
        while run:
            reward, end_episode = self.update(action)
            run = not end_episode
            if action and run:
                act, params = action
                if act == "run":
                    run = False
                elif act == "jump":
                    run = not self.on_platforms()
            action = None
        state = self.get_state()
        return state, reward, end_episode

class Enemy:

    dx = 10.0
    size = vector(20.0, 30.0)

    def __init__(self, platform, time):
        self.platform = platform
        self.set_position(time)

    def update(self, time):
        self.set_position(time)

    def set_position(self, time):
        period = (self.platform.size[0] - self.size[0]) / self.dx
        time = time % (2*period)
        xpos = self.platform.position[0]
        if time < period:
            xpos += self.platform.size[0]-self.size[0] - time * self.dx
        else:
            xpos += (time - period) * self.dx
        self.position = vector(xpos, self.platform.position[1] + self.platform.size[1])

class Player(Enemy):

    decay = 1.0

    def __init__(self):
        self.position = vector(0, self.size[1])
        self.velocity = vector(0.0, 0.0)

    def update(self, time):
        ''' Update the position and velocity. '''
        self.position += self.velocity * Simulator.dt

    def accelerate(self, accel):
        ''' Applies a power to the entity in direction theta. '''
        accel = bound_vector(accel, MAX_ACCEL)
        self.velocity += accel * (Simulator.dt ** 2)
        self.velocity = bound_vector(self.velocity, MAX_SPEED)

    def run(self, power):
        self.velocity[0] *= self.decay
        self.accelerate(vector(power, 0.0))

    def jump(self, power):
        self.accelerate(vector(0.0, power))

    def fall(self):
        self.accelerate(vector(0.0, -9.8))

    def decollide(self, other):
        ''' Shift overlapping entities apart. '''
        precorner = other.position - self.size
        postcorner = other.position + other.size
        newx, newy = self.position[0], self.position[1]
        if self.position[0] < other.position[0]:
            newx = precorner[0]
        elif self.position[0] > postcorner[0] - self.size[0]:
            newx = postcorner[0]
        if self.position[1] < other.position[1]:
            newy = precorner[1]
        elif self.position[1] > postcorner[1] - self.size[1]:
            newy = postcorner[1]
        if newx == self.position[0]:
            self.velocity[1] = 0.0
            self.position[1] = newy
        elif newy == self.position[1]:
            self.velocity[0] = 0.0
            self.position[0] = newx
        elif abs(self.position[0] - newx) < abs(self.position[1] - newy):
            self.velocity[0] = 0.0
            self.position[0] = newx
        else:
            self.velocity[1] = 0.0
            self.position[1] = newy

    def above_platform(self, platform):
        return 0.0 <= self.position[0] - platform.position[0] <= platform.size[0]

    def on_platform(self, platform):
        ony = self.position[1] - platform.position[1] == platform.size[1]
        return self.above_platform(platform) and ony

    def colliding(self, other):
        ''' Check if two entities are overlapping. '''
        precorner = other.position - self.size
        postcorner = other.position + other.size
        collide = (precorner < self.position).all()
        collide = collide and (self.position < postcorner).all()
        return collide
