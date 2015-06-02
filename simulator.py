'''
This file implements the platformer rules.
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

MIN_PLATWIDTH = 200.0
MAX_PLATWIDTH = 300.0
PLATHEIGHT = 40.0
MIN_GAP = 150.0
MAX_GAP = 200.0
HEIGHT_DIFF = 50.0
MIN_SPIKES = 250.0
MAX_SPIKES = 300.0
SPIKES_HEIGHT = 10.0
MAX_WIDTH = 100000.0
DT = 0.05
MAX_ACCEL = 50.0 / DT
MAX_SPEED = 50.0 / DT


class Platform:

    def __init__(self, position):
        self.position = position
        self.size = vector(uniform(MIN_PLATWIDTH, MAX_PLATWIDTH), PLATHEIGHT)

class Simulator:
    ''' This class represents the environment. '''

    xpos = 0.0

    def __init__(self):
        ''' The entities are set up and added to a space. '''
        self.player = Player()
        self.platform1 = Platform(vector(0.0, 0.0))
        self.gap = uniform(MIN_GAP, MAX_GAP)
        self.platform2 = Platform(vector(self.gap + self.platform1.size[0], uniform(-HEIGHT_DIFF, HEIGHT_DIFF)))
        self.enemy1 = Enemy(self.platform1)
        self.enemy2 = Enemy(self.platform2)
        self.spikes = Platform(vector(self.platform1.size[0], self.platform1.position[1] + uniform(MIN_SPIKES, MAX_SPIKES)))
        self.spikes.size = vector(self.gap, SPIKES_HEIGHT)
        self.states = []

    def regenerate_platforms(self):
        self.platform1 = self.platform2
        self.enemy1 = self.enemy2
        self.gap = uniform(MIN_GAP, MAX_GAP)
        self.platform2 = Platform(vector(self.gap + self.platform1.size[0] + self.platform1.position[0],
            self.platform1.position[1] + uniform(-HEIGHT_DIFF, HEIGHT_DIFF)))
        self.enemy2 = Enemy(self.platform2)
        self.spikes = Platform(vector(self.platform1.position[0] + self.platform1.size[0],
            self.platform1.position[1] + uniform(MIN_SPIKES, MAX_SPIKES)))
        self.spikes.size = vector(self.gap, SPIKES_HEIGHT)

    def get_state(self):
        ''' Returns the representation of the current state. '''
        state = np.array([
            self.player.position[0],
            self.player.velocity[0],
            self.platform1.size[0],
            self.gap,
            self.spikes.position[1],
            self.platform2.position[1],
            self.platform2.size[0],
            self.enemy1.position[0],
            self.enemy1.dx,
            self.platform1.position[0],
            self.platform1.position[1]])
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

    def lower_bound(self):
        ''' Returns the lower bound of the platforms. '''
        lower = min(self.platform1.position[1], self.platform2.position[1])
        return lower - self.platform1.size[1]

    def terminal_check(self, reward = 0.0):
        ''' Determines if the episode is ended, and the reward. '''
        end_episode = self.player.position[1] < self.lower_bound()
        for entity in [self.enemy1, self.enemy2, self.spikes]:
            if self.player.colliding(entity):
                end_episode = True
        return reward, end_episode

    def update(self, action):
        ''' Performs a single transition with the given action,
            then returns the new state and a reward. '''
        self.states.append([self.player.position.copy(),
                            self.platform1.position.copy(),
                            self.platform2.position.copy(),
                            self.enemy1.position.copy(),
                            self.enemy2.position.copy(),
                            self.spikes.position.copy(),
                            self.platform1.size.copy(),
                            self.platform2.size.copy(),
                            self.spikes.size.copy()])
        self.perform_action(action, self.player)
        for entity in [self.player, self.enemy1]:
            entity.update()
        for platform in [self.platform1, self.platform2]:
            if self.player.colliding(platform):
                self.player.decollide(platform)
        reward = self.player.position[0] - self.xpos
        if self.player.above_platform(self.platform2):
            self.regenerate_platforms()
        return self.terminal_check(reward)

    def take_action(self, action):
        ''' Take a full, stabilised update. '''
        end_episode = False
        run = True
        act, params = action
        self.xpos = self.player.position[0]
        step = 0
        while run:
            reward, end_episode = self.update(action)
            if act == "run":
                run = False
            elif act == "jump":
                run = not self.on_platforms()
            if end_episode:
                run = False
            action = None
            step += 1
        state = self.get_state()
        return state, reward, end_episode, step

class Enemy:

    size = vector(20.0, 30.0)

    def __init__(self, platform):
        self.dx = -20.0
        self.platform = platform
        self.position = self.platform.size + self.platform.position
        self.position[0] -= self.size[0]

    def update(self):
        self.position += vector(self.dx * DT, 0)
        if not (0 <= self.position[0] - self.platform.position[0] <= self.platform.size[0] - self.size[0]):
            self.dx *= -1

class Player(Enemy):

    decay = 1.0

    def __init__(self):
        self.position = vector(0, PLATHEIGHT)
        self.velocity = vector(0.0, 0.0)

    def update(self):
        ''' Update the position and velocity. '''
        self.position += self.velocity * DT

    def accelerate(self, accel):
        ''' Applies a power to the entity in direction theta. '''
        accel = bound_vector(accel, MAX_ACCEL)
        self.velocity += accel * DT
        self.velocity = bound_vector(self.velocity, MAX_SPEED)

    def run(self, power):
        self.velocity[0] *= self.decay
        self.accelerate(vector(power / DT, 0.0))

    def jump(self, power):
        self.accelerate(vector(0.0, power / DT))

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
