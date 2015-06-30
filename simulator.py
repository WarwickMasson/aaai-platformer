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

MIN_PLATWIDTH = 300.0
MAX_PLATWIDTH = 400.0
PLATHEIGHT = 40.0
MIN_GAP = 150.0
MAX_GAP = 200.0
HEIGHT_DIFF = 50.0
MAX_WIDTH = 3*MAX_PLATWIDTH + 2*MAX_GAP
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
        gap2 = uniform(MIN_GAP, MAX_GAP)
        self.platform3 = Platform(self.platform2.position + vector(self.gap + self.platform2.size[0], uniform(-HEIGHT_DIFF, HEIGHT_DIFF)))
        self.enemy1 = Enemy(self.platform1)
        self.enemy2 = Enemy(self.platform2)
        self.states = []

    def get_state(self):
        ''' Returns the representation of the current state. '''
        if self.player.position[0] > self.platform2.position[0]:
            plat1 = self.platform2
            plat2 = self.platform3
        else:
            plat1 = self.platform1
            plat2 = self.platform2
        state = np.array([
            self.player.position[0],    #0
            self.player.velocity[0],    #1
            self.enemy1.position[0],    #2
            self.enemy1.dx,             #3
            plat1.position[0],          #4
            plat1.size[0],              #5
            plat2.position[0],          #6
            plat2.position[1],          #7
            plat2.size[0]])             #8
        return state

    def on_platforms(self):
        return self.player.on_platform(self.platform1) or self.player.on_platform(self.platform2) or self.player.on_platform(self.platform3)

    def perform_action(self, action, agent, dt = DT):
        ''' Applies for selected action for the given agent. '''
        if self.on_platforms():
            if action:
                act, parameters = action
                if act == 'jump':
                    self.player.jump(parameters)
                elif act == 'run':
                    self.player.run(parameters, dt)
        else:
            self.player.fall()

    def lower_bound(self):
        ''' Returns the lower bound of the platforms. '''
        lower = min(self.platform1.position[1], self.platform2.position[1], self.platform3.position[1])
        return lower - self.platform1.size[1]

    def right_bound(self):
        return self.platform3.position[0] + self.platform3.size[0]

    def terminal_check(self, reward = 0.0):
        ''' Determines if the episode is ended, and the reward. '''
        end_episode = self.player.position[1] < self.lower_bound()
        right = self.player.position[0] >= self.right_bound()
        for entity in [self.enemy1, self.enemy2]:
            if self.player.colliding(entity):
                end_episode = True
        if right:
            reward = (self.right_bound() - self.xpos) / self.right_bound()
            end_episode = True
        return reward, end_episode

    def update(self, action, dt = DT):
        ''' Performs a single transition with the given action,
            then returns the new state and a reward. '''
        self.states.append([self.player.position.copy(),
                            self.enemy1.position.copy(),
                            self.enemy2.position.copy()])
        self.perform_action(action, self.player, dt)
        if self.player.position[0] > self.platform2.position[0]:
            self.enemy1, self.enemy2 = self.enemy2, self.enemy1
        for entity in [self.player, self.enemy1]:
            entity.update(dt)
        for platform in [self.platform1, self.platform2, self.platform3]:
            if self.player.colliding(platform):
                self.player.decollide(platform)
        reward = (self.player.position[0] - self.xpos) / self.right_bound()
        return self.terminal_check(reward)

    def take_action(self, action):
        ''' Take a full, stabilised update. '''
        end_episode = False
        run = True
        act, params = action
        self.xpos = self.player.position[0]
        step = 0
        while run:
            if act == "run":
                diff = DT
                if params < DT:
                    diff =  params 
                reward, end_episode = self.update(('run', 2.0), diff)
                params -= diff
                run = params > 0
            elif act == "jump":
                reward, end_episode = self.update(action)
                run = not self.on_platforms()
                action = None
            if end_episode:
                run = False
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

    def update(self, dt):
        self.position += vector(self.dx * dt, 0)
        if not (0 <= self.position[0] - self.platform.position[0] <= self.platform.size[0] - self.size[0]):
            self.dx *= -1

class Player(Enemy):

    decay = 1.0

    def __init__(self):
        self.position = vector(0, PLATHEIGHT)
        self.velocity = vector(0.0, 0.0)

    def update(self, dt):
        ''' Update the position and velocity. '''
        self.position += self.velocity * dt

    def accelerate(self, accel, dt = DT):
        ''' Applies a power to the entity in direction theta. '''
        accel = bound_vector(accel, MAX_ACCEL)
        self.velocity += accel * dt
        self.velocity = bound_vector(self.velocity, MAX_SPEED)

    def run(self, power, dt):
        self.velocity[0] *= self.decay
        if dt > 0:
            self.accelerate(vector(power / dt, 0.0), dt)

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
