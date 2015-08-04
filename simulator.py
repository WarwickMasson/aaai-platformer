'''
This file implements the platformer rules.
'''
import numpy as np
from numpy.random import uniform
from util import vector

def bound(value, lower, upper):
    ''' Clips off a value which exceeds the lower or upper bounds. '''
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value

def bound_vector(vect, xmax, ymax):
    ''' Bounds a vector between a negative and positive maximum range. '''
    xval = bound(vect[0], -xmax, xmax)
    yval = bound(vect[1], -ymax, ymax)
    return vector(xval, yval)

WIDTH1 = 250
WIDTH2 = 325
WIDTH3 = 100
GAP1 = 100
GAP2 = 150
HEIGHT1 = 0.0
HEIGHT2 = HEIGHT1 + 15.0
HEIGHT3 = HEIGHT2 - 5.0
MAX_PLATWIDTH = max(WIDTH1, WIDTH2, WIDTH3)
PLATHEIGHT = 40.0
MAX_WIDTH = WIDTH1 + WIDTH2 + WIDTH3 + GAP1 + GAP2
MAX_GAP = max(GAP1, GAP2)
MAX_DIFF = max(HEIGHT1, HEIGHT2, HEIGHT3)
DT = 0.05
MAX_DX = 100.0
MAX_DY = 50.0
MAX_DX_ON = 50.0
MAX_DDX = 50.0 / DT
MAX_DDY = MAX_DY / DT
ENEMY_SPEED = 30.0
LEAP_DEV = 5.0
HOP_DEV = 1.0
ENEMY_NOISE = 0.5
CHECK_SCALE = False

def scale_state(state):
    ''' Scale state variables between 0 and 1. '''
    new_state = np.copy(state)
    scaled = (new_state + SHIFT_VECTOR) / SCALE_VECTOR
    if CHECK_SCALE:
        for i in range(scaled.size):
            if not 0 <= scaled[i] <= 1:
                print i, scaled[i], new_state[i]
                assert 0 <= scaled[i] <= 1
    return scaled

def platform_features(state):
    ''' Compute the implicit features of the platforms. '''
    xpos = state[0]
    if xpos < WIDTH1 + GAP1:
        wd1 = WIDTH1
        wd2 = WIDTH2
        gap = GAP1
        diff = HEIGHT2 - HEIGHT1
    elif xpos < WIDTH1 + GAP1 + WIDTH2 + GAP2:
        wd1 = WIDTH2
        wd2 = WIDTH3
        gap = GAP2
        diff = HEIGHT3 - HEIGHT2
    else:
        wd1 = WIDTH3
        wd2 = 0.0
        gap = 0.0
        diff = 0.0
    return [wd1 / MAX_PLATWIDTH, wd2 / MAX_PLATWIDTH, gap / MAX_GAP, diff / MAX_DIFF]

class Platform:
    ''' Represents a fixed platform. '''

    def __init__(self, xpos, width, height):
        self.position = vector(xpos, height)
        self.size = vector(width, PLATHEIGHT)

class Simulator:
    ''' This class represents the environment. '''


    def __init__(self):
        ''' The entities are set up and added to a space. '''
        self.xpos = 0.0
        self.player = Player()
        self.platform1 = Platform(0.0, WIDTH1, HEIGHT1)
        self.platform2 = Platform(GAP1 + self.platform1.size[0], WIDTH2, HEIGHT2)
        self.platform3 = Platform(self.platform2.position[0] +
            GAP2 + self.platform2.size[0], WIDTH3, HEIGHT3)
        self.enemy1 = Enemy(self.platform1)
        self.enemy2 = Enemy(self.platform2)
        self.states = []

    def get_state(self):
        ''' Returns the representation of the current state. '''
        if self.player.position[0] > self.platform2.position[0]:
            enemy = self.enemy2
        else:
            enemy = self.enemy1
        state = np.array([
            self.player.position[0],    #0
            self.player.velocity[0],    #1
            enemy.position[0],          #2
            enemy.dx])                  #3
        return state

    def on_platforms(self):
        ''' Checks if the player is on any of the platforms. '''
        for platform in [self.platform1, self.platform2, self.platform3]:
            if self.player.on_platform(platform):
                return True
        return False

    def perform_action(self, action, dt=DT):
        ''' Applies for selected action for the given agent. '''
        if self.on_platforms():
            if action:
                act, parameters = action
                if act == 'jump':
                    self.player.jump(parameters)
                elif act == 'run':
                    self.player.run(parameters, dt)
                elif act == 'leap':
                    self.player.leap_to(parameters)
                elif act == 'hop':
                    self.player.hop_to(parameters)
        else:
            self.player.fall()

    def lower_bound(self):
        ''' Returns the lowest height of the platforms. '''
        lower = min(self.platform1.position[1], self.platform2.position[1], self.platform3.position[1])
        return lower - self.platform1.size[1]

    def right_bound(self):
        ''' Returns the edge of the game. '''
        return self.platform3.position[0] + self.platform3.size[0]

    def terminal_check(self, reward=0.0):
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

    def update(self, action, dt=DT, interface = False):
        ''' Performs a single transition with the given action,
            then returns the new state and a reward. '''
        if interface:
            self.xpos = self.player.position[0]
        self.states.append([self.player.position.copy(),
                            self.enemy1.position.copy(),
                            self.enemy2.position.copy()])
        self.perform_action(action, dt)
        if self.on_platforms():
            self.player.ground_bound()
        if self.player.position[0] > self.platform2.position[0]:
            enemy = self.enemy2
        else:
            enemy = self.enemy1
        for entity in [self.player, enemy]:
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
                    diff = params
                if params > 0:
                    reward, end_episode = self.update(('run', 2.0), diff)
                    params -= diff
                run = params > 0
            elif act in ['jump', 'hop', 'leap']:
                reward, end_episode = self.update(action)
                run = not self.on_platforms()
                action = None
            if end_episode:
                run = False
            step += 1
        state = self.get_state()
        return state, reward, end_episode, step

class Enemy:
    ''' Defines the enemy. '''

    size = vector(20.0, 30.0)

    def __init__(self, platform):
        ''' Initializes the enemy on the platform. '''
        self.dx = -ENEMY_SPEED
        self.platform = platform
        self.position = self.platform.size + self.platform.position
        self.position[0] -= self.size[0]

    def update(self, dt):
        ''' Shift the enemy along the platform. '''
        right = self.platform.position[0] + self.platform.size[0] - self.size[0]
        if not self.platform.position[0] < self.position[0] < right:
            self.dx *= -1
        self.dx += np.random.normal(0.0, ENEMY_NOISE*dt)
        self.dx = bound(self.dx, -ENEMY_SPEED, ENEMY_SPEED)
        self.position[0] += self.dx * dt
        self.position[0] = bound(self.position[0], self.platform.position[0], right)

class Player(Enemy):
    ''' Represents the player character. '''
    gravity = 9.8
    decay = 0.99

    def __init__(self):
        ''' Initialize the position to the starting platform. '''
        self.position = vector(0, PLATHEIGHT)
        self.velocity = vector(0.0, 0.0)

    def update(self, dt):
        ''' Update the position and velocity. '''
        self.position += self.velocity * dt
        self.position[0] = bound(self.position[0], 0.0, MAX_WIDTH)
        self.velocity[0] *= self.decay

    def accelerate(self, accel, dt=DT):
        ''' Applies a power to the entity in direction theta. '''
        accel = bound_vector(accel, MAX_DDX, MAX_DDY)
        self.velocity += accel * dt
        self.velocity[0] -= abs(np.random.normal(0.0, ENEMY_NOISE*dt))
        self.velocity = bound_vector(self.velocity, MAX_DX, MAX_DY)
        self.velocity[0] = max(self.velocity[0], 0.0)

    def ground_bound(self):
        ''' Bound dx while on the ground. '''
        self.velocity[0] = bound(self.velocity[0], 0.0, MAX_DX_ON)

    def run(self, power, dt):
        ''' Run for a given power and time. '''
        if dt > 0:
            self.accelerate(vector(power / dt, 0.0), dt)

    def jump(self, power):
        ''' Jump up for a single step. '''
        self.accelerate(vector(0.0, power / DT))

    def jump_to(self, diffx, dy0, dev):
        ''' Jump to a specific position. '''
        time = 2.0 * dy0 / self.gravity + 1.0
        dx0 = diffx / time - self.velocity[0]
        dx0 = bound(dx0, -MAX_DDX, MAX_DY - dy0)
        noise = -abs(np.random.normal(0.0, dev, 2))
        accel = vector(dx0, dy0) + noise
        self.accelerate(accel / DT)

    def hop_to(self, diffx):
        ''' Jump high to a position. '''
        self.jump_to(diffx, 35.0, HOP_DEV)

    def leap_to(self, diffx):
        ''' Jump over a gap. '''
        self.jump_to(diffx, 25.0, LEAP_DEV)

    def fall(self):
        ''' Apply gravity. '''
        self.accelerate(vector(0.0, -self.gravity))

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
        ''' Checks the player is above the platform. '''
        return -self.size[0] <= self.position[0] - platform.position[0] <= platform.size[0]

    def on_platform(self, platform):
        ''' Checks the player is standing on the platform. '''
        ony = self.position[1] - platform.position[1] == platform.size[1]
        return self.above_platform(platform) and ony

    def colliding(self, other):
        ''' Check if two entities are overlapping. '''
        precorner = other.position - self.size
        postcorner = other.position + other.size
        collide = (precorner < self.position).all()
        collide = collide and (self.position < postcorner).all()
        return collide

SHIFT_VECTOR = np.array([Player.size[0], 0.0, 0.0,
    ENEMY_SPEED])
SCALE_VECTOR = np.array([MAX_WIDTH + Player.size[0], MAX_DX,
    MAX_WIDTH, 2*ENEMY_SPEED])
STATE_DIM = Simulator().get_state().size
