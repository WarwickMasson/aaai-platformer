'''
This file implements a simple pygame interface for the simulator.
The simulator can be controlled with the keyboard, or updated without controls.
'''
import numpy as np
from simulator import Simulator
import pygame
import sys
from util import vector_to_tuple, vector

WIDTH = 300
LENGTH = 750

class Interface:
    ''' Implements a pygame interface that allows keyboard control
        of the player, and draws the field, players, and ball. '''

    def __init__(self, simulator = Simulator()):
        ''' Sets up the colors, pygame, and screen. '''
        pygame.init()
        size = (LENGTH, WIDTH)
        self.window = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.simulator = simulator
        self.background = pygame.Surface(size)
        self.white = pygame.Color(255, 255, 255, 0)
        self.black = pygame.Color(0, 0, 0, 0)
        self.green = pygame.Color(0, 255, 0, 0)
        self.red = pygame.Color(255, 0, 0, 0)
        self.yellow = pygame.Color(255, 255, 0, 0)
        self.background.fill(self.black)

    def control_update(self):
        ''' Uses input from the keyboard to control the player. '''
        keys_pressed = pygame.key.get_pressed()
        action_map = {
            pygame.K_SPACE: ('jump', 3000),
            pygame.K_d: ('run', 100),
        }
        action = None
        for key in action_map:
            if keys_pressed[key]:
                action = action_map[key]
                break
        reward, end_episode = self.simulator.update(action)
        if end_episode:
            self.simulator = Simulator()

    def update(self):
        ''' Performs a single 100ms update. '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.control_update()
        self.draw()
        self.clock.tick(100)

    def draw(self):
        ''' Draw the field and players. '''
        self.window.blit(self.background, (0, 0))
        self.centre = -self.simulator.player.position + vector(LENGTH, WIDTH)/2
        self.draw_entity(self.simulator.player, self.green)
        self.draw_entity(self.simulator.enemy1, self.red)
        self.draw_entity(self.simulator.enemy2, self.red)
        self.draw_entity(self.simulator.platform1, self.white)
        self.draw_entity(self.simulator.platform2, self.white)
        self.draw_entity(self.simulator.spikes, self.yellow)
        surf = pygame.transform.flip(self.window, False, True)
        self.window.blit(surf, (0, 0))
        pygame.display.update()

    def draw_entity(self, entity, colour):
        ''' Draws an entity as a rectangle. '''
        rect = pygame.Rect(entity.position[0] + self.centre[0], entity.position[1] + self.centre[1], entity.size[0], entity.size[1])
        pygame.draw.rect(self.window, colour, rect)

def main():
    ''' Runs the interface. '''
    interface = Interface()
    while True:
        interface.update()

if __name__ == '__main__':
    main()
