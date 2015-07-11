'''
This file implements a simple pygame interface for the simulator.
The simulator can be controlled with the keyboard, or updated without controls.
'''
from simulator import Simulator, DT, MAX_WIDTH
from learn import INITIAL_HOP, INITIAL_LEAP
import pygame
import sys
from util import vector

WIDTH = 500
LENGTH = int(MAX_WIDTH)

class Interface:
    ''' Implements a pygame interface that allows keyboard control
        of the player, and draws the field, players, and ball. '''

    def __init__(self, simulator=Simulator()):
        ''' Sets up the colors, pygame, and screen. '''
        pygame.init()
        size = (LENGTH, WIDTH)
        self.window = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.simulator = simulator
        self.background = pygame.image.load('./sprites/background.png')
        self.platform = pygame.image.load('./sprites/platform.png')
        self.enemy = pygame.image.load('./sprites/enemy.png')
        self.player = pygame.image.load('./sprites/player.png')
        self.centre = vector(0, WIDTH)/2
        self.total = 0.0

    def control_update(self):
        ''' Uses input from the keyboard to control the player. '''
        keys_pressed = pygame.key.get_pressed()
        action_map = {
            pygame.K_SPACE: ('hop', INITIAL_HOP),
            pygame.K_l: ('leap', INITIAL_LEAP),
            pygame.K_d: ('run', 2),
        }
        action = ('run', 0)
        for key in action_map:
            if keys_pressed[key]:
                action = action_map[key]
                break
        reward, end_episode = self.simulator.update(action, DT, True)
        self.total += reward
        if end_episode:
            print 'Episode Reward:', self.total
            self.total = 0.0
            self.simulator = Simulator()

    def update(self):
        ''' Performs a single 10ms update. '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.control_update()
        self.draw()
        self.clock.tick(1000*DT)

    def draw(self):
        ''' Draw the field and players. '''
        self.window.blit(self.background, (0, 0))
        self.draw_entity(self.simulator.player, self.player)
        self.draw_entity(self.simulator.enemy1, self.enemy)
        self.draw_entity(self.simulator.enemy2, self.enemy)
        self.draw_entity(self.simulator.platform1, self.platform)
        self.draw_entity(self.simulator.platform2, self.platform)
        self.draw_entity(self.simulator.platform3, self.platform)
        surf = pygame.transform.flip(self.window, False, True)
        self.window.blit(surf, (0, 0))
        pygame.display.update()

    def draw_episode(self, simulator, name, save=False):
        ''' Draw each state in the simulator into folder name. '''
        lines = ""
        self.simulator = simulator
        for index, state in enumerate(self.simulator.states):
            self.simulator.player.position = state[0]
            self.simulator.enemy1.position = state[1]
            self.simulator.enemy2.position = state[2]
            self.draw()
            if save:
                pygame.image.save(self.window, 'screens/'+ name + '/' + str(index)+'.png')
            lines += str(index) + '.png\n'
            self.clock.tick()
        with open('screens/' + name + '/filenames.txt', 'w') as filename:
            filename.write(lines)

    def draw_entity(self, entity, sprite):
        ''' Draws an entity as a rectangle. '''
        for i in range(int(entity.size[0] / sprite.get_width())):
            pos = entity.position + self.centre
            pos[0] += int(sprite.get_width()*i)
            self.window.blit(sprite, (pos[0], pos[1]))

def main():
    ''' Runs the interface. '''
    interface = Interface()
    while True:
        interface.update()

if __name__ == '__main__':
    main()
