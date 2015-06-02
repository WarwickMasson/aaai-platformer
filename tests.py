'''
Implements unit tests.
'''

from unittest import TestCase
import numpy as np
import simulator
import util
import learn

ITERATIONS = 1000

class TestSimulator(TestCase):

    def test_colliding(self):
        player = simulator.Player()
        player2 = simulator.Player()
        self.assertTrue(player.colliding(player2))
        for _ in range(ITERATIONS):
            player2.position = player.position + player.size * np.random.rand(2)
            self.assertTrue(player.colliding(player2))
            self.assertTrue(player2.colliding(player))
            player2.position = player.position + player.size * (1 + np.random.rand(2))
            self.assertFalse(player.colliding(player2))
            self.assertFalse(player2.colliding(player))
            player2.position = player.position - player.size * (1 + np.random.rand(2))
            self.assertFalse(player.colliding(player2))
            self.assertFalse(player2.colliding(player))

    def test_decollide(self):
        player = simulator.Player()
        player2 = simulator.Player()
        for _ in range(ITERATIONS):
            player2.position = player.position + player.size * np.random.rand(2)
            player.decollide(player2)
            self.assertFalse(player.colliding(player2))

    def test_jump(self):
        for _ in range(ITERATIONS):
            sim = simulator.Simulator()
            player = sim.player
            player.velocity[0] = 20.0*np.random.rand()
            jump = 100.0*np.random.rand()
            state, reward, end, steps = sim.take_action(('jump', jump))
            on_first = player.on_platform(sim.platform1)
            self.assertTrue(end or on_first)
            self.assertEquals(reward, player.position[0])
            self.assertNotEquals(reward, 0.0)
            if end:
                collide1 = player.colliding(sim.enemy1)
                collide2 = player.colliding(sim.enemy2)
                collide3 = player.colliding(sim.spikes)
                collide = collide1 or collide2 or collide3
                self.assertTrue(collide or player.position[1] < sim.lower_bound())

    def test_run(self):
        for _ in range(ITERATIONS):
            sim = simulator.Simulator()
            run = 20.0*np.random.rand()
            state, reward, end, steps = sim.take_action(('run', run))
            on_first = sim.player.on_platform(sim.platform1)
            self.assertTrue(end or on_first)
            self.assertEquals(reward, sim.player.position[0])
            if end:
                self.assertTrue(sim.player.colliding(sim.enemy1))
