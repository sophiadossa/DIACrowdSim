import pygame
import math
import random
from pathfinding import astar
from crowd_environment import get_grid, GRID_SIZE

class Pedestrian:
    def __init__(self, spawn, target, colour=(0, 0, 255), speed=2):
        self.pos = [spawn[0], spawn[1]]
        self.target = target
        self.colour = colour
        self.speed = speed
        self.radius = 8
        self.direction = [0, 0]

        self.grid = get_grid()
        self.path = self.find_path()

    def find_path(self):
        start_node = (int(self.pos[0] // GRID_SIZE), int(self.pos[1] // GRID_SIZE))
        goal_node = (int(self.target[0] // GRID_SIZE), int(self.target[1] // GRID_SIZE))

        return astar(self.grid, start_node, goal_node)

    def update_direction(self):
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.direction = [dx / dist, dy / dist]
        else:
            self.direction = [0, 0]

    def move(self):
        if not self.path:
            return

        next_tile = self.path[0]
        next_px = [(next_tile[0] + 0.5) * GRID_SIZE, (next_tile[1] + 0.5) * GRID_SIZE]
        dx = next_px[0] - self.pos[0]
        dy = next_px[1] - self.pos[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5

        if dist < self.speed:
            self.pos = next_px
            self.path.pop(0)
        else:
            self.pos[0] += self.speed * dx / dist
            self.pos[1] += self.speed * dy / dist
        # self.update_direction()
        # self.pos[0] += self.direction[0] * self.speed
        # self.pos[1] += self.direction[1] * self.speed

    def draw(self, surface):
        pygame.draw.circle(surface, self.colour, (int(self.pos[0]), int(self.pos[1])), self.radius)

    def set_movement(self, speed):
        self.speed = speed

    def get_position(self):
        return self.pos

    def get_radius(self):
        return self.radius


# class PanicPedestrian(Pedestrian):
#     def __init__(self, spawn, target):
#         super().__init__(spawn, target, colour=(255, 0, 0), speed=3)
#         self.angle = 0

#     def move(self):
#         super().move()
#         self.angle += 30  # For visual spinning behaviour

#     def draw(self, surface):
#         pygame.draw.circle(surface, self.colour, (int(self.pos[0]), int(self.pos[1])), self.radius)
#         pygame.draw.arc(surface, (0, 0, 0), pygame.Rect(int(self.pos[0])-8, int(self.pos[1])-8, 16, 16), 0, self.angle % 360, 2)


# class ConfusedPedestrian(Pedestrian):
#     def __init__(self, spawn, target):
#         super().__init__(spawn, target, colour=(128, 0, 128), speed=1.5)
#         self.pause_timer = random.randint(40, 100)
#         self.counter = 0

#     def move(self):
#         self.counter += 1
#         if self.counter < self.pause_timer:
#             super().move()
#         else:
#             if self.counter > self.pause_timer + 40:
#                 self.counter = 0


class CalmPedestrian(Pedestrian):
    def __init__(self, spawn, target):
        super().__init__(spawn, target, colour=(255, 255, 0), speed=1.2)

    def move(self, obstacles=[]):
        for obs in obstacles:
            if self.close_to_wall(obs):
                self.speed = 0.6
                break
        else:
            self.speed = 1.2
        super().move()

    def close_to_wall(self, obs_rect):
        x, y = self.pos
        px, py = x / GRID_SIZE, y / GRID_SIZE
        ox, oy, ow, oh = obs_rect
        return (ox - 0.5 < px < ox + ow + 0.5 and oy - 0.5 < py < oy + oh + 0.5)
