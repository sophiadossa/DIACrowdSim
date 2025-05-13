import pygame
import math
import collections
import random
from pathfinding import astar, get_neighbours
from crowd_environment import get_grid, get_grid_obstacles, GRID_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH

class Pedestrian:
    def __init__(self, spawn, target, colour=(0,0,255), speed=2):
        self.pos    = [spawn[0], spawn[1]]
        self.target = target
        self.colour = colour
        self.speed  = speed
        self.radius = 5

    def draw(self, surface):
        pygame.draw.circle(surface, self.colour,
                           (int(self.pos[0]), int(self.pos[1])),
                           self.radius)

    def get_radius(self):
        return self.radius
    
    def _clamp(self):
        # keep circle fully on-screen
        self.pos[0] = max(self.radius, min(self.pos[0], WINDOW_WIDTH - self.radius))
        self.pos[1] = max(self.radius, min(self.pos[1], WINDOW_HEIGHT - self.radius))


# —————————————————————————————————

class CalmPedestrian(Pedestrian):
    def __init__(self, spawn, target):
        super().__init__(spawn, target,
                         colour=(30,144,255),
                         speed=random.uniform(0.5, 1.2))
        # pick an initial random direction
        angle = random.random() * math.tau
        self.dir = [math.cos(angle), math.sin(angle)]
        self.change_timer = random.randint(30, 90)
        self.hear_count = 0

    def move(self, all_agents, obstacles=None):
        # 1) occasionally pick a new random direction
        self.change_timer -= 1
        if self.change_timer <= 0:
            angle = random.random() * math.tau
            self.dir = [math.cos(angle), math.sin(angle)]
            self.change_timer = random.randint(30, 90)

        # 2) step in that direction
        dx = self.dir[0] * self.speed
        dy = self.dir[1] * self.speed
        new_pos = [self.pos[0] + dx, self.pos[1] + dy]

        # grid check
        grid = get_grid()
        cell_x = int(new_pos[0] // GRID_SIZE)
        cell_y = int(new_pos[1] // GRID_SIZE)
        # only step if inside bounds ans free
        if (0 <= cell_x < len(grid) and
            0 <= cell_y < len(grid[0]) and
            grid[cell_x][cell_y] == 0):
            self.pos = new_pos
            self._clamp()
        else:
            self.dir[0] *= -1
            self.dir[1] *= -1


# —————————————————————————————————

class ConfusedPedestrian(Pedestrian):
    def __init__(self, spawn, target):
        super().__init__(spawn, target, colour=(128,0,128),
                         speed=random.uniform(1.2, 1.8))
        self.pause_timer = random.randint(50, 100)
        self.counter     = 0
        self.hear_count  = 0

    def move(self, all_agents, obstacles=None):
        self.counter += 1

        # find nearest panic (if any)
        panics = [a for a in all_agents if isinstance(a, PanicPedestrian)]
        if panics:
            nearest = min(panics,
                          key=lambda p: math.hypot(p.pos[0]-self.pos[0],
                                                   p.pos[1]-self.pos[1]))
            dx = nearest.pos[0] - self.pos[0]
            dy = nearest.pos[1] - self.pos[1]
            d  = math.hypot(dx, dy)
            if d > 1e-6:
                ux, uy = dx/d, dy/d
            else:
                ux = uy = 0.0
        else:
            ux = uy = 0.0

        # during pause, jitter + strong pull
        if self.counter < self.pause_timer:
            wander = 4
            # *increase* the panic‐pull to 50% of top speed
            cand_x = self.pos[0] + random.uniform(-wander, wander) + ux * (self.speed * 0.5)
            cand_y = self.pos[1] + random.uniform(-wander, wander) + uy * (self.speed * 0.5)

            grid = get_grid()
            cx, cy = int(cand_x // GRID_SIZE), int(cand_y // GRID_SIZE)
            if 0 <= cx < len(grid) and 0 <= cy < len(grid[0]) and grid[cx][cy] == 0:
                self.pos[0] = cand_x
                self.pos[1] = cand_y
                self._clamp()

        # after pause window, reset cycle
        elif self.counter > self.pause_timer + 30:
            self.counter     = 0
            self.pause_timer = random.randint(60, 120)

# —————————————————————————————————
class PanicPedestrian(Pedestrian):
    def __init__(self, spawn, target,
                 speed: float = 3.0,
                 panic_radius: float = 15.0,
                 vision_radius: float = 60.0):
        super().__init__(spawn, target, colour=(255,0,0), speed=speed)
        self.panic_radius   = panic_radius
        self.vision_radius  = vision_radius
        # we’ll store our own A* path here
        self.path = []
        self.recalc_timer = 0
        self.recalc_interval = 12

    def move(self, all_agents=None, obstacles=None):
        # 1) recompute a fresh A* path to the exit every frame
        grid  = get_grid()
        start = (int(self.pos[0] // GRID_SIZE), int(self.pos[1] // GRID_SIZE))
        goal  = (int(self.target[0] // GRID_SIZE), int(self.target[1] // GRID_SIZE))
        path = astar(grid, start, goal)

        if not path: 
            return   # no path found, stay put

        # 2) take the very first step on that path
        nx, ny   = path[0]
        next_px  = ((nx + 0.5) * GRID_SIZE, (ny + 0.5) * GRID_SIZE)
        dx, dy   = next_px[0] - self.pos[0], next_px[1] - self.pos[1]
        dist     = math.hypot(dx, dy)

        if dist < 1e-6:
            return  # already there

        step = min(self.speed, dist)
        cand_x = self.pos[0] + step * dx / dist
        cand_y = self.pos[1] + step * dy / dist

        grid = get_grid()
        cx, cy = int(cand_x // GRID_SIZE), int(cand_y // GRID_SIZE)
        if 0 <= cx < len(grid) and 0 <= cy < len(grid[0]) and grid[cx][cy] == 0:
            self.pos[0] = cand_x
            self.pos[1] = cand_y
            self._clamp()

    def draw(self, surface):
        super().draw(surface)
        # draw your panic/vision radii if desired
        pygame.draw.circle(surface, (255,0,0),
                           (int(self.pos[0]), int(self.pos[1])),
                           int(self.panic_radius), 1)
        pygame.draw.circle(surface, (0,255,0),
                           (int(self.pos[0]), int(self.pos[1])),
                           int(self.vision_radius), 1)
        