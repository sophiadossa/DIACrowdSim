import pygame
import math
import collections
import random
from pathfinding import astar, get_neighbours
from crowd_environment import get_grid, get_grid_obstacles, GRID_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH, dom, FLOORPLAN_SURFACE
from cromosim.micro import (
    compute_contacts,
    compute_forces,
    move_people,
    people_update_destination,
)

class Pedestrian:
    def __init__(self, spawn, target, colour=(0,0,255), speed=2):
        # self.pos    = [spawn[0], spawn[1]]
        self.target = target
        self.colour = colour
        self.speed  = speed
        # self.radius = 5

        self.pos = [spawn[0], spawn[1]]
        self.vel = [0.0, 0.0]
        self.radius = 8.0
        self.mass = 80.0
        self.desired_velocity = 1.2
        self.domain = dom

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

    def check_collision(self, new_x, new_y):
        """
        Return True if the pixel at (new_x,new_y) in the floorplan
        is NOT black – i.e. it's free space.
        """
        w, h = FLOORPLAN_SURFACE.get_size()
        # directions: center + 8 compass points
        checks = [(0, 0)]
        for angle in [i * math.pi/4 for i in range(8)]:
            checks.append((
                self.radius * math.cos(angle),
                self.radius * math.sin(angle)
            ))

        for dx, dy in checks:
            px = int(new_x + dx)
            py = int(new_y + dy)
            # clamp to image bounds
            px = max(0, min(px, w-1))
            py = max(0, min(py, h-1))
            r, g, b, *_ = FLOORPLAN_SURFACE.get_at((px, py))
            if (r, g, b) == (0, 0, 0):
                return False
        return True

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

    # def move(self, *args, **kwargs):
    #     pass

    def move(self, all_agents, obstacles=None):
        # 1) occasionally pick a new random direction
        self.change_timer -= 1
        if self.change_timer <= 0:
            angle = random.random() * math.tau
            self.dir = [math.cos(angle), math.sin(angle)]
            self.change_timer = random.randint(30, 90)

        # 2) attempt up to 5 small hops along that heading
        for _ in range(5):
            dx = self.dir[0] * self.speed
            dy = self.dir[1] * self.speed
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if self.check_collision(nx, ny):
                self.pos[0], self.pos[1] = nx, ny
                self._clamp()
                return
            # nudge direction slightly & retry
            self.dir[0] += (random.random() - 0.5) * 0.5
            self.dir[1] += (random.random() - 0.5) * 0.5

        # 3) if still blocked, pick a totally fresh heading
        angle = random.random() * math.tau
        self.dir = [math.cos(angle), math.sin(angle)]


# —————————————————————————————————

class ConfusedPedestrian(Pedestrian):
    def __init__(self, spawn, target):
        super().__init__(spawn, target, colour=(128,0,128),
                         speed=random.uniform(1.2, 1.8))
        self.pause_timer = random.randint(50, 100)
        self.counter     = 0
        self.hear_count  = 0

    # def move(self, *args, **kwargs):
    #     pass

    def move(self, all_agents, obstacles=None):
        self.counter += 1

        # steer partly toward nearest panic
        panics = [p for p in all_agents if isinstance(p, PanicPedestrian)]
        if panics:
            nearest = min(panics,
                          key=lambda p: math.hypot(p.pos[0]-self.pos[0],
                                                   p.pos[1]-self.pos[1]))
            dx, dy = nearest.pos[0]-self.pos[0], nearest.pos[1]-self.pos[1]
            d = math.hypot(dx, dy) or 1.0
            ux, uy = dx/d, dy/d
        else:
            ux = uy = 0.0

        # 1) during pause, jitter + pull
        if self.counter < self.pause_timer:
            wander = 4
            cand_x = self.pos[0] + random.uniform(-wander, wander) + ux*(self.speed*0.5)
            cand_y = self.pos[1] + random.uniform(-wander, wander) + uy*(self.speed*0.5)
            if self.check_collision(cand_x, cand_y):
                self.pos[0], self.pos[1] = cand_x, cand_y
                self._clamp()
        # 2) after pause window, reset the cycle
        elif self.counter > self.pause_timer + 30:
            self.counter = 0
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

    # def move(self, *args, **kwargs):
    #     pass

    def move(self, all_agents=None, obstacles=None):
       # recompute A* path every frame
        grid = get_grid()
        start = (int(self.pos[0] // GRID_SIZE), int(self.pos[1] // GRID_SIZE))
        goal  = (int(self.target[0] // GRID_SIZE), int(self.target[1] // GRID_SIZE))
        self.path = astar(grid, start, goal) or []

        if not self.path:
            return

        # pick the first reachable path step
        cand_step = None
        while self.path:
            nx, ny = self.path[0]
            next_x = (nx + 0.5) * GRID_SIZE
            next_y = (ny + 0.5) * GRID_SIZE
            if self.check_collision(next_x, next_y):
                cand_step = (next_x, next_y)
                break
            else:
                self.path.pop(0)

        if cand_step is None:
            return

        # actually move toward that step
        cx, cy = cand_step
        dx, dy = cx - self.pos[0], cy - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return

        step = min(self.speed, dist)
        nx = self.pos[0] + (dx/dist) * step
        ny = self.pos[1] + (dy/dist) * step

        if self.check_collision(nx, ny):
            self.pos[0], self.pos[1] = nx, ny
            self._clamp()
        else:
            # force a re-route next tick
            self.path = []

    def draw(self, surface):
        super().draw(surface)
        # draw your panic/vision radii if desired
        pygame.draw.circle(surface, (255,0,0),
                           (int(self.pos[0]), int(self.pos[1])),
                           int(self.panic_radius), 1)
        pygame.draw.circle(surface, (0,255,0),
                           (int(self.pos[0]), int(self.pos[1])),
                           int(self.vision_radius), 1)
        