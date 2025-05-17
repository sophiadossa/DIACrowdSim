import math, random
from collections import defaultdict, deque
import pygame
import numpy as np
from pedestrian import Pedestrian, CalmPedestrian, ConfusedPedestrian
from pathfinding import astar
from crowd_environment import get_grid, GRID_SIZE, FLOORPLAN_SURFACE, WINDOW_HEIGHT, WINDOW_WIDTH
from pedestrian_type import PedestrianType

BETA = 2.0

class RLAgent(Pedestrian):
    def __init__(self, spawn, target,
                 alpha=0.1, gamma=0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float   = 0.01,
                 epsilon_decay: float = 0.995):
        super().__init__(spawn, target, colour=(255,20,147), speed=2.0)

        self.state = PedestrianType.CALM

        # ─── RL hyperparameters ──────────────────────────────────────
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Discrete action space: 8 compass headings
        self.action_space = [
            ( 1,  0), (-1,  0),
            ( 0,  1), ( 0, -1),
            ( 1,  1), ( 1, -1),
            (-1,  1), (-1, -1),
        ]
        self.n_actions = len(self.action_space)

        # Q-table for state→action values
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

        # Track infections caused this tick
        self.infections_this_step = 0
        self.dir = [1.0, 0.0]
        self.change_timer = random.randint(30,90)
        self.pause_timer = random.randint(50, 100)
        self.counter = 0
        self.vision_radius = 40.0
        self.panic_radius = 30.0

        self.state_dim = 3
        self.action_dim = self.n_actions

        self.collision_count = 0
        self.speed_reduced_times = 0

    def get_state_vector(self):
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        return [dx / WINDOW_WIDTH, dy / WINDOW_HEIGHT]
    
    def get_state(self, all_agents):
        """
        Minimal state: normalized vector toward exit.
        Returns a hashable tuple for Q-table indexing.
        """
        dx = (self.target[0] - self.pos[0]) / WINDOW_WIDTH
        dy = (self.target[1] - self.pos[1]) / WINDOW_HEIGHT

        calm_neighbors = 0
        for p in all_agents:
            if p is not self and (p.is_calm() or p.is_confused()):
                dist = math.hypot(p.pos[0] - self.pos[0], p.pos[1] - self.pos[1])
                if dist <= self.vision_radius:
                    calm_neighbors += 1
        # cap for normalization
        max_neighbors = 20
        calm_fraction = min(calm_neighbors, max_neighbors) / max_neighbors

        return (round(dx,2), round(dy,2), round(calm_fraction,2))
    
    def select_action(self, state):
        # explore
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        # exploit
        return int(np.argmax(self.q_table[state]))
    
    def apply_action(self, action):
        dx, dy = self.action_space[action]
        nx = self.pos[0] + dx * self.speed
        ny = self.pos[1] + dy * self.speed
        # only move if no wall collision
        if self.check_collision(nx, ny):
            self.pos = [nx, ny]
        self._clamp()

    def on_infect(self):
        self.infections_this_step += 1

    def on_collision(self):
        """Called by manager when this RL agent bumps into another panicked one."""
        self.collision_count += 1
        # once we exceed 5 collisions, slow them down and penalize
        if self.collision_count > 5 and self.speed_reduced_times == 0:
            # reduce speed once
            self.speed = max(0.1, self.speed - 0.1)
            self.speed_reduced_times += 1
            # apply a one-time -1 to last action in Q-table,
            # assuming you've stored self.prev_state and self.last_action in move()
            if hasattr(self, 'prev_state') and hasattr(self, 'last_action'):
                self.q_table[self.prev_state][self.last_action] -= 1

    def become_confused(self):
        if self.state != PedestrianType.CONFUSED:
            self.state = PedestrianType.CONFUSED
            self.colour = (128,0,128)
            self.counter = 0
            self.pause_timer = random.randint(50,100)

    def become_panicked(self):
        if self.state != PedestrianType.PANIC:
            self.state = PedestrianType.PANIC
            self.colour = (255,0,0)
            self.speed = max(self.speed, 2.0)

    def is_calm(self):     return self.state is PedestrianType.CALM
    def is_confused(self): return self.state is PedestrianType.CONFUSED
    def is_panicked(self): return self.state is PedestrianType.PANIC

    def draw(self, surface):
        super().draw(surface)
        # if self.is_panicked():
        #     pygame.draw.circle(surface, (255,0,0),
        #                        (int(self.pos[0]), int(self.pos[1])),
        #                        int(self.panic_radius), 1)
        #     pygame.draw.circle(surface, (0,255,0),
        #                        (int(self.pos[0]), int(self.pos[1])),
        #                        int(self.vision_radius), 1)

    def move(self, all_agents, obstacles=None):
        # Reset contagion counter at start of tick
        self.infections_this_step = 0
        

        # Rule-based until panicked
        if not self.is_panicked():
            if self.is_calm():
                CalmPedestrian.move(self, all_agents, obstacles)
            else:
                ConfusedPedestrian.move(self, all_agents, obstacles)
            return

        # — Panicked: perform RL step —

        # 1) observe current state & distance
        s = self.get_state(all_agents)
        self.prev_state = s
        old_dist = self.distance_to_exit()

        # record position for collision/stuck detection
        old_x, old_y = self.pos

        # 2) select & apply action
        a = self.select_action(s)
        self.last_action = a
        self.apply_action(a)

        # 3) observe next state & distance
        s2 = self.get_state(all_agents)
        new_dist = self.distance_to_exit()

        # compute how far we actually moved this tick
        dx = self.pos[0] - old_x
        dy = self.pos[1] - old_y
        dist_moved = math.hypot(dx, dy)

        # 4) base reward = decrease in distance to exit
        spatial_reward = old_dist - new_dist

        # 5) collision/stuck penalty if we didn’t move at least our radius
        collision_penalty = 0.0
        if dist_moved < self.radius:
            collision_penalty = -0.5

        # 6) contagion bonus
        infection_bonus = BETA * self.infections_this_step

        # total combined reward
        r_total = spatial_reward + infection_bonus + collision_penalty

        # 7) Q-learning update
        best_next = np.max(self.q_table[s2])
        self.q_table[s][a] += self.alpha * (
            r_total + self.gamma * best_next - self.q_table[s][a]
        )

        # 8) decay ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _move_calm(self):
        # identical to CalmPedestrian.move but stay inside self.home_zone
        sx, sy, sw, sh = getattr(self, "home_zone", (0,0,0,0))
        min_x = sx     * GRID_SIZE
        max_x = (sx+sw)* GRID_SIZE
        min_y = sy     * GRID_SIZE
        max_y = (sy+sh)* GRID_SIZE

        self.change_timer -= 1
        if self.change_timer <= 0:
            angle = random.random() * math.tau
            self.dir = [math.cos(angle), math.sin(angle)]
            self.change_timer = random.randint(30,90)

        for _ in range(5):
            nx = self.pos[0] + self.dir[0] * self.speed
            ny = self.pos[1] + self.dir[1] * self.speed

            #  a) clamp to screen
            super()._clamp()
            #  b) if you have a home_zone, clamp back inside it:
            if hasattr(self, 'home_zone'):
                min_x, max_x, min_y, max_y = self.home_zone
                self.pos[0] = max(min_x, min(self.pos[0], max_x))
                self.pos[1] = max(min_y, min(self.pos[1], max_y))

            # 1) must stay inside home_zone
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                # bounce off by picking a fresh heading and retry
                angle = random.random() * math.tau
                self.dir = [math.cos(angle), math.sin(angle)]
                continue

            # 2) must not hit a wall
            if self.check_collision(nx, ny):
                self.pos = [nx, ny]
                return

            # 3) nudge & retry
            self.dir[0] += (random.random() - 0.5) * 0.5
            self.dir[1] += (random.random() - 0.5) * 0.5

        # if all else fails, pick a totally new heading
        angle = random.random() * math.tau
        self.dir = [math.cos(angle), math.sin(angle)]

    def _move_confused(self, all_agents):
        # identical to ConfusedPedestrian.move
        self.counter += 1
        panics = [p for p in all_agents if p.is_panicked()]
        if panics:
            nearest = min(panics, key=lambda p: math.hypot(p.pos[0]-self.pos[0], p.pos[1]-self.pos[1]))
            dx,dy = nearest.pos[0]-self.pos[0], nearest.pos[1]-self.pos[1]
            d = math.hypot(dx,dy) or 1.0
            ux,uy = dx/d, dy/d
        else:
            ux = uy = 0.0
        if self.counter < self.pause_timer:
            wander = 4
            nx = self.pos[0] + random.uniform(-wander,wander) + ux*(self.speed*0.5)
            ny = self.pos[1] + random.uniform(-wander,wander) + uy*(self.speed*0.5)
            if self.check_collision(nx, ny):
                self.pos = [nx,ny]
        elif self.counter > self.pause_timer+30:
            self.counter = 0
            self.pause_timer = random.randint(60,120)
        
        self._clamp()

    def _move_panic(self):
        # identical to PanicPedestrian.move
        grid = get_grid()
        start = (int(self.pos[0]//GRID_SIZE), int(self.pos[1]//GRID_SIZE))
        goal  = (int(self.target[0]//GRID_SIZE), int(self.target[1]//GRID_SIZE))
        path = astar(grid, start, goal) or []
        if not path: return
        # step to first reachable
        for nx,ny in path:
            px,py = (nx+0.5)*GRID_SIZE, (ny+0.5)*GRID_SIZE
            if self.check_collision(px,py):
                dx,dy = px-self.pos[0], py-self.pos[1]
                d = math.hypot(dx,dy)
                step = min(self.speed, d)
                nx = self.pos[0] + dx/d*step
                ny = self.pos[1] + dy/d*step
                if self.check_collision(nx,ny):
                    self.pos = [nx,ny]
                return
            
            self._clamp()

        angle = random.random() * 2*math.pi
        nx = self.pos[0] + math.cos(angle)*self.speed
        ny = self.pos[1] + math.sin(angle)*self.speed
        if self.check_collision(nx, ny):
            self.pos = [nx, ny]
    


