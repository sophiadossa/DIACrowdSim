import math, random
from collections import defaultdict, deque
import pygame
import numpy as np
from pedestrian import Pedestrian, CalmPedestrian, ConfusedPedestrian
from pathfinding import astar
from crowd_environment import get_grid, GRID_SIZE, FLOORPLAN_SURFACE, WINDOW_HEIGHT, WINDOW_WIDTH
from pedestrian_type import PedestrianType

BETA = 70.0

class RLAgent(Pedestrian):

    Q_TABLE = defaultdict(lambda:np.zeros(8))
    EMPTY_CELLS = defaultdict(int)
    EMPTY_DECAY_RATE = 0.99
    
    def __init__(self, spawn=None, target=None,
                 alpha=0.1, gamma=0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float   = 0.2,
                 epsilon_decay: float = 0.9995):

        if spawn is None:
            spawn = [0,0]
        if target is None:
            target = [0,0]

        super().__init__(spawn, target, colour=(0,200,0), speed=2.0)

        self.state = PedestrianType.CALM
    
        # ─── RL hyperparameters ──────────────────────────────────
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

        # Discrete action space: which exit‐rectangle index to choose
        # (we’ll fill in the real exit‐coords later)
        from crowd_environment import targets
        self.exit_choices = list(targets)
        self.n_actions    = len(self.exit_choices)
        # we won’t need action_space any more

        # Q-table for state→action values
        self.q_table = RLAgent.Q_TABLE

        # Track infections caused this tick
        self.infections_this_step = 0
        self.total_infections = 0
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

        # for stuck‐detection
        self.prev_dist       = None          # last tick’s distance to exit
        self.stuck_time      = 0             # consecutive frames with no net progress
        self.STUCK_LIMIT     = 5.0           # seconds to declare “stuck”
        self.FPS             = 60            # simulation frame rate
        self.STUCK_FRAMES    = int(self.STUCK_LIMIT * self.FPS)
        self.stuck_frames    = 0             # consecutive frames with no net progress
        self.total_agents    = None          # set on first panicked step
        self.STUCK_THRESHOLD   = 0.6           # fraction of panicked agents stuck
        self.total_agents    = None          # set on first panicked step

        # fire awareness
        self.FIRE_DELAY = 5.0
        self.FIRE_DURATION = 180.0
        self._panic_start = None

    def get_fire_frac(self):
        if self._panic_start is None:
            return 0.0
        now = (pygame.time.get_ticks() - self._panic_start) / 1000.0
        if now < self.FIRE_DELAY:
            return 0.0
        return min((now - self.FIRE_DELAY) / self.FIRE_DURATION, 1.0)

    def get_state_vector(self):
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]

        return [dx / WINDOW_WIDTH, dy / WINDOW_HEIGHT]
    
    def get_state(self, all_agents):
        """
        State = (dx_exit, dy_exit, dx_crowd, dy_crowd, calm_fraction, dist_from_fire)
        - dx_exit,dy_exit: normalized vector toward exit
        - dx_crowd,dy_crowd: normalized vector toward centroid of calm/confused
        - calm_fraction: fraction of calm/confused neighbors in vision
        - dist_from_fire: normalized x-distance to fire front
        """
        # 1) exit vector
        dx_exit = (self.target[0] - self.pos[0]) / WINDOW_WIDTH
        dy_exit = (self.target[1] - self.pos[1]) / WINDOW_HEIGHT

        # 2) collect calm/confused positions
        calm_list = []
        for p in all_agents:
            if p is self: 
                continue
            if getattr(p, 'is_calm', lambda: False)() or getattr(p, 'is_confused', lambda: False)():
                calm_list.append(p.pos)

        # 3) centroid of calm/confused (fallback to self if none)
        if calm_list:
            cx = sum(x for x, y in calm_list) / len(calm_list)
            cy = sum(y for x, y in calm_list) / len(calm_list)
        else:
            cx, cy = self.pos

        dx_crowd = (cx - self.pos[0]) / WINDOW_WIDTH
        dy_crowd = (cy - self.pos[1]) / WINDOW_HEIGHT

        # 4) local calm/confused density
        calm_neighbors = sum(
            1 for (x,y) in calm_list
            if math.hypot(x - self.pos[0], y - self.pos[1]) <= self.vision_radius
        )
        calm_fraction = min(calm_neighbors, 20) / 20

        # 5) fire distance
        fire_frac = self.get_fire_frac()
        dist_from_fire = ((1 - fire_frac) * WINDOW_WIDTH - self.pos[0]) / WINDOW_WIDTH

        # 6) return fixed tuple
        return (
            round(dx_exit, 2),
            round(dy_exit, 2),
            round(dx_crowd, 2),
            round(dy_crowd, 2),
            round(calm_fraction, 2),
            round(dist_from_fire, 2),
        )

    def select_action(self, state):

        if self.n_actions <= 0:
            return 0
        
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
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
        self.total_infections += 1

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
            self.pull_strength = 5.0

    def become_panicked(self):
        if self.state != PedestrianType.PANIC:
            self.state = PedestrianType.PANIC
            self.colour = (255,0,0)
            self.speed = max(self.speed, 4.0)
            self.panic_radius = 30.0
            self.vision_radius = 40.0

    def is_calm(self):     return self.state is PedestrianType.CALM
    def is_confused(self): return self.state is PedestrianType.CONFUSED
    def is_panicked(self): return self.state is PedestrianType.PANIC

    def draw(self, surface):
        # Only draw trail and special color for leader agents
        if getattr(self, 'is_leader', False):
            # initialize trail buffer
            if not hasattr(self, '_trail'): 
                self._trail = deque(maxlen=200)
            self._trail.append(tuple(self.pos))
            if len(self._trail) > 1:
                pygame.draw.lines(surface, (0,200,0), False, self._trail, 2)

            # solid green circle
            pygame.draw.circle(surface, (0,200,0),
                            (int(self.pos[0]), int(self.pos[1])),
                            int(self.radius))
        else:
            # default: call base class draw (solid filled circle, original colour)
            super().draw(surface)



    def move(self, all_agents, obstacles=None):
        # ── reset contagion counter
        self.infections_this_step = 0

        # ── if not yet panicked, fall back to calm/confused logic
        if not self.is_panicked():
            if self.is_calm():
                return CalmPedestrian.move(self, all_agents, obstacles)
            else:
                return ConfusedPedestrian.move(self, all_agents, obstacles)

        # ── *only* true panic‐leaders get here
        if not getattr(self, 'is_leader', False):
            # non‐leader panics just idle or follow last heading
            return

        # ── 0) decay the shared “empty cell” memory
        for cell in list(RLAgent.EMPTY_CELLS):
            RLAgent.EMPTY_CELLS[cell] *= RLAgent.EMPTY_DECAY_RATE
            if RLAgent.EMPTY_CELLS[cell] < 0.1:
                del RLAgent.EMPTY_CELLS[cell]

        # ── on first panic tick, record crowd size & timestamp
        if self.total_agents is None:
            self.total_agents  = len(all_agents)
            self._panic_start  = pygame.time.get_ticks()
            # reset epsilon for fresh exploration each trial
            self.epsilon = self.epsilon_start

        # ── discretize position for both stuck and emptiness checks
        gx, gy = int(self.pos[0]//GRID_SIZE), int(self.pos[1]//GRID_SIZE)

        # ── compute A* path‐length before move (for stuck)
        grid = get_grid()
        start = (gx, gy)
        goal = (int(self.target[0]//GRID_SIZE), int(self.target[1]//GRID_SIZE))
        path0 = astar(grid, start, goal) or []
        d0 = len(path0)

        # ── 1) observe & select
        s0 = self.get_state(all_agents)
        self.prev_state = s0
        a = self.select_action(s0)
        self.last_action = a

        # ── 2) apply
        old_x, old_y = self.pos.copy()
        self.apply_action(a)

        # ── 3) measure movement
        dx = self.pos[0] - old_x
        dy = self.pos[1] - old_y
        dist_moved = math.hypot(dx, dy)

        # ── 4) A* after move
        nx, ny = int(self.pos[0]//GRID_SIZE), int(self.pos[1]//GRID_SIZE)
        path1 = astar(grid, (nx, ny), goal) or []
        d1 = len(path1)

        # ── stuck detection
        self.stuck_frames = (getattr(self, 'stuck_frames', 0) + 1) if d1 >= d0 else 0
        stuck_penalty = 0.0
        if self.stuck_frames >= self.STUCK_FRAMES:
            stuck_penalty = -1.0
            self.q_table[s0][a] -= 1.0
            self.stuck_frames = 0

        # ── 5) count calm/confused nearby for “emptiness”
        nearby = sum(
            1 for p in all_agents
            if (getattr(p,'is_calm',lambda:False)() or getattr(p,'is_confused',lambda:False)())
               and math.hypot(p.pos[0]-self.pos[0], p.pos[1]-self.pos[1]) < self.vision_radius
        )
        if nearby == 0:
            RLAgent.EMPTY_CELLS[(gx,gy)] += 1.0
        emptiness_penalty = -4.0 * RLAgent.EMPTY_CELLS.get((gx,gy), 0.0)

        # inside RLAgent.move, after you compute `nearby`…
        crowd_density = sum(
            1 for p in all_agents
            if not isinstance(p, RLAgent) and
            math.hypot(p.pos[0]-self.pos[0], p.pos[1]-self.pos[1]) < self.vision_radius
        )
        # penalize high-density cells
        DESIRED_MIN_FOLLOWERS = 10
        density_penalty = -0.1 * max(0, crowd_density - DESIRED_MIN_FOLLOWERS)

        # ── 6) neglect penalty
        remaining = sum(
            1 for p in all_agents
            if getattr(p,'is_calm',lambda:False)() or getattr(p,'is_confused',lambda:False)()
        )
        neglect_penalty = -1.0 if (remaining>0 and d1==0) else -(remaining/self.total_agents)

        # ── 7) other reward bits
        spatial_reward    = d0 - d1
        infection_bonus   = BETA * self.infections_this_step
        collision_penalty = -0.5 if dist_moved < self.radius else 0.0
        time_penalty      = -0.01

        # bonus for being within vision radius of any calm/conf
        nearby = sum(1 for p in all_agents if p is not self
                    and (p.is_calm() or p.is_confused())
                    and math.hypot(p.pos[0]-self.pos[0], p.pos[1]-self.pos[1])
                        <= self.vision_radius)
        proximity_bonus = 2.0 * nearby

        fire_frac = self.get_fire_frac()
        fire_x = (1 - fire_frac)*WINDOW_WIDTH

        near_fire_penalty = 0.0
        if self.pos[0] + self.radius*1.0 >= fire_x:
            near_fire_penalty = -40.0

        R_total = (
            spatial_reward
          + infection_bonus
          + collision_penalty
          + stuck_penalty
          + time_penalty
          + neglect_penalty
          + emptiness_penalty
          + proximity_bonus
          + near_fire_penalty
          + density_penalty
        )

        # ── 8) Q-learning update
        s1 = self.get_state(all_agents)
        best_next = np.max(self.q_table[s1])
        self.q_table[s0][a] += self.alpha * (
            R_total + self.gamma*best_next - self.q_table[s0][a]
        )

        # ── 9) ε-decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ── 10) global abort checks
        now = (pygame.time.get_ticks() - self._panic_start)/1000.0
        panicked = [p for p in all_agents if getattr(p,'is_panicked',lambda:False)()]
        stuck_pct = sum(1 for p in panicked if getattr(p,'stuck_frames',0)>=self.STUCK_FRAMES) / max(1,len(panicked))
        if stuck_pct >= self.STUCK_THRESHOLD:
            raise StopIteration
        if len(panicked)==self.total_agents and now>120.0:
            raise StopIteration
        if not all_agents:
            raise StopIteration


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
            wander = 3
            # apply class-level pull_strength instead of fixed 0.5
            pull = self.pull_strength
            cand_x = self.pos[0] + random.uniform(-wander, wander) + ux * (self.speed * pull)
            cand_y = self.pos[1] + random.uniform(-wander, wander) + uy * (self.speed * pull)
            if self.check_collision(cand_x, cand_y):
                self.pos[0], self.pos[1] = cand_x, cand_y
                self._clamp()
        elif self.counter > self.pause_timer+30:
            self.counter = 0
            self.pause_timer = random.randint(60,120)
        
        self._clamp()

    def _move_panic(self, all_agents = None, obstacles = None):
        # 1) find nearest RL leader
        leaders = [p for p in all_agents if isinstance(p, RLAgent) and p.is_panicked()]
        if leaders:
            leader = min(leaders, key=lambda L: math.hypot(L.pos[0]-self.pos[0], L.pos[1]-self.pos[1]))
            # head toward the leader’s current position (one step)
            dx, dy = leader.pos[0]-self.pos[0], leader.pos[1]-self.pos[1]
            dist = math.hypot(dx, dy) or 1.0
            step = min(self.speed, dist)
            nx = self.pos[0] + (dx/dist)*step
            ny = self.pos[1] + (dy/dist)*step
            if self.check_collision(nx, ny):
                self.pos = [nx, ny]
                return

        # 2) fallback: pure A* to exit
        grid = get_grid()
        start = (int(self.pos[0]//GRID_SIZE), int(self.pos[1]//GRID_SIZE))
        goal  = (int(self.target[0]//GRID_SIZE), int(self.target[1]//GRID_SIZE))
        path = astar(grid, start, goal) or []
        if not path: return

        nx, ny = ((path[0][0]+0.5)*GRID_SIZE, (path[0][1]+0.5)*GRID_SIZE)
        dx, dy = nx-self.pos[0], ny-self.pos[1]
        d = math.hypot(dx, dy) or 1.0
        step = min(self.speed, d)
        cand_x = self.pos[0] + dx/d*step
        cand_y = self.pos[1] + dy/d*step
        if self.check_collision(cand_x, cand_y):
            self.pos = [cand_x, cand_y]

    


