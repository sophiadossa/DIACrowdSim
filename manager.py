from pedestrian import CalmPedestrian, PanicPedestrian, ConfusedPedestrian, Pedestrian
import random
import numpy as np
import sys
import math
from enum import Enum
from pathfinding import astar
from crowd_environment import get_grid, get_grid_obstacles, GRID_SIZE, dom, WINDOW_HEIGHT, WINDOW_WIDTH
from cromosim.micro import (
    compute_contacts,
    compute_forces,
    move_people,
    people_update_destination,
)

class PedestrianType(Enum):
    CALM     = 4
    CONFUSED = 1
    PANIC    = 2

class PedestrianManager:
    def __init__(self, sources, targets):
        self.agents = []
        self.sources = sources
        self.targets = targets
        self.pending_replacements = []
        self.last_counts = (-1, -1, -1)
        self.wave_timer = 5

    def update(self, dt):
        # 0) drop evacuees who reached their target
        self.agents = [
            a for a in self.agents
            if math.hypot(a.pos[0] - a.target[0], a.pos[1] - a.target[1]) > a.radius
        ]

        # 1) immediate Calm→Confused: any Calm within any Panic’s vision
        panics = [p for p in self.agents if isinstance(p, PanicPedestrian)]
        for calm in [a for a in self.agents if isinstance(a, CalmPedestrian)]:
            if any(
                math.hypot(calm.pos[0] - p.pos[0], calm.pos[1] - p.pos[1]) 
                  <= p.vision_radius
                for p in panics
            ):
                self._replace_agent(calm, PedestrianType.CONFUSED)

        # 2) apply those conversions now (so next move sees the new types)
        for old, new_t in self.pending_replacements:
            if old in self.agents:
                self.agents.remove(old)
                self.spawn_agent(new_t, custom_spawn=old.pos)
        self.pending_replacements.clear()

        # 3) shared A* path planning for all Panics
        panics = [p for p in self.agents if isinstance(p, PanicPedestrian)]
        if panics:
            cx = sum(p.pos[0] for p in panics) / len(panics)
            cy = sum(p.pos[1] for p in panics) / len(panics)
            start = (int(cx // GRID_SIZE), int(cy // GRID_SIZE))

            tx, ty, *_ = self._get_random_target()
            goal_px = ((tx + 0.5) * GRID_SIZE, (ty + 0.5) * GRID_SIZE)
            goal = (int(goal_px[0] // GRID_SIZE), int(goal_px[1] // GRID_SIZE))

            shared_path = astar(get_grid(), start, goal) or []
            for p in panics:
                p.path = shared_path.copy()

        # 4) now move everyone (each move() handles its own collision logic)
        for a in self.agents:
            a.move(self.agents, get_grid_obstacles())

        # 5) handle interpersonal collisions & panic‐spreading
        self.handle_collisions()

        # 6) debug counts
        calm_c  = sum(isinstance(a, CalmPedestrian)    for a in self.agents)
        conf_c  = sum(isinstance(a, ConfusedPedestrian) for a in self.agents)
        panic_c = sum(isinstance(a, PanicPedestrian)    for a in self.agents)
        counts = (calm_c, conf_c, panic_c)
        if counts != self.last_counts:
            print(f"CALM={calm_c}  CONF={conf_c}  PANIC={panic_c}")
            self.last_counts = counts


    def spawn_agent(self, agent_type, custom_spawn=None, custom_target=None):
        """
        agent_type: PedestrianType
        custom_spawn: [x_px, y_px] for PANIC or CONFUSED replacements
        """
        def is_free(px, py):
            return Pedestrian([px,py], None).check_collision(px, py)

        # 1) pick a free spawn_px
        if agent_type in (PedestrianType.CALM, PedestrianType.CONFUSED):
            # if we're REPLACING (custom_spawn) use that
            if agent_type is PedestrianType.CONFUSED and custom_spawn:
                spawn_px = custom_spawn.copy()
            else:
                # otherwise sample one of your green zones
                spawn_px = None
                for _ in range(20):
                    sx, sy, sw, sh = self._get_random_source()
                    gx = random.uniform(sx, sx+sw)
                    gy = random.uniform(sy, sy+sh)
                    cand = [(gx+0.5)*GRID_SIZE, (gy+0.5)*GRID_SIZE]
                    if is_free(*cand):
                        spawn_px = cand
                        break
                if spawn_px is None:
                    sx, sy, sw, sh = self._get_random_source()
                    spawn_px = [ (sx+sw/2)*GRID_SIZE,
                                 (sy+sh/2)*GRID_SIZE ]
        else:  # PANIC
            if custom_spawn:
                spawn_px = custom_spawn.copy()
            else:
                sx, sy, sw, sh = self._get_random_source()
                spawn_px = [ (sx+sw/2)*GRID_SIZE, (sy+sh/2)*GRID_SIZE ]

            # nudge off any black‐pixel collision
            cols = WINDOW_WIDTH // GRID_SIZE
            for _ in range(cols):
                if is_free(*spawn_px): break
                spawn_px[0] -= GRID_SIZE

            # clamp on‐screen
            r = Pedestrian(spawn_px, None).radius
            spawn_px[0] = max(r, min(spawn_px[0], WINDOW_WIDTH-r))
            spawn_px[1] = max(r, min(spawn_px[1], WINDOW_HEIGHT-r))

        # 2) pick exit as target
        tx, ty, tw, th = self._get_random_target()
        gx2 = random.uniform(tx, tx+tw)
        gy2 = random.uniform(ty, ty+th)
        target_px = [(gx2+0.5)*GRID_SIZE, (gy2+0.5)*GRID_SIZE]

        # 3) instantiate correct class
        if agent_type is PedestrianType.CALM:
            agent = CalmPedestrian(spawn_px, target_px)
        elif agent_type is PedestrianType.CONFUSED:
            agent = ConfusedPedestrian(spawn_px, target_px)
        else:
            agent = PanicPedestrian(spawn_px, target_px)

        # 4) add to world
        self.agents.append(agent)

        # 5) if we just spawned a panic, immediately convert any blue in its vision
        if agent_type is PedestrianType.PANIC:
            for calm in list(self.agents):
                if isinstance(calm, CalmPedestrian):
                    dx = calm.pos[0]-agent.pos[0]
                    dy = calm.pos[1]-agent.pos[1]
                    if math.hypot(dx,dy) <= agent.vision_radius:
                        self._replace_agent(calm, PedestrianType.CONFUSED)
            # apply those replacements now:
            for old, new_t in self.pending_replacements:
                if old in self.agents:
                    self.agents.remove(old)
                    self.spawn_agent(new_t, custom_spawn=old.pos)
            self.pending_replacements.clear()


    # def update(self, dt):
    #     # 0) drop evacuees who reached their target
    #     self.agents = [
    #         a for a in self.agents
    #         if math.hypot(a.pos[0] - a.target[0], a.pos[1] - a.target[1]) > a.radius
    #     ]

    #     # 1) immediate Calm→Confused: any Calm within any Panic’s vision
    #     panics = [p for p in self.agents if isinstance(p, PanicPedestrian)]
    #     for calm in [a for a in self.agents if isinstance(a, CalmPedestrian)]:
    #         if any(
    #             math.hypot(calm.pos[0] - p.pos[0], calm.pos[1] - p.pos[1]) 
    #               <= p.vision_radius
    #             for p in panics
    #         ):
    #             self._replace_agent(calm, PedestrianType.CONFUSED)

    #     # 2) apply those conversions now (so next move sees the new types)
    #     for old, new_t in self.pending_replacements:
    #         if old in self.agents:
    #             self.agents.remove(old)
    #             self.spawn_agent(new_t, custom_spawn=old.pos)
    #     self.pending_replacements.clear()

    #     # 3) shared A* path planning for all Panics
    #     panics = [p for p in self.agents if isinstance(p, PanicPedestrian)]
    #     if panics:
    #         cx = sum(p.pos[0] for p in panics) / len(panics)
    #         cy = sum(p.pos[1] for p in panics) / len(panics)
    #         start = (int(cx // GRID_SIZE), int(cy // GRID_SIZE))

    #         tx, ty, *_ = self._get_random_target()
    #         goal_px = ((tx + 0.5) * GRID_SIZE, (ty + 0.5) * GRID_SIZE)
    #         goal = (int(goal_px[0] // GRID_SIZE), int(goal_px[1] // GRID_SIZE))

    #         shared_path = astar(get_grid(), start, goal) or []
    #         for p in panics:
    #             p.path = shared_path.copy()

    #     # 4) now move everyone (each move() handles its own collision logic)
    #     for a in self.agents:
    #         a.move(self.agents, get_grid_obstacles())

    #     # 5) handle interpersonal collisions & panic‐spreading
    #     self.handle_collisions()

    #     # 6) debug counts
    #     calm_c  = sum(isinstance(a, CalmPedestrian)    for a in self.agents)
    #     conf_c  = sum(isinstance(a, ConfusedPedestrian) for a in self.agents)
    #     panic_c = sum(isinstance(a, PanicPedestrian)    for a in self.agents)
    #     counts = (calm_c, conf_c, panic_c)
    #     if counts != self.last_counts:
    #         print(f"CALM={calm_c}  CONF={conf_c}  PANIC={panic_c}")
    #         self.last_counts = counts


    # def spawn_agent(self, agent_type, custom_spawn=None, custom_target=None):
    #     """
    #     agent_type: PedestrianType
    #     custom_spawn: [x_px, y_px] for PANIC only (click)
    #     """
    #     # helper to test a pixel for wall‐collision
    #     def is_free(px, py):
    #         probe = Pedestrian([px,py], None)
    #         return probe.check_collision(px, py)
        
    #     # 1) pick a free spawn_px
    #     if agent_type in (PedestrianType.CALM, PedestrianType.CONFUSED):
    #         spawn_px = None
    #         for _ in range(20):
    #             sx, sy, sw, sh = self._get_random_source()
    #             gx = random.uniform(sx, sx+sw)
    #             gy = random.uniform(sy, sy+sh)
    #             cand = [(gx+0.5)*GRID_SIZE, (gy+0.5)*GRID_SIZE]
    #             if is_free(*cand):
    #                 spawn_px = cand
    #                 break
    #         if spawn_px is None:
    #             # fallback to center of a source block
    #             sx, sy, sw, sh = self._get_random_source()
    #             spawn_px = [ (sx+sw/2)*GRID_SIZE,
    #                          (sy+sh/2)*GRID_SIZE ]
    #     else:  # PANIC
    #         if custom_spawn:
    #             spawn_px = custom_spawn.copy()
    #         else:
    #             sx, sy, sw, sh = self._get_random_source()
    #             spawn_px = [(sx+sw/2)*GRID_SIZE,
    #                         (sy+sh/2)*GRID_SIZE]
                

    #         # nudge left until we find free space
    #         cols = WINDOW_WIDTH // GRID_SIZE
    #         for _ in range(cols):
    #             if is_free(*spawn_px):
    #                 break
    #             spawn_px[0] -= GRID_SIZE
    #         # clamp inside window
    #         r = Pedestrian(spawn_px, None).radius
    #         spawn_px[0] = max(r, min(spawn_px[0], WINDOW_WIDTH-r))
    #         spawn_px[1] = max(r, min(spawn_px[1], WINDOW_HEIGHT-r))

    #     # 2) pick a random exit as target_px
    #     tx, ty, tw, th = self._get_random_target()
    #     gx2 = random.uniform(tx, tx+tw)
    #     gy2 = random.uniform(ty, ty+th)
    #     target_px = [(gx2+0.5)*GRID_SIZE, (gy2+0.5)*GRID_SIZE]

    #     # 3) instantiate
    #     if agent_type is PedestrianType.CALM:
    #         agent = CalmPedestrian(spawn_px, target_px)
    #     elif agent_type is PedestrianType.CONFUSED:
    #         agent = ConfusedPedestrian(spawn_px, target_px)
    #     else:  # PANIC
    #         agent = PanicPedestrian(spawn_px, target_px)

    #     # 4) add to world
    #     self.agents.append(agent)

    #     # 5) **immediately** turn any Calm inside this new Panic’s vision → Confused
    #     if agent_type is PedestrianType.PANIC:
    #         for calm in list(self.agents):
    #             if isinstance(calm, CalmPedestrian):
    #                 dx = calm.pos[0] - agent.pos[0]
    #                 dy = calm.pos[1] - agent.pos[1]
    #                 if math.hypot(dx, dy) <= agent.vision_radius:
    #                     self._replace_agent(calm, PedestrianType.CONFUSED)

    #         # flush them right away so they show up purple next frame
    #         for old, new_t in self.pending_replacements:
    #             if old in self.agents:
    #                 self.agents.remove(old)
    #                 self.spawn_agent(new_t, custom_spawn=old.pos)
    #         self.pending_replacements.clear()

    def panic_waves(self):
        for panic in [a for a in self.agents if isinstance(a, PanicPedestrian)]:
            for calm in [a for a in self.agents if isinstance(a, CalmPedestrian)]:
                dx = calm.pos[0] - panic.pos[0]
                dy = calm.pos[1] - panic.pos[1]
                if math.hypot(dx, dy) <= panic.vision_radius:
                    calm.hear_count += 1

    def draw(self, surface):
        for agent in self.agents:
            agent.draw(surface)

    def handle_collisions(self):
        for i, a in enumerate(self.agents):
            if isinstance(a, PanicPedestrian):
                self.spread_panic_radius(a)
            for b in self.agents[i+1:]:
                if self.in_direct_contact(a, b):
                    self._handle_state_interaction(a, b)

    def in_direct_contact(self, a, b):
        dx = a.pos[0] - b.pos[0]
        dy = a.pos[1] - b.pos[1]
        dist = (dx*dx + dy*dy) ** 0.5
        return dist < (a.get_radius() + b.get_radius())

    def _handle_state_interaction(self, a, b):
        if isinstance(a, PanicPedestrian) and isinstance(b, CalmPedestrian):
            self._replace_agent(b, PedestrianType.CONFUSED)
            return
        if isinstance(b, PanicPedestrian) and isinstance(a, CalmPedestrian):
            self._replace_agent(a, PedestrianType.CONFUSED)
            return
        if isinstance(a, PanicPedestrian) and isinstance(b, ConfusedPedestrian):
            self._replace_agent(b, PedestrianType.PANIC)
            return
        if isinstance(b, PanicPedestrian) and isinstance(a, ConfusedPedestrian):
            self._replace_agent(a, PedestrianType.PANIC)
            return
        if isinstance(a, ConfusedPedestrian) and isinstance(b, CalmPedestrian):
            self._replace_agent(b, PedestrianType.CONFUSED)
            return
        if isinstance(b, ConfusedPedestrian) and isinstance(a, CalmPedestrian):
            self._replace_agent(a, PedestrianType.CONFUSED)
            return

    def spread_panic_radius(self, panic_agent):
        from pedestrian import CalmPedestrian, ConfusedPedestrian

        vr = panic_agent.vision_radius
        pr = panic_agent.panic_radius

        for agent in self.agents:
            if agent is panic_agent:
                continue

            dx = panic_agent.pos[0] - agent.pos[0]
            dy = panic_agent.pos[1] - agent.pos[1]
            dist = (dx*dx + dy*dy)**0.5

            if isinstance(agent, CalmPedestrian) and dist <= vr:
                self._replace_agent(agent, PedestrianType.CONFUSED)
            elif isinstance(agent, ConfusedPedestrian) and dist <= pr:
                self._replace_agent(agent, PedestrianType.PANIC)

    def _replace_agent(self, old_agent, new_type):
        self.pending_replacements.append((old_agent, new_type))

    def _get_random_source(self):
        return random.choice(self.sources)

    def _get_random_target(self):
        return random.choice(self.targets)
