from pedestrian import CalmPedestrian, PanicPedestrian, ConfusedPedestrian
import random
import sys
import math
from enum import Enum
from pathfinding import astar
from crowd_environment import get_grid, get_grid_obstacles, GRID_SIZE

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

    def spawn_agent(self, agent_type, custom_spawn=None, custom_target=None):
        """
        If custom_spawn is provided, it should be [x_px, y_px].
        Otherwise, we pick a random source (in grid coords) and convert to pixels.
        """
        if custom_spawn:
            spawn_px = [custom_spawn[0], custom_spawn[1]]
        else:
            sx, sy, sw, sh = self._get_random_source()
            gx = random.uniform(sx, sx + sw)
            gy = random.uniform(sy, sy + sh)
            spawn_px = [(gx + 0.5) * GRID_SIZE,
                        (gy + 0.5) * GRID_SIZE]

        tx, ty, tw, th = self._get_random_target()
        gx2 = random.uniform(tx, tx + tw)
        gy2 = random.uniform(ty, ty + th)
        target_px = [(gx2 + 0.5) * GRID_SIZE,
                     (gy2 + 0.5) * GRID_SIZE]

        if agent_type is PedestrianType.CALM:
            agent = CalmPedestrian(spawn_px, target_px)
        elif agent_type is PedestrianType.CONFUSED:
            agent = ConfusedPedestrian(spawn_px, target_px)
        else:
            agent = PanicPedestrian(spawn_px, target_px)

        self.agents.append(agent)

    def panic_waves(self):
        for panic in [a for a in self.agents if isinstance(a, PanicPedestrian)]:
            for calm in [a for a in self.agents if isinstance(a, CalmPedestrian)]:
                dx = calm.pos[0] - panic.pos[0]
                dy = calm.pos[1] - panic.pos[1]
                if math.hypot(dx, dy) <= panic.vision_radius:
                    calm.hear_count += 1

    def update(self, obstacles):
        # 0) remove evacuees
        survivors = []
        for a in self.agents:
            tx, ty = a.target
            if math.hypot(a.pos[0]-tx, a.pos[1]-ty) > a.radius:
                survivors.append(a)
        self.agents = survivors

        # 1) broadcast panic waves
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self.panic_waves()
            self.wave_timer = 5

        # 2) convert based on hear_count
        for a in list(self.agents):
            if isinstance(a, CalmPedestrian):
                if a.hear_count >= 2:
                    self._replace_agent(a, PedestrianType.PANIC)
                elif a.hear_count == 1:
                    self._replace_agent(a, PedestrianType.CONFUSED)

        for old, _ in self.pending_replacements:
            if isinstance(old, CalmPedestrian):
                old.hear_count = 0

        # 3) shared A* planning every tick
        panics = [a for a in self.agents if isinstance(a, PanicPedestrian)]
        cx = cy = None
        if panics:
            cx = sum(p.pos[0] for p in panics) / len(panics)
            cy = sum(p.pos[1] for p in panics) / len(panics)
            start = (int(cx // GRID_SIZE), int(cy // GRID_SIZE))

            tx, ty, *_ = self._get_random_target()
            goal_px = ((tx + 0.5) * GRID_SIZE, (ty + 0.5) * GRID_SIZE)
            goal = (int(goal_px[0] // GRID_SIZE), int(goal_px[1] // GRID_SIZE))

            shared = astar(get_grid(), start, goal)
            for p in panics:
                p.path = shared.copy()

        # 4) move all agents
        for agent in self.agents:
            agent.move(self.agents, obstacles)

        # 5) handle collisions
        self.handle_collisions()

        # 6) apply replacements
        for old_agent, new_type in self.pending_replacements:
            if old_agent in self.agents:
                self.agents.remove(old_agent)
                self.spawn_agent(new_type, custom_spawn=old_agent.pos)
        self.pending_replacements.clear()

        # 7) debug counts
        calm_count = sum(isinstance(a, CalmPedestrian) for a in self.agents)
        conf_count = sum(isinstance(a, ConfusedPedestrian) for a in self.agents)
        panic_count = sum(isinstance(a, PanicPedestrian) for a in self.agents)
        counts = (calm_count, conf_count, panic_count)
        if counts != self.last_counts:
            sys.stdout.write(f"CALM={calm_count} CONF={conf_count} PANIC={panic_count}\n")
            sys.stdout.flush()
            self.last_counts = counts

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
