import random, math
from pathfinding import astar
from crowd_environment import get_grid_obstacles, GRID_SIZE
from pedestrian import Pedestrian, CalmPedestrian, ConfusedPedestrian, PanicPedestrian
from rl_agent import RLAgent
from pedestrian_type import PedestrianType

class PedestrianManager:
    def __init__(self, sources, targets):
        self.agents = []
        self.sources = sources
        self.targets = targets
        self.pending_replacements = []

    def _get_random_source(self):
        return random.choice(self.sources)

    def _get_random_target(self):
        return random.choice(self.targets)

    def _replace_agent(self, old, new_type, initial_state=None):
        self.pending_replacements.append((old, new_type, initial_state))

    def spawn_agent(self, agent_type, custom_spawn=None, initial_state=None):
        sx, sy, sw, sh = self._get_random_source()
        min_x, max_x = sx*GRID_SIZE, (sx+sw)*GRID_SIZE
        min_y, max_y = sy*GRID_SIZE, (sy+sh)*GRID_SIZE
        home = (min_x, max_x, min_y, max_y)

        # pick spawn point
        if custom_spawn:
            spawn_px = custom_spawn[:]
        else:
            spawn_px = None
            for _ in range(30):
                gx = random.uniform(sx, sx+sw)
                gy = random.uniform(sy, sy+sh)
                px, py = (gx+0.5)*GRID_SIZE, (gy+0.5)*GRID_SIZE
                if Pedestrian([px,py], None).check_collision(px, py):
                    spawn_px = [px, py]
                    break
            if spawn_px is None:
                spawn_px = [ (sx+sw/2)*GRID_SIZE, (sy+sh/2)*GRID_SIZE ]

        tx, ty, tw, th = self._get_random_target()
        target_px = [(random.uniform(tx,tx+tw)+0.5)*GRID_SIZE,
                     (random.uniform(ty,ty+th)+0.5)*GRID_SIZE]

        if agent_type is PedestrianType.CALM:
            agent = CalmPedestrian(spawn_px, target_px, home)
        elif agent_type is PedestrianType.CONFUSED:
            agent = ConfusedPedestrian(spawn_px, target_px)
        elif agent_type is PedestrianType.PANIC:
            agent = PanicPedestrian(spawn_px, target_px)
        else:  # RL
            agent = RLAgent(spawn_px, target_px)
            agent.home_zone = home

        if isinstance(agent, RLAgent) and initial_state:
            if initial_state == 'confused': agent.become_confused()
            if initial_state == 'panic':   agent.become_panicked()

        self.agents.append(agent)
        return agent

    def update(self, dt):
        # 0) apply pending replacements
        if self.pending_replacements:
            for old, new_t, state in self.pending_replacements:
                if old in self.agents:
                    self.agents.remove(old)
                    self.spawn_agent(new_t, custom_spawn=old.pos, initial_state=state)
            self.pending_replacements.clear()



        # 1) Remove exited
        survivors = []
        for a in self.agents:
            exited = False
            for tx, ty, tw, th in self.targets:
                x0, y0 = tx*GRID_SIZE, ty*GRID_SIZE
                if x0 <= a.pos[0] <= x0+tw*GRID_SIZE and y0 <= a.pos[1] <= y0+th*GRID_SIZE:
                    exited = True
                    break
            # if this is one of our two leaders, defer its exit until the crowd is empty
            if exited and isinstance(a, RLAgent) and getattr(a, 'is_leader', False):
                # only allow exit when _no_ other agents remain
                if any(not isinstance(b, RLAgent) or not getattr(b, 'is_leader', False)
                    for b in self.agents):
                    exited = False
            if not exited:
                survivors.append(a)
        self.agents = survivors

        # 2) queue contagion
        panics = [
            a for a in self.agents
            if getattr(a, "is_panicked", lambda: False)()
        ]
        for a in list(self.agents):
            if getattr(a, "is_calm", lambda: False)():
                if any(
                    math.hypot(a.pos[0]-p.pos[0], a.pos[1]-p.pos[1]) <= getattr(p, "vision_radius", 0)
                    for p in panics
                ):
                    self._replace_agent(a, PedestrianType.CONFUSED)
            elif getattr(a, "is_confused", lambda: False)():
                for p in panics:
                    if math.hypot(a.pos[0]-p.pos[0], a.pos[1]-p.pos[1]) <= getattr(p, "panic_radius", 0):
                        if isinstance(p, RLAgent):
                            p.on_infect()
                        self._replace_agent(a, PedestrianType.PANIC)
                        break

        # 3) flush replacements
        if self.pending_replacements:
            for old, new_t, state in self.pending_replacements:
                if old in self.agents:
                    self.agents.remove(old)
                    self.spawn_agent(new_t, custom_spawn=old.pos, initial_state=state)
            self.pending_replacements.clear()

        # 4) move everyone
        obstacles = get_grid_obstacles()
        for a in self.agents:
            a.move(self.agents, obstacles)

        # 5) collisions + bump contagion
        self.handle_collisions()

    def handle_collisions(self):
        n = len(self.agents)
        for i in range(n):
            a = self.agents[i]
            for j in range(i+1, n):
                b = self.agents[j]
                dx, dy = a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]
                dist = math.hypot(dx, dy)
                rsum = a.get_radius() + b.get_radius()
                if 0 < dist < rsum:
                    # if one is RLAgent and the other isn’t, only move the non-RL one
                    if isinstance(a, RLAgent) and not isinstance(b, RLAgent):
                        mover, stay = b, a
                    elif isinstance(b, RLAgent) and not isinstance(a, RLAgent):
                        mover, stay = a, b
                    else:
                        mover, stay = None, None

                    if mover is not None:
                        # compute half-overlap push
                        overlap = (rsum - dist)
                        ux, uy = dx/dist, dy/dist
                        # push `mover` away from `stay`
                        new_x = mover.pos[0] - ux * overlap
                        new_y = mover.pos[1] - uy * overlap
                        if mover.check_collision(new_x, new_y):
                            mover.pos = [new_x, new_y]
                        # skip the rest of this pair so RLAgent isn't moved
                        continue

                    # otherwise fall back to symmetric push
                    overlap = (rsum - dist)
                    ux, uy = dx/dist, dy/dist
                    a_nx = a.pos[0] + ux*(overlap*0.5)
                    a_ny = a.pos[1] + uy*(overlap*0.5)
                    b_nx = b.pos[0] - ux*(overlap*0.5)
                    b_ny = b.pos[1] - uy*(overlap*0.5)
                    if a.check_collision(a_nx, a_ny):
                        a.pos = [a_nx, a_ny]
                    if b.check_collision(b_nx, b_ny):
                        b.pos = [b_nx, b_ny]

        # bump‐contagion
        panics = [
            a for a in self.agents
            if getattr(a, "is_panicked", lambda: False)()
        ]
        for a in self.agents:
            for b in self.agents:
                if a is b: continue
                d = math.hypot(a.pos[0]-b.pos[0], a.pos[1]-b.pos[1])
                if d < (a.get_radius()+b.get_radius()):
                    # panic+calm
                    if getattr(a, "is_panicked", lambda: False)() and getattr(b, "is_calm", lambda: False)():
                        if isinstance(a, RLAgent): a.on_infect()
                        b.is_confused()
                    if getattr(b, "is_panicked", lambda: False)() and getattr(a, "is_calm", lambda: False)():
                        if isinstance(b, RLAgent): b.on_infect()
                        a.is_confused()
                    # panic+confused
                    if getattr(a, "is_panicked", lambda: False)() and getattr(b, "is_confused", lambda: False)():
                        if isinstance(a, RLAgent): a.on_infect()
                        b.is_panicked()
                    if getattr(b, "is_panicked", lambda: False)() and getattr(a, "is_confused", lambda: False)():
                        if isinstance(b, RLAgent): b.on_infect()
                        a.is_panicked()
                    # confused+calm
                    if getattr(a, "is_confused", lambda: False)() and getattr(b, "is_calm", lambda: False)():
                        b.is_confused()
                    if getattr(b, "is_confused", lambda: False)() and getattr(a, "is_calm", lambda: False)():
                        a.is_confused()

        # clamp any remaining calm to their home zones
        for a in self.agents:
            if hasattr(a, 'home_zone') and getattr(a, "is_calm", lambda: False)():
                min_x, max_x, min_y, max_y = a.home_zone
                a.pos[0] = max(min_x, min(a.pos[0], max_x))
                a.pos[1] = max(min_y, min(a.pos[1], max_y))

    def draw(self, surface):
        for a in self.agents:
            a.draw(surface)
