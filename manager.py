import random, math
from pathfinding import astar
from crowd_environment import get_grid, get_grid_obstacles, GRID_SIZE, dom, WINDOW_HEIGHT, WINDOW_WIDTH, FLOORPLAN_SURFACE
from pedestrian import Pedestrian, CalmPedestrian, ConfusedPedestrian, PanicPedestrian
from rl_agent import RLAgent
from pedestrian_type import PedestrianType

class PedestrianManager:
    def __init__(self, sources, targets):
        self.agents = []
        self.sources = sources
        self.targets = targets
        self.pending_replacements = []
        self.last_counts = (-1, -1, -1, -1)
        self.free_spawn_sites = None

    def _get_random_source(self):
        return random.choice(self.sources)
    def _get_random_target(self):
        return random.choice(self.targets)

    def _replace_agent(self, old, new_type, initial_state=None):
        """Queue a replacement of `old` with a NEW RLAgent in state `new_type`."""
        self.pending_replacements.append((old, new_type, initial_state))


    def spawn_agent(self, agent_type, custom_spawn=None, initial_state=None):
        """Spawn exactly one Pedestrian (Calm/Confused/Panic) or RLAgent,
        always inside one of the green source zones, and record its home_zone."""
        # 0) choose a source‐zone (in grid coords) and record it
        sx, sy, sw, sh = self._get_random_source()
        min_x = sx     * GRID_SIZE
        max_x = (sx+sw)* GRID_SIZE
        min_y = sy     * GRID_SIZE
        max_y = (sy+sh)* GRID_SIZE
        home = (min_x, max_x, min_y, max_y)

        # 1) pick spawn_px: if custom, use it (for replacements/panic),
        if custom_spawn:
            spawn_px = list(custom_spawn)
        else:
            spawn_px = None
            for _ in range(30):
                gx = random.uniform(sx, sx+sw)
                gy = random.uniform(sy, sy+sh)
                px = (gx + 0.5) * GRID_SIZE
                py = (gy + 0.5) * GRID_SIZE
                if Pedestrian([px,py], None).check_collision(px, py):
                    spawn_px = [px, py]
                    break
            if spawn_px is None:
                # fallback to geometric center
                px = (sx + sw/2) * GRID_SIZE
                py = (sy + sh/2) * GRID_SIZE
                spawn_px = [px, py]

        # 2) pick a random exit target as before
        tx, ty, tw, th = self._get_random_target()
        gx2 = random.uniform(tx, tx+tw)
        gy2 = random.uniform(ty, ty+th)
        target_px = [(gx2+0.5)*GRID_SIZE, (gy2+0.5)*GRID_SIZE]

        # 3) instantiate the right class, passing home into Calm/RL
        if agent_type is PedestrianType.CALM:
            agent = CalmPedestrian(spawn_px, target_px, home)
        elif agent_type is PedestrianType.CONFUSED:
            agent = ConfusedPedestrian(spawn_px, target_px)
        elif agent_type is PedestrianType.PANIC:
            agent = PanicPedestrian(spawn_px, target_px)
        else:  # RL
            agent = RLAgent(spawn_px, target_px)
            agent.home_zone = home

        # 4) optionally force RL initial state
        if isinstance(agent, RLAgent):
            if initial_state == 'confused':
                agent.become_confused()
            elif initial_state == 'panic':
                agent.become_panicked()

        # # 5) add to world and handle immediate contagion if panic
        # self.agents.append(agent)
        # if isinstance(agent, PanicPedestrian) or (isinstance(agent, RLAgent) and agent.is_panicked()):
        #     for other in list(self.agents):
        #         if hasattr(other, "is_calm") and other.is_calm():
        #             dx = other.pos[0] - agent.pos[0]
        #             dy = other.pos[1] - agent.pos[1]
        #             if math.hypot(dx, dy) <= agent.vision_radius:
        #                 if isinstance(agent, RLAgent):
        #                     agent.on_infect()
        #                 self._replace_agent(other, PedestrianType.CONFUSED)

        # # 6) flush replacements right away (keeping same zone)
        # for old, new_t, istate in self.pending_replacements:
        #     if old in self.agents:
        #         self.agents.remove(old)
        #         self.spawn_agent(
        #             PedestrianType.RL,
        #             custom_spawn=old.pos,
        #             initial_state=('panic' if new_t==PedestrianType.PANIC else 'confused')
        #         )
        # self.pending_replacements.clear()
        self.agents.append(agent)

        return agent

    def update(self, dt):

            # ─── apply any pending replacements now ───────────────────────
        if self.pending_replacements:
            for old, new_t, istate in self.pending_replacements:
                if old in self.agents:
                    self.agents.remove(old)
                    # spawn the new RLAgent
                    new = self.spawn_agent(
                        PedestrianType.RL,
                        custom_spawn=old.pos,
                        initial_state=('panic' if new_t==PedestrianType.PANIC else 'confused')
                    )
            self.pending_replacements.clear()
    # ───────────────────────────────────────────────────────────────
        new_agents = []
        for a in self.agents:
            exited = False
            for tx,ty,tw,th in self.targets:
                x0 = tx * GRID_SIZE
                y0 = ty * GRID_SIZE
                w = tw *GRID_SIZE
                h = th * GRID_SIZE
                if x0 <= a.pos[0] <= x0 + w and y0 <= a.pos[1] <= y0 + h:
                    exited = True
                    break
            if not exited:
                new_agents.append(a)
        self.agents = new_agents

        # 1) contagion: Calm→Confused by vision, Confused→Panic by panic_radius
        panics = [a for a in self.agents if a.is_panicked()]
        for a in list(self.agents):
            if a.is_calm():
            # calm→confused still happens, but no RL bonus
                if any(math.hypot(a.pos[0]-p.pos[0],
                                a.pos[1]-p.pos[1]) <= p.vision_radius
                    for p in panics):
                    self._replace_agent(a, PedestrianType.CONFUSED)
                elif a.is_confused():
                    # confused→panic via panic_radius, *this* is where RL gets infected bonus
                    for p in panics:
                        if math.hypot(a.pos[0]-p.pos[0],
                                    a.pos[1]-p.pos[1]) <= p.panic_radius:
                            # only count true panic contagion
                            if isinstance(p, RLAgent):
                                p.on_infect()
                            self._replace_agent(a, PedestrianType.PANIC)
                            break

        # 2) apply replacements immediately
        for old, new_t, istate in self.pending_replacements:
            if old in self.agents:
                self.agents.remove(old)
                self.spawn_agent(PedestrianType.RL, custom_spawn=old.pos,
                                 initial_state=('panic' if new_t==PedestrianType.PANIC
                                                else 'confused'))
        self.pending_replacements.clear()

        # 3) move everyone
        for a in self.agents:
            a.move(self.agents, get_grid_obstacles())

        # 4) collisions → physical separation + further contagion
        self.handle_collisions()

        # 5) debug counts
        # c0 = sum(a.is_calm()     for a in self.agents)
        # c1 = sum(a.is_confused() for a in self.agents)
        # c2 = sum(a.is_panicked() for a in self.agents)
        # c3 = len(self.agents) - (c0+c1+c2)
        # counts = (c0,c1,c2,c3)
        # if counts != self.last_counts:
        #     print(f"CALM={c0}  CONF={c1}  PANIC={c2}  OTHER={c3}")
        #     self.last_counts = counts

    def handle_collisions(self):
        """Push overlapping agents apart (never through walls),
        then apply contagion and clamp calm agents to their home zones."""
        n = len(self.agents)
        for i in range(n):
            a = self.agents[i]
            for j in range(i+1, n):
                b = self.agents[j]
                dx = a.pos[0] - b.pos[0]
                dy = a.pos[1] - b.pos[1]
                dist = math.hypot(dx, dy)
                rsum = a.get_radius() + b.get_radius()

                if 0 < dist < rsum:
                    # 1) register a panicked–panicked collision
                    if a.is_panicked() and b.is_panicked():
                        if isinstance(a, RLAgent): a.on_collision()
                        if isinstance(b, RLAgent): b.on_collision()

                    # 2) compute the half-overlap push
                    overlap = (rsum - dist)
                    ux, uy = dx/dist, dy/dist

                    # candidate new positions
                    a_nx = a.pos[0] + ux*(overlap*0.5)
                    a_ny = a.pos[1] + uy*(overlap*0.5)
                    b_nx = b.pos[0] - ux*(overlap*0.5)
                    b_ny = b.pos[1] - uy*(overlap*0.5)

                    # 3) only move if that spot is obstacle-free
                    if a.check_collision(a_nx, a_ny):
                        a.pos[0], a.pos[1] = a_nx, a_ny
                    if b.check_collision(b_nx, b_ny):
                        b.pos[0], b.pos[1] = b_nx, b_ny

        # now re-apply contagion pairwise as before
        panics = [a for a in self.agents if a.is_panicked()]
        for a in self.agents:
            for b in self.agents:
                if a is b: continue
                d = math.hypot(a.pos[0]-b.pos[0], a.pos[1]-b.pos[1])
                if d < (a.get_radius()+b.get_radius()):
                    # panic+calm
                    if a.is_panicked() and b.is_calm():
                        if isinstance(a, RLAgent):
                            a.on_infect()
                        b.become_confused()
                    if b.is_panicked() and a.is_calm():
                        if isinstance(b, RLAgent):
                            b.on_infect()
                        a.become_confused()
                    # panic+confused
                    if a.is_panicked() and b.is_confused():
                        if isinstance(a, RLAgent):
                            a.on_infect()
                        b.become_panicked()
                    if b.is_panicked() and a.is_confused():
                        if isinstance(b, RLAgent):
                            b.on_infect()
                        a.become_panicked()
                    # confused+calm
                    if a.is_confused() and b.is_calm():
                        b.become_confused()
                    if b.is_confused() and a.is_calm():
                        a.become_confused()

        # finally clamp calm pedestrians back into their home zones
        for a in self.agents:
            if hasattr(a, 'home_zone') and a.is_calm():
                min_x, max_x, min_y, max_y = a.home_zone
                a.pos[0] = max(min_x, min(a.pos[0], max_x))
                a.pos[1] = max(min_y, min(a.pos[1], max_y))


            # clamp calm agents to their home zones
            for a in self.agents:
                if hasattr(a, 'home_zone') and a.is_calm():
                    min_x, max_x, min_y, max_y = a.home_zone
                    a.pos[0] = max(min_x, min(a.pos[0], max_x))
                    a.pos[1] = max(min_y, min(a.pos[1], max_y))

    def draw(self, surface):
        for a in self.agents:
            a.draw(surface)
