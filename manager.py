from pedestrian import CalmPedestrian
import random
from enum import Enum
import pygame

class PedestrianType(Enum):
    CALM = 4
    CONFUSED = 1
    PANIC = 2

class PedestrianManager:
    def __init__(self, sources, targets):
        self.agents = []
        self.sources = sources
        self.targets = targets

    def spawn_agent(self, agent_type):
        spawn = self._get_random_source()
        target = self._get_random_target()
        spawn_px = [(spawn[0] + 0.5) * 40, (spawn[1] + 0.5) * 40]
        target_px = [(target[0] + 0.5) * 40, (target[1] + 0.5) * 40]

        if agent_type == PedestrianType.CALM:
            agent = CalmPedestrian(spawn_px, target_px)
        # elif agent_type == PedestrianType.CONFUSED:
        #     agent = ConfusedPedestrian(spawn_px, target_px)
        # else:
        #     agent = PanicPedestrian(spawn_px, target_px)

        self.agents.append(agent)

    def update(self, obstacles):
        for agent in self.agents:
            if isinstance(agent, CalmPedestrian):
                agent.move(obstacles)
            else:
                agent.move()

        self.handle_collisions()

    def draw(self, surface):
        for agent in self.agents:
            agent.draw(surface)

    def handle_collisions(self):
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                a = self.agents[i]
                b = self.agents[j]
                dist = ((a.pos[0] - b.pos[0]) ** 2 + (a.pos[1] - b.pos[1]) ** 2) ** 0.5
                if dist < a.get_radius() + b.get_radius():
                    self._handle_state_interaction(a, b)

    # def _handle_state_interaction(self, a, b):
    #     types = (type(a), type(b))
    #     if PanicPedestrian in types:
    #         if isinstance(a, CalmPedestrian):
    #             self._replace_agent(a, PedestrianType.PANIC)
    #         elif isinstance(b, CalmPedestrian):
    #             self._replace_agent(b, PedestrianType.PANIC)
    #         if isinstance(a, ConfusedPedestrian):
    #             self._replace_agent(a, PedestrianType.PANIC)
    #         elif isinstance(b, ConfusedPedestrian):
    #             self._replace_agent(b, PedestrianType.PANIC)
    #     elif isinstance(a, ConfusedPedestrian) and isinstance(b, CalmPedestrian):
    #         self._replace_agent(b, PedestrianType.CONFUSED)
    #     elif isinstance(b, ConfusedPedestrian) and isinstance(a, CalmPedestrian):
    #         self._replace_agent(a, PedestrianType.CONFUSED)

    def _replace_agent(self, old_agent, new_type):
        self.agents.remove(old_agent)
        self.spawn_agent(new_type)

    def _get_random_source(self):
        return random.choice(self.sources)

    def _get_random_target(self):
        return random.choice(self.targets)
