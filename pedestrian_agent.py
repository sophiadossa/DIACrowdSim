import pygame
from crowd_environment import run_environment, sources, targets, obstacles
from manager import PedestrianManager, PedestrianType


manager = PedestrianManager(sources, targets)

# Spawn initial agents
manager.spawn_agent(PedestrianType.CALM)
# manager.spawn_agent(PedestrianType.CONFUSED)
# manager.spawn_agent(PedestrianType.PANIC)

def simulation_draw(surface):
    manager.update(obstacles)
    manager.draw(surface)

if __name__ == "__main__":
    run_environment(simulation_draw)
