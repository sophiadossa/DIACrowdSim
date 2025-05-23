import sys
import random
import pygame
import crowd_environment as env
from crowd_environment import (
    run_environment,
    GRID_SIZE,
    get_grid,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    get_grid_obstacles,
    debug_draw_grid,
    dom,
    debug_draw_sources,
    load_floorplan,
)
from manager import PedestrianManager, PedestrianType


load_floorplan("floorplan1.png")

manager = PedestrianManager(env.crowd_zones, env.targets)

# scatter calm peds *everywhere* on free cells 
grid = get_grid()
cols, rows = len(grid), len(grid[0])

for _ in range(50):
    # choose one of your source‐zones at random
    zx, zy, zw, zh = random.choice(env.crowd_zones)
    # ensure we’re working in integer cell‐coords
    zx, zy, zw, zh = int(zx), int(zy), int(zw), int(zh)

    while True:
        # pick a random cell inside that rectangle
        cell_x = random.randint(zx, zx + zw - 1)
        cell_y = random.randint(zy, zy + zh - 1)

        # clamp to domain bounds just in case
        cell_x = max(0, min(cell_x, cols - 1))
        cell_y = max(0, min(cell_y, rows - 1))

        # only accept a walkable (grid=0) cell
        if grid[cell_x][cell_y] == 0:
            # convert to pixel position at cell center
            spawn_px = [
                (cell_x + 0.5) * GRID_SIZE,
                (cell_y + 0.5) * GRID_SIZE
            ]
            manager.spawn_agent(
                PedestrianType.RL)
            break
# Button config
# button_rect = pygame.Rect(650, 20, 130, 40)
# button_font = pygame.font.SysFont("arial", 18)

def simulation_draw(surface, dt):
    # panic button
    # pygame.draw.rect(surface, (255,100,100), button_rect)
    # label = button_font.render("SPAWN PANIC", True, (255,255,255))
    # surface.blit(label, (button_rect.x+10, button_rect.y+10))

    # debug_draw_obstacles(surface)
    # manager.update(get_grid_obstacles())
    # manager.draw(surface)
    
    debug_draw_sources(surface)
    manager.update(dt)
    manager.draw(surface)


    # for event in pygame.event.get():
    #     if event.type == pygame.MOUSEBUTTONDOWN and button_rect.collidepoint(event.pos):
    #         grid = get_grid()
    #         cols, rows = len(grid), len(grid[0])
    #         x_cell = cols - 2
    #         y_cell = random.randint(0, rows - 1)
    #         # ensure it’s walkable; if not, move one inwards
    #         if grid[x_cell][y_cell] == 1:
    #             x_cell = cols - 2
    #         spawn_px = [ (cols-2+0.5)*GRID_SIZE, random.randint(0,rows-1)+0.5*GRID_SIZE ]
    #         manager.spawn_agent(PedestrianType.RL,
    #                     custom_spawn=spawn_px,
    #                     initial_state='panic')

    # if event.type == pygame.QUIT:
    #     pygame.quit()
    #     sys.exit()

if __name__ == "__main__":
    run_environment(simulation_draw)
