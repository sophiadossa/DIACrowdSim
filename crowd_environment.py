import pygame
import sys

# Init
pygame.init()

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
GRID_SIZE = 40
BG_COLOR = (255, 255, 255)
SOURCE_COLOR = (0, 200, 0)
TARGET_COLOR = (200, 0, 0)
OBSTACLE_COLOR = (120, 120, 120)

# Environment layout (x, y, width, height) in grid units

# Red targets in top-left and bottom-left corners
targets = [
    (1, 0.3, 1, 1),
    (1, 14, 1, 1)
]

# Green source blocks in 4 segments (with padding)
sources = [
    (9.6, 5.8, 5, 1),  # left segment slim source (horizontal)
    (15.7, 5.8, 4, 1),   # right segment slim source (horizontal)
    (9.6, 7.7, 5, 1),  # left segment slim source (horizontal)
    (15.7, 7.7, 4, 1),   # right segment slim source (horizontal)
    (6.8, 5.8, 2, 1),  # left segment slim source (horizontal)
    (6.8, 7.7, 2, 1)   # right segment slim source (horizontal)
]


# Obstacles: centered horizontal and two verticals
obstacles = [
    (2, 7, 18, 0.5),  # horizontal center

    (15, 2, 0.4, 11),  # vertical right
    (9, 3, 0.4, 9)  # vertical lwft
]


window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Crowd Simulation Environment")
clock = pygame.time.Clock()

def draw_rects(rects, color):
    for x, y, w, h in rects:
        pygame.draw.rect(window, color, (x * GRID_SIZE, y * GRID_SIZE, w * GRID_SIZE, h * GRID_SIZE))

def draw_bottleneck(x, y, flipped=False):
    wall_width = 0.2
    corridor_length = 1.5
    polygon_offset = 1.0  # Make polygons stretch wider/larger

    # --- DRAW CORRIDORS ---
    if flipped:
        # Bottom-left target → corridors go up, shifted left of target
        corridors = [
            (x - 1, y - corridor_length, wall_width, corridor_length),
            (x - 1 + 1 - wall_width, y - corridor_length, wall_width, corridor_length)
        ]
    else:
        # Top-left target → corridors go downward
        corridors = [
            (x, y + 1, wall_width, corridor_length),
            (x + 1 - wall_width, y + 1, wall_width, corridor_length)
        ]

    draw_rects(corridors, OBSTACLE_COLOR)

    # --- DRAW POLYGONAL SIDE BLOCKERS ---
    if flipped:
        # Bottom-left (same as before)
        pygame.draw.polygon(window, OBSTACLE_COLOR, [
            ((x - 1) * GRID_SIZE, y * GRID_SIZE),
            ((x - 1 - polygon_offset) * GRID_SIZE, (y + 1) * GRID_SIZE),
            ((x - 1) * GRID_SIZE, (y + 1) * GRID_SIZE)
        ])
        pygame.draw.polygon(window, OBSTACLE_COLOR, [
            ((x) * GRID_SIZE, y * GRID_SIZE),
            ((x + polygon_offset) * GRID_SIZE, (y + 1) * GRID_SIZE),
            ((x) * GRID_SIZE, (y + 1) * GRID_SIZE)
        ])
    else:
        # Top-left (now FLIPPED to match bottom bottleneck style)
        pygame.draw.polygon(window, OBSTACLE_COLOR, [
            (x * GRID_SIZE, (y + 1) * GRID_SIZE),  # bottom left
            ((x - polygon_offset) * GRID_SIZE, y * GRID_SIZE),  # upper-left point
            (x * GRID_SIZE, y * GRID_SIZE)  # top left
        ])
        pygame.draw.polygon(window, OBSTACLE_COLOR, [
            ((x + 1) * GRID_SIZE, (y + 1) * GRID_SIZE),  # bottom right
            ((x + 1 + polygon_offset) * GRID_SIZE, y * GRID_SIZE),  # upper-right point
            ((x + 1) * GRID_SIZE, y * GRID_SIZE)  # top right
        ])


def main():
    # 
    while True:
        window.fill(BG_COLOR)
        
        draw_rects(obstacles, OBSTACLE_COLOR)
        draw_bottleneck(1, 0.3, flipped = False)
        draw_bottleneck(2, 13.5, flipped =True)

        draw_rects(targets, TARGET_COLOR)
        draw_rects(sources, SOURCE_COLOR)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        clock.tick(60)

# creates 2D array (gri[x][y]) where 0 is walkable and 1 is obstacle
def get_grid():
    cols = WINDOW_WIDTH // GRID_SIZE
    rows = WINDOW_HEIGHT // GRID_SIZE
    grid = [[0 for _ in range(rows)] for _ in range(cols)]

    # Mark obstacles as 1
    for ox, oy, ow, oh in obstacles:
        for x in range(int(ox * 10), int((ox + ow) * 10)):
            for y in range(int(oy * 10), int((oy + oh) * 10)):
                gx = int(x / 10)
                gy = int(y / 10)
                if 0 <= gx < cols and 0 <= gy < rows:
                    grid[gx][gy] = 1

    return grid

def run_environment(custom_draw_fn=None):
    while True:
        window.fill(BG_COLOR)
        
        draw_rects(obstacles, OBSTACLE_COLOR)
        draw_bottleneck(1, 0.3, flipped=False)
        draw_bottleneck(2, 14, flipped=True)

        draw_rects(targets, TARGET_COLOR)
        draw_rects(sources, SOURCE_COLOR)

        if custom_draw_fn:
            custom_draw_fn(window)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        clock.tick(60)
