import pygame
import sys
from PIL import Image
import numpy as np
from cromosim.domain import Domain, Destination

# 1) Pygame + constants
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
GRID_SIZE      = 40
BG_COLOR       = (255, 255, 255)
SOURCE_COLOR   = (34, 177,  76)
EXIT_COLOR     = (237,  28,  36)
WALL_COLOR     = (0,    0,    0)

# how many cells across/down
cols = WINDOW_WIDTH  // GRID_SIZE  # e.g. 20
rows = WINDOW_HEIGHT // GRID_SIZE  # e.g. 15

# 2) Load the full-detail PNG and also make a tiny 20×15 version
img_full = Image.open("floorplan.png").convert("RGB")
img_small = img_full.resize((cols, rows), Image.NEAREST)
raw = np.array(img_small)
r, g, b = raw[:,:,0], raw[:,:,1], raw[:,:,2]

DARK_THRESH = 50

# 3) Derive grid‐level lists from the tiny image:
#    A) occupancy grid
grid = [[0]*rows for _ in range(cols)]
LUM_THRESH = 500
wall_mask = (r + g+ b) < LUM_THRESH
for y, x in zip(*np.where(wall_mask)):
    grid[x][y] = 1

#    B) exit cells
exit_mask = (r==EXIT_COLOR[0]) & (g==EXIT_COLOR[1]) & (b==EXIT_COLOR[2])
targets = [(int(x), int(y), 1, 1) for y,x in zip(*np.where(exit_mask))]
if not targets:
    # fallback if your red wasn’t exact
    targets = [(1, 0, 1, 1), (1, rows-1, 1, 1)]

#    C) (optional) derive source cells from green
source_mask = (r==SOURCE_COLOR[0]) & (g==SOURCE_COLOR[1]) & (b==SOURCE_COLOR[2])
sources = [(int(x), int(y), 1, 1) for y,x in zip(*np.where(source_mask))]
if not sources:
    # fallback to your old hard-coded bars
    sources = [
        (3.6, 5.8, 5,   1),
        (9.6, 5.8, 9.8, 1),
        (9.6, 7.7, 9.8, 1),
        (3.6, 7.7, 5,   1),
    ]

#    D) manual obstacles (if you still need rectangular walls)
obstacles = [
    (2, 7, 18, 0.5),
    (2, 7, 8, 0.5),
    (14, 7, 6, 0.5),
    (19.6, 0, 0.4, 17),
    (9, 3, 0.4, 9)
]

# 4) Build Cromosim Domain off the tiny image
dom = Domain(
    name="env",
    pixel_size=GRID_SIZE,
    width=cols,
    height=rows,
    wall_colors=[list(WALL_COLOR)],
    background="floorplan.png"
)
dom.build_domain()
# register exits so Cromosim’s distance-map works too
dest = Destination(
    name="exit",
    colors=[list(EXIT_COLOR)],
    excluded_colors=[list(WALL_COLOR)],
    desired_velocity_from_color=[],
    velocity_scale=1.0
)
dom.add_destination(dest)

# 5) Pre-load & up-scale the full-detail image into Pygame
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Crowd Simulation Environment")
clock = pygame.time.Clock()

_pg_floorplan = pygame.image.load("floorplan.png").convert()
_pg_floorplan = pygame.transform.scale(
    _pg_floorplan,
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)

def draw_rects(rects, color):
    """Draws list of grid (x,y,w,h) rects into Pygame coords."""
    for x, y, w, h in rects:
        pygame.draw.rect(
            window,
            color,
            (x*GRID_SIZE, y*GRID_SIZE, w*GRID_SIZE, h*GRID_SIZE)
        )

def get_grid():
    """Return occupancy grid as 2D list: 1=wall, 0=free."""
    grid = [[0 for _ in range(rows)] for _ in range(cols)]
    for x in range(cols):
        for y in range(rows):
            if (r[y, x] < DARK_THRESH
            and  g[y, x] < DARK_THRESH
            and  b[y, x] < DARK_THRESH):
                grid[x][y] = 1
    return grid

def get_grid_obstacles():
    """Return manual obstacle rectangles for your collision code."""
    return obstacles

def debug_draw_grid(surface):
    """Outline each occupied cell for debugging."""
    grid = get_grid()
    for x in range(cols):
        for y in range(rows):
            if grid[x][y] == 1:
                pygame.draw.rect(
                    surface,
                    (200,200,200),
                    (x*GRID_SIZE, y*GRID_SIZE,
                     GRID_SIZE,    GRID_SIZE),
                    1
                )

def run_environment(custom_draw_fn=None):
    """
    1) blit the full-detail background
    2) overlay manual walls, exits, sources
    3) call your agents’ draw/update
    """
    while True:
        window.fill(BG_COLOR)
        window.blit(_pg_floorplan, (0,0))

        if custom_draw_fn:
            custom_draw_fn(window)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        clock.tick(60)
