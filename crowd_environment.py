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
LUM_THRESH = 200
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

# Manual obstacles defined in GRID‐CELLS (x, y, width, height)
# obstacles = [
#     # ── CENTRAL HORIZONTAL BAR ────────────────────────────────
#     # spans from cell x=2.0 to x=20.0, at y≈7.0, thickness ≈0.5
#     (4.8, 6.5, 15.0, 0.5),

#     # ── CENTRAL VERTICAL BAR ──────────────────────────────────
#     # spans from cell y=2.5 to y=11.5, at x≈9.95, thickness ≈0.5
#     (10.5, 2.25, 0.5, 10.9),

#     # RIGHT STAGE LINE
#     (19.5, 0, 0.5, 25.0),

#     # ── TOP‐LEFT EXIT BOTTLENECK ────────────────────────────
#     # vertical corridors: must span the full 2 cells of red exit + a bit extra
#     (1.2, 0.8, 0.2, 2.4),   # left corridor
#     (2.7, 0.8, 0.2, 2.4),   # right corridor
#     # vertical blocker posts: exactly 1 cell tall
#     (1.2, 0.0, 0.2, 1.0),   # left blocker
#     (2.7, 0.0, 0.2, 1.0),   # right blocker

#     # ── BOTTOM‐LEFT EXIT BOTTLENECK ─────────────────────────
#     # vertical corridors: same length, positioned just above the bottom exit
#     (1.2, 12.6, 0.2, 2.4),  # left corridor
#     (2.7, 12.6, 0.2, 2.4),  # right corridor
#     # blocker posts beside the bottom exit
#     (1.2, 14.0, 0.2, 1.0),  # left blocker
#     (2.7, 14.0, 0.2, 1.0),  # right blocker
# ]

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

FLOORPLAN_SURFACE = _pg_floorplan

def draw_rects(rects, color):
    """Draws list of grid (x,y,w,h) rects into Pygame coords."""
    for x, y, w, h in rects:
        pygame.draw.rect(
            window,
            color,
            (x*GRID_SIZE, y*GRID_SIZE, w*GRID_SIZE, h*GRID_SIZE)
        )

def get_grid():
    raw = np.array(img_small)            # shape (rows,cols,3)
    r, g, b = raw[:,:,0], raw[:,:,1], raw[:,:,2]

    # any “near black” pixel is a wall
    wall = (r <  50) & (g <  50) & (b <  50)

    grid = [[1 if wall[y,x] else 0
             for y in range(rows)]
            for x in range(cols)]
    return grid


def get_grid_obstacles():
    """Return manual obstacle rectangles for your collision code."""
    # return obstacles
    return []

def debug_draw_grid(surface, padding_frac=0.1):
    """Outline each occupied cell for debugging."""
    pad = GRID_SIZE * padding_frac
    cell = GRID_SIZE - 2 * pad
    grid = get_grid()
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == 1:
                rect = pygame.Rect(
                    x*GRID_SIZE + pad,
                    y*GRID_SIZE + pad,
                    cell, cell
                )
                pygame.draw.rect(surface, (200,200,200), rect, 1)

def run_environment(custom_draw_fn=None):
    """
    Each frame:
      1) cap the frame‐rate & get dt
      2) draw background
      3) call simulation_draw(window, dt)
    """
    while True:
        # 1) cap at 60fps, and get milliseconds since last tick
        dt_ms = clock.tick(60)
        dt = dt_ms / 1000.0    # convert to seconds

        # 2) draw static background
        window.fill(BG_COLOR)
        window.blit(_pg_floorplan, (0, 0))

        # 3) hand off to your simulation, with dt
        if custom_draw_fn:
            custom_draw_fn(window, dt)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()



