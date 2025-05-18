import pygame
import sys
from PIL import Image
import numpy as np
from cromosim.domain import Domain, Destination

# ─── PYGAME + CONSTANTS ──────────────────────────────────────
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
GRID_SIZE      = 40
BG_COLOR       = (255, 255, 255)
SOURCE_COLOR   = (34, 177,  76)
EXIT_COLOR     = (237,  28,  36)
WALL_COLOR     = (0,    0,    0)

cols = WINDOW_WIDTH  // GRID_SIZE
rows = WINDOW_HEIGHT // GRID_SIZE

# globals we rebuild in load_floorplan()
img_full = None
img_small = None
_raw     = None
grid     = None
targets  = []   # list of (tx,ty,tw,th) exit rectangles
dom      = None
window   = None
clock    = None
_pg_floorplan = None
FLOORPLAN_SURFACE = None

#(x,y,width,height)
crowd_zones = [
    (4.8, 2.2, 5.5, 4), (11.4,2.2,8,4),
    (4.8, 7.5, 5.5, 4), (11.4,7.5,8,4), 
]

def load_floorplan(path: str):
    """
    Load a new floorplan PNG, rebuild:
     - img_full, img_small, raw pixel arrays
     - grid occupancy
     - source‐zones (clustering green cells into rects)
     - targets (red exit cells)
     - cromosim Domain + Destination
     - pygame surfaces
    """
    global img_full, img_small, _raw, grid, sources, targets
    global dom, window, clock, _pg_floorplan, FLOORPLAN_SURFACE

    # 1) load and downscale
    img_full  = Image.open(path).convert("RGB")
    img_small = img_full.resize((cols, rows), Image.NEAREST)
    raw = np.array(img_small)
    r, g, b = raw[:,:,0], raw[:,:,1], raw[:,:,2]
    _raw = raw

    # 2) build occupancy grid
    wall_mask = (r<50)&(g<50)&(b<50)
    grid = [[1 if wall_mask[y,x] else 0
             for y in range(rows)]
            for x in range(cols)]

    # 3) find exits (red) → single‐cell rectangles
    exit_mask = (r==EXIT_COLOR[0]) & (g==EXIT_COLOR[1]) & (b==EXIT_COLOR[2])
    targets = [(int(x), int(y), 1, 1) for y,x in zip(*np.where(exit_mask))]
    if not targets:
        # fallback: whole left and right edge
        targets = [(0, 0, 1, rows), (cols-1, 0, 1, rows)]

    sources = [
        (sx, sy, sw, sh)
        for (sx, sy, sw, sh) in crowd_zones
    ]
    # 5) build cromosim Domain
    dom = Domain(
        name="env",
        pixel_size=GRID_SIZE,
        width=cols,
        height=rows,
        wall_colors=[list(WALL_COLOR)],
        background=path
    )
    dom.build_domain()
    dest = Destination(
        name="exit",
        colors=[list(EXIT_COLOR)],
        excluded_colors=[list(WALL_COLOR)],
        desired_velocity_from_color=[],
        velocity_scale=1.0
    )
    dom.add_destination(dest)

    # 6) set up Pygame
    if window is None:
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Crowd Simulation Environment")
        clock = pygame.time.Clock()
    _pg_floorplan = pygame.image.load(path).convert()
    _pg_floorplan = pygame.transform.scale(_pg_floorplan,
                                           (WINDOW_WIDTH, WINDOW_HEIGHT))
    FLOORPLAN_SURFACE = _pg_floorplan


def get_grid():
    """Return the latest occupancy grid (1=wall, 0=free)."""
    return grid


def get_grid_obstacles():
    """No additional manual obstacles; walls come from image."""
    return []


def debug_draw_sources(surface, color=(0,255,0), alpha=80, border=2):
    """
    Draw each (sx,sy,sw,sh) in 'sources' as a translucent rectangle.
    """
    fill = pygame.Surface((1,1), pygame.SRCALPHA)
    fill.fill((*color, alpha))
    for sx, sy, sw, sh in sources:
        x,y,w,h = sx*GRID_SIZE, sy*GRID_SIZE, sw*GRID_SIZE, sh*GRID_SIZE
        textured = pygame.transform.scale(fill, (w,h))
        surface.blit(textured,(x,y))
        pygame.draw.rect(surface, color, (x,y,w,h), border)


def debug_draw_grid(surface, padding_frac=0.1):
    """Outline each occupied cell for debugging."""
    pad  = GRID_SIZE * padding_frac
    cell = GRID_SIZE - 2*pad
    g = get_grid()
    for x in range(cols):
        for y in range(rows):
            if g[x][y]==1:
                rect = pygame.Rect(x*GRID_SIZE+pad,
                                   y*GRID_SIZE+pad,
                                   cell, cell)
                pygame.draw.rect(surface, (200,200,200), rect, 1)


def run_environment(custom_draw_fn=None, show_sources=False):
    """
    Main loop:
     1) cap at 60fps
     2) draw background
     3) optionally overlay source‐zones
     4) call your simulation draw(dt)
    """
    assert img_full is not None, "You must call load_floorplan(...) first!"
    while True:
        dt_ms = clock.tick(90)
        dt    = dt_ms / 1000.0

        window.fill(BG_COLOR)
        window.blit(_pg_floorplan, (0,0))

        if show_sources:
            debug_draw_sources(window)

        if custom_draw_fn:
            custom_draw_fn(window, dt)

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.flip()
