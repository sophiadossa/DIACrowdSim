# experiments.py
import time, csv, math
from tqdm import tqdm
import pandas as pd

import crowd_environment as env
from rl_agent import RLAgent, BETA
from crowd_environment import WINDOW_WIDTH, WINDOW_HEIGHT, GRID_SIZE
from manager import PedestrianManager
from pedestrian_type import PedestrianType

SHARED_RL = RLAgent()


env.load_floorplan("floorplan1.png")  # dummy load to init pygame

def run_trial(floorplan, n_calm=10, n_panic=2, max_steps=500):
    # reset map & manager
    env.load_floorplan(floorplan)
    mgr = PedestrianManager(env.sources, env.targets)

    # shared RL agent
    rl = SHARED_RL

    # spawn calm agents
    for _ in range(n_calm):
        mgr.spawn_agent(PedestrianType.RL, initial_state="calm")

    # panic‐seed timers
    seed_times    = [1.0, 1.5]
    seeds_spawned = [False, False]
    prev_agents = []
    t_full = None
    step   = 0
    max_congestion = 0

    def draw_fn(window, dt):
        nonlocal prev_agents, step, t_full, max_congestion

        now = time.time() - start

        if now >= 120.0:
            # penalize all panicked RL agents
            for a in mgr.agents:
                if isinstance(a, RLAgent) and a.is_panicked():
                    # one‐time −1 for timeout
                    a.q_table[a.prev_state][a.last_action] -= 1.0
            raise StopIteration

        # A) seed panics
        for i, t in enumerate(seed_times):
            if not seeds_spawned[i] and now >= t:
                y_coord = GRID_SIZE if i == 0 else WINDOW_HEIGHT - GRID_SIZE
                mgr.spawn_agent(
                    PedestrianType.RL,
                    custom_spawn=[WINDOW_WIDTH*3/4, y_coord],
                    initial_state="panic"
                )
                seeds_spawned[i] = True

        # B) one step of the world
        mgr.update(dt)
        mgr.draw(window)

        # C) congestion tracking
        if mgr.agents:
            R = max(a.get_radius() for a in mgr.agents) * 2
            for a in mgr.agents:
                neigh = sum(
                    1 for b in mgr.agents
                    if b is not a and
                       math.hypot(a.pos[0]-b.pos[0], a.pos[1]-b.pos[1]) < R
                )
                max_congestion = max(max_congestion, neigh)

        # D) pick next actions for panicked RL agents
        for a in mgr.agents:
            if isinstance(a, RLAgent) and a.is_panicked():
                st = a.get_state(mgr.agents)
                act = rl.select_action(st)
                a._prev_state  = st
                a._prev_dist   = a.distance_to_exit()
                a._last_action = act
                a.apply_action(act)

        # E) record “full panic” time
        if mgr.agents and all(a.is_panicked() for a in mgr.agents) and t_full is None:
            t_full = now

        # if all(seeds_spawned):
        #     non_panicked = sum(1 for a in mgr.agents if not a.is_panicked())
        #     if non_panicked == 0 and time_to_full_panic is None:
        #         time_to_full_panic = now

        # # final evacuation check — only here do we end the trial
        # if not mgr.agents:
        #     raise StopIteration# H) full‐panic timing (but don’t abort yet)
        

    # run it
    start = time.time()
    try:
        env.run_environment(draw_fn, show_sources=False)
    except StopIteration:
        pass
    now = time.time() - start

    # metrics
    t_full_panic = t_full or float("nan")
    t_evac       = now    if not mgr.agents else float("nan")
    pct_panicked = 100.0 * sum(a.is_panicked() for a in mgr.agents) / max(1, len(mgr.agents))
    passed       = (not mgr.agents)
    return t_full_panic, t_evac, max_congestion, pct_panicked, passed


def experiment(fps, trials=50):
    total = len(fps) * trials
    with open("results.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "floorplan","trial",
            "time_to_full_panic","time_to_evac",
            "max_congestion","pct_panicked","pass_fail"
        ])
        bar_fmt = "{l_bar}{bar}| Iter {n}/{total} | Elapsed: {elapsed} | Left: {remaining}"
        with tqdm(total=total, desc="Trials", bar_format=bar_fmt) as pbar:
            for fp in fps:
                for i in range(trials):
                    row = run_trial(fp, n_calm=100, n_panic=2, max_steps=500)
                    w.writerow([
                        fp, i,
                        f"{row[0]:.2f}", f"{row[1]:.2f}",
                        row[2], f"{row[3]:.1f}",
                        "PASS" if row[4] else "FAIL"
                    ])
                    pbar.update(1)

    # dump the shared Q-table
    df = pd.DataFrame(
        list(SHARED_RL.q_table.values()),
        index=list(SHARED_RL.q_table.keys()),
        columns=[f"action_{i}" for i in range(SHARED_RL.n_actions)]
    )
    df.to_csv("q_table.csv")


if __name__ == "__main__":
    experiment(["floorplan1.png"], trials=100)
