# experiments.py
import time, csv, math
from tqdm import tqdm

import crowd_environment as env
from crowd_environment import WINDOW_HEIGHT, WINDOW_WIDTH, GRID_SIZE
from manager import PedestrianManager
from pedestrian_type import PedestrianType
from rl_agent import RLAgent

# ──────────────────────────────────────────────────────────────────────────────
# tmp = RLAgent([0,0],[800,600])
# print("state vector length:", len(tmp.get_state_vector()))
# print("Action space size:", len(tmp.action_space))
# Instantiate a single RL agent to be shared across all trials
# load a dummy map so env.window, env.clock, etc. get initialized
env.load_floorplan("floorplan1.png")


def run_trial(floorplan, n_agents, panic_time, show_sources=False, 
              max_stuck_steps=120):
    """
    Runs one trial on `floorplan` with n_agents calm→RL.
    Returns (time_to_full_panic, time_to_evacuate, max_congestion, pass_fail)
    """
    # 1) load this floorplan fresh
    env.load_floorplan(floorplan)

    rl_agent = RLAgent

    # 2) new manager tied to THIS map's sources/targets
    mgr = PedestrianManager(env.sources, env.targets)

    # 3) spawn n_agents calm-RL, passing our shared rl_agent
    for _ in range(n_agents):
        mgr.spawn_agent(PedestrianType.RL, initial_state="calm")

    # stuck logic
    prev_pos = {a: tuple(a.pos) for a in mgr.agents}
    stuck_time = {a: 0.0        for a in mgr.agents}
    total_agents = len(mgr.agents)
    stuck_limit = 20.0
    threshold = 0.6 # 60% stuck == abort

    # new: confused-progress trackers
    prev_dist_conf   = {}
    stuck_time_conf  = {}
    confused_thresh  = 0.6   # abort if 60% of confused are stalled

    sim_start = time.time()
    time_to_full_panic = None
    max_congestion    = 0
    panic_spawned     = False
    panic_seeds_done = [False, False]
    spawn_delay = 2.0

    # variables for "stuck" detection
    prev_avg_distance = None
    no_progress_steps = 0

    def draw_fn(screen, dt):
        nonlocal time_to_full_panic, max_congestion, panic_spawned
        nonlocal prev_avg_distance, no_progress_steps
        nonlocal prev_pos, stuck_time

        now = time.time() - sim_start

        # A) at panic_time, inject one panicked RL
        if not panic_seeds_done[0] and now >= spawn_delay:
            mgr.spawn_agent(
                PedestrianType.RL,
                custom_spawn=[WINDOW_WIDTH*3/4, GRID_SIZE],
                initial_state="panic"
            )
            panic_seeds_done[0] = True
        # B) bottom-middle
        if not panic_seeds_done[1] and now >= spawn_delay:
            mgr.spawn_agent(
                PedestrianType.RL,
                custom_spawn=[WINDOW_WIDTH*3/4, WINDOW_HEIGHT - GRID_SIZE],
                initial_state="panic"
            )
            panic_seeds_done[1] = True

        # B) --- RL step: for each RL agent, select and apply action ---
        for ped in mgr.agents:
            if hasattr(ped, 'policy'):
                # 1) observe state
                state = ped.get_state_vector()
                # 2) choose action
                action = rl_agent.select_action(state)
                # 3) apply it
                ped.apply_rl_action(action)
                # store for reward computation
                ped._prev_state = state
                ped._prev_dist  = ped.distance_to_exit()

        # C) step + draw
        mgr.update(dt)
        mgr.draw(screen)

        for a in mgr.agents:
            if a not in prev_pos:
                prev_pos[a] = (a.pos[0], a.pos[1])
                stuck_time[a] = 0.0

        # for confused‐progress:
            if a.is_confused() and a not in prev_dist_conf:
                prev_dist_conf[a]  = a.distance_to_exit()
                stuck_time_conf[a] = 0.0

        #=== update stuck timers
        stuck_count = 0
        total_panicked = sum(1 for a in mgr.agents if a.is_panicked())

        for a in mgr.agents:
            if not a.is_panicked(): 
               continue
            # how far moved since last frame?
            oldx, oldy = prev_pos[a]
            dist = math.hypot(a.pos[0] - oldx, a.pos[1] - oldy)
            if dist < a.get_radius():
                stuck_time[a] += dt
            else:
               stuck_time[a] = 0.0
            prev_pos[a] = (a.pos[0], a.pos[1])

            if stuck_time[a] > stuck_limit:
                stuck_count += 1

        # if ≥40% of panicked folks are stuck, abort early
        if total_panicked > 0 and stuck_count >= threshold * total_panicked:
            # penalize each panicked RL agent
            for a in mgr.agents:
                if isinstance(a, RLAgent) and a.is_panicked():
                    s = a.get_state(mgr.agents)
                    # give a one-off -1 to its chosen action
                    # (you can adapt this to your reward API)
                    a.q_table[s][0] -= 1.0
            raise StopIteration
    
                # ─── confused-agents: check forward progress ───────────────
        stuck_conf = 0
        total_conf = sum(1 for a in mgr.agents if a.is_confused())

        for a in mgr.agents:
            if not a.is_confused():
                continue

            # initialize trackers for newly confused
            if a not in prev_dist_conf:
                prev_dist_conf[a] = a.distance_to_exit()
                stuck_time_conf[a] = 0.0

            # compare current vs last distance
            oldd = prev_dist_conf[a]
            newd = a.distance_to_exit()
            if newd >= oldd - 1e-3:
                stuck_time_conf[a] += dt
            else:
                stuck_time_conf[a] = 0.0

            # store for next frame
            prev_dist_conf[a] = newd

            if stuck_time_conf[a] > stuck_limit:
                stuck_conf += 1

        # abort if too many confused are stalled
        if total_conf > 0 and stuck_conf >= confused_thresh * total_conf:
            raise StopIteration

        # D) --- RL reward & learning ---
        for ped in mgr.agents:
            if hasattr(ped, '_prev_state'):
                next_state = ped.get_state_vector()
                curr_dist  = ped.distance_to_exit()
                # reward = reduction in distance (positive), or -1 on collision/stuck
                reward = (ped._prev_dist - curr_dist)
                rl_agent.store_transition(
                    ped._prev_state, ped.last_action, reward, next_state
                )
                rl_agent.learn()

        #E) count states
        counts = {
            'calm':  sum(1 for a in mgr.agents if hasattr(a,"is_calm")    and a.is_calm()),
            'conf':  sum(1 for a in mgr.agents if hasattr(a,"is_confused")and a.is_confused()),
            'panic': sum(1 for a in mgr.agents if hasattr(a,"is_panicked")and a.is_panicked()),
        }
        
        # F) full panic when no calm/conf left
        if time_to_full_panic is None and counts['panic'] == n_agents:
            time_to_full_panic = now

        # G) congestion = max neighbors within 2×radius
        if mgr.agents:
            R = max(a.get_radius() for a in mgr.agents)*2
            for a in mgr.agents:
                neigh = sum(
                    1 for b in mgr.agents
                    if b is not a and
                       math.hypot(a.pos[0]-b.pos[0], a.pos[1]-b.pos[1]) < R
                )
                max_congestion = max(max_congestion, neigh)

        # H) check for evacuation
        if not mgr.agents:
            raise StopIteration

        # I) "stuck" detection: average distance to exit
        distances = [a.distance_to_exit() for a in mgr.agents]
        avg_dist = sum(distances) / len(distances)
        if prev_avg_distance is not None and avg_dist >= prev_avg_distance - 1e-3:
            no_progress_steps += 1
        else:
            no_progress_steps = 0
        prev_avg_distance = avg_dist

        if no_progress_steps > max_stuck_steps:
            # abort this trial as stuck
            raise StopIteration

    # 4) run until StopIteration
    try:
        env.run_environment(draw_fn, show_sources=show_sources)
    except StopIteration:
        pass

    t_evac = time.time() - sim_start
    # ensure we never leave it None
    if time_to_full_panic is None:
        time_to_full_panic = float("nan")

    pass_fail = (len(mgr.agents) == 0)

    if not pass_fail:
        t_evac = float("nan")

    return time_to_full_panic, t_evac, max_congestion, pass_fail


def experiment(floorplans, trials_per=10, n_agents=50, panic_time=5.0):

    total = len(floorplans) * trials_per

    with open("results.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "floorplan", "trial",
            "time_to_full_panic", "time_to_evacuate",
            "max_congestion", "pass_fail"
        ])


        with tqdm(
            total=total,
            desc="Trials",
            bar_format="{l_bar}{bar} | Elapsed: {elapsed} | Left: {remaining} | {rate_fmt}"
        ) as pbar:
            for fp in floorplans:
                for trial in range(trials_per):
                    # retry up to 3 times if stuck (pass_fail False)
                    for attempt in range(3):
                        t_p, t_e, cong, ok = run_trial(
                            fp, n_agents, panic_time
                        )
                        if ok:
                            break
                    w.writerow([
                        fp, trial,
                        f"{t_p:.2f}", f"{t_e:.2f}",
                        cong, "PASS" if ok else "FAIL"
                    ])

                    pbar.update(1)


if __name__ == "__main__":
    # only run on floorplan1.png
   experiment(
        ["floorplan1.png"],
        trials_per=100,
        n_agents=100,
        panic_time=5.0
    )
    #  = [
    #   "floorplan1.png",
    #   "blockedexit2.png",
    #   "emergencyexits3.png",
    #   "blockades4.png"
    # ]
    # experiment(fps,
    #            trials_per=50,
    #            n_agents=100,
    #            panic_time=5.0)