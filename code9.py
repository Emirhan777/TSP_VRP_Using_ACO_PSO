import math
import random

import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib import patches

# =========================
# CONFIG
# =========================

USE_AGG = False
SAVE_GIF = False
SAVE_MP4 = False
GIF_PATH = "aco_path_animation.gif"
MP4_PATH = "aco_path_animation.mp4"

ANTS = 20
ITERATIONS = 25
FRAMES_PER_EDGE = 4
MAX_STEPS = 20

ALPHA = 1.0
BETA = 1.0
RHO = 0.3
Q = 1.0
EPSILON = 0.25
TAU0 = 1.0
TAU0_RANGE = (0.2, 2.0)
RANDOM_INIT = True
SEED = 3

ORIGIN = "A"
DESTINATION = "J"

if USE_AGG:
    matplotlib.use("Agg")

RNG = random.Random(SEED)

# =========================
# GRAPH DEFINITION
# =========================

G = nx.DiGraph()

base_edges = [
    ("A", "B", 3),
    ("B", "C", 4),
    ("C", "D", 3),
    ("D", "E", 5),
    ("E", "F", 4),
    ("F", "G", 5),
    ("G", "H", 3),
    ("H", "I", 4),
    ("I", "J", 3),
    ("J", "A", 6),
    ("A", "E", 7),
    ("C", "G", 6),
    ("E", "I", 5),
    ("B", "F", 7),
    ("A", "D", 6),
    ("B", "E", 6),
    ("C", "F", 7),
    ("D", "G", 6),
    ("E", "H", 7),
    ("F", "I", 6),
    ("G", "J", 7),
    ("B", "H", 8),
    ("C", "I", 8),
]

for u, v, w in base_edges:
    G.add_weighted_edges_from([(u, v, w), (v, u, w)])

pos = nx.spring_layout(G, seed=5)
NODES = list(G.nodes())

undirected_edges = []
seen = set()
for u, v in G.edges():
    key = tuple(sorted((u, v)))
    if key not in seen:
        seen.add(key)
        undirected_edges.append(key)

segments = [(pos[u], pos[v]) for u, v in undirected_edges]

# =========================
# ACO HELPERS (ORIGIN -> DESTINATION)
# =========================

def heuristic(u, v):
    return 1.0 / G[u][v]["weight"]


def choose_next(current, candidates, tau, epsilon, alpha, beta):
    if RNG.random() < epsilon:
        return RNG.choice(candidates)
    weights = []
    total = 0.0
    for v in candidates:
        value = (tau[(current, v)] ** alpha) * (heuristic(current, v) ** beta)
        weights.append(value)
        total += value
    r = RNG.random() * total
    acc = 0.0
    for v, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return v
    return candidates[-1]


def build_path(start, goal, tau, traits):
    path = [start]
    visited = {start}
    current = start
    steps = 0

    while current != goal and steps < MAX_STEPS:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            return path, False
        unvisited = [n for n in neighbors if n not in visited]
        candidates = unvisited if unvisited else neighbors
        nxt = choose_next(
            current,
            candidates,
            tau,
            traits["epsilon"],
            traits["alpha"],
            traits["beta"],
        )
        path.append(nxt)
        visited.add(nxt)
        current = nxt
        steps += 1

    return path, current == goal


def path_cost(path):
    return sum(G[u][v]["weight"] for u, v in zip(path, path[1:]))


def init_ant_traits():
    traits = []
    for _ in range(ANTS):
        if RNG.random() < 0.3:
            epsilon = RNG.uniform(0.35, 0.6)
            beta = RNG.uniform(0.6, 1.0)
        else:
            epsilon = RNG.uniform(0.05, 0.2)
            beta = RNG.uniform(1.0, 2.0)
        alpha = ALPHA * RNG.uniform(0.8, 1.2)
        traits.append({"epsilon": epsilon, "alpha": alpha, "beta": beta})
    return traits


def init_ant_offsets():
    return np.array(
        [(RNG.uniform(-0.02, 0.02), RNG.uniform(-0.02, 0.02)) for _ in range(ANTS)]
    )


def build_simulation(destination, ant_offsets, ant_traits):
    tau = {}
    for u, v in G.edges():
        if RANDOM_INIT:
            init_val = RNG.uniform(TAU0_RANGE[0], TAU0_RANGE[1])
        else:
            init_val = TAU0
        tau[(u, v)] = init_val
        tau[(v, u)] = init_val

    pheromone_frames = []
    ant_frames = []
    iter_frames = []
    best_path_by_iter = []
    best_cost_by_iter = []

    best_path = None
    best_cost = math.inf

    for it in range(ITERATIONS):
        ant_paths = []
        for ant_idx, traits in enumerate(ant_traits):
            path, valid = build_path(ORIGIN, destination, tau, traits)
            ant_paths.append((path, valid))
            if valid:
                cost = path_cost(path)
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)

        best_path_by_iter.append(best_path)
        best_cost_by_iter.append(best_cost)

        max_steps = max(len(p[0]) - 1 for p in ant_paths)
        for step in range(max_steps):
            for sub in range(FRAMES_PER_EDGE):
                t = (sub + 1) / FRAMES_PER_EDGE
                positions = []
                for idx, (path, _) in enumerate(ant_paths):
                    if step < len(path) - 1:
                        u = path[step]
                        v = path[step + 1]
                        u_pos = np.array(pos[u])
                        v_pos = np.array(pos[v])
                        positions.append(u_pos * (1 - t) + v_pos * t + ant_offsets[idx])
                    else:
                        positions.append(np.array(pos[path[-1]]) + ant_offsets[idx])

                pheromone_snapshot = []
                for u, v in undirected_edges:
                    pheromone_snapshot.append(0.5 * (tau[(u, v)] + tau[(v, u)]))
                pheromone_frames.append(pheromone_snapshot)
                ant_frames.append(positions)
                iter_frames.append(it + 1)

        for e in tau:
            tau[e] *= (1 - RHO)

        for path, valid in ant_paths:
            if not valid:
                continue
            cost = path_cost(path)
            deposit = Q / cost
            for u, v in zip(path, path[1:]):
                tau[(u, v)] += deposit
                tau[(v, u)] += deposit

        pheromone_snapshot = []
        for u, v in undirected_edges:
            pheromone_snapshot.append(0.5 * (tau[(u, v)] + tau[(v, u)]))
        pheromone_frames.append(pheromone_snapshot)
        ant_frames.append([np.array(pos[ORIGIN]) + ant_offsets[i] for i in range(ANTS)])
        iter_frames.append(it + 1)

    return {
        "pheromone_frames": pheromone_frames,
        "ant_frames": ant_frames,
        "iter_frames": iter_frames,
        "best_path_by_iter": best_path_by_iter,
        "best_cost_by_iter": best_cost_by_iter,
    }


def nearest_node(x, y):
    best_node = None
    best_dist = float("inf")
    for n in NODES:
        dx = pos[n][0] - x
        dy = pos[n][1] - y
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_node = n
    return best_node


# =========================
# ANIMATION SETUP
# =========================

fig, ax = plt.subplots(figsize=(9, 7))
ax.axis("off")

node_collection = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=520, node_color="lightblue")
nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)

base_lc = LineCollection(segments, colors="#888888", linewidths=1.0, alpha=0.25)
ax.add_collection(base_lc)

pher_lc = LineCollection(segments, cmap=plt.cm.Reds, linewidths=2.0, alpha=0.7)
ax.add_collection(pher_lc)

best_lc = LineCollection([], colors="#b30000", linewidths=4.0, alpha=0.6)
ax.add_collection(best_lc)

best_arrows = []
ant_scatter = ax.scatter(np.zeros(ANTS), np.zeros(ANTS), s=35, c="black", zorder=5)

iter_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
)

best_text = ax.text(
    0.02,
    0.02,
    "",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    family="monospace",
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
)

help_text = ax.text(
    0.98,
    0.02,
    "Click a node to set destination",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
)

ani = None
ant_offsets = None
pheromone_frames = []
ant_frames = []
iter_frames = []
best_path_by_iter = []
best_cost_by_iter = []
total_array = None
total_min = 0.0
total_max = 1.0


def update_node_colors():
    colors = []
    for n in G.nodes():
        if n == ORIGIN:
            colors.append("#2ca02c")
        elif n == DESTINATION:
            colors.append("#ff7f0e")
        else:
            colors.append("lightblue")
    node_collection.set_facecolors(colors)


def reset_simulation(destination):
    global DESTINATION
    global ant_offsets
    global pheromone_frames, ant_frames, iter_frames
    global best_path_by_iter, best_cost_by_iter
    global total_array, total_min, total_max
    global best_arrows, ani

    DESTINATION = destination
    ant_offsets = init_ant_offsets()
    ant_traits = init_ant_traits()
    data = build_simulation(destination, ant_offsets, ant_traits)

    pheromone_frames = data["pheromone_frames"]
    ant_frames = data["ant_frames"]
    iter_frames = data["iter_frames"]
    best_path_by_iter = data["best_path_by_iter"]
    best_cost_by_iter = data["best_cost_by_iter"]

    total_array = np.array(pheromone_frames)
    total_min = float(np.min(total_array))
    total_max = float(np.max(total_array))
    if abs(total_max - total_min) < 1e-6:
        total_max = total_min + 1.0

    update_node_colors()
    ax.set_title(f"ACO Path Simulation: {ORIGIN} to {DESTINATION}")
    best_lc.set_segments([])
    for arrow in best_arrows:
        arrow.remove()
    best_arrows = []

    if ani is not None:
        ani.event_source.stop()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(pheromone_frames),
        interval=120,
        blit=False,
    )
    fig.canvas.draw_idle()


def update(frame_idx):
    if total_array is None:
        return tuple()

    total_pher = total_array[frame_idx]
    total_norm = (total_pher - total_min) / (total_max - total_min)
    total_norm = 0.15 + 0.85 * total_norm
    pher_lc.set_array(total_norm)
    pher_lc.set_linewidths(2.0 + 6.0 * total_norm)

    ants = np.array(ant_frames[frame_idx])
    ant_scatter.set_offsets(ants)

    iter_idx = iter_frames[frame_idx] - 1
    iter_text.set_text(f"Iteration {iter_frames[frame_idx]}")

    best_path = best_path_by_iter[iter_idx]
    best_cost = best_cost_by_iter[iter_idx]
    if best_path:
        best_segments = [(pos[u], pos[v]) for u, v in zip(best_path, best_path[1:])]
        best_lc.set_segments(best_segments)
        while len(best_arrows) < len(best_segments):
            arrow = patches.FancyArrowPatch(
                (0, 0),
                (0, 0),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#b30000",
                linewidth=1.2,
                alpha=0.6,
            )
            ax.add_patch(arrow)
            best_arrows.append(arrow)
        for i, arrow in enumerate(best_arrows):
            if i < len(best_segments):
                start, end = best_segments[i]
                arrow.set_positions(start, end)
                arrow.set_visible(True)
            else:
                arrow.set_visible(False)
        if math.isfinite(best_cost):
            best_text.set_text(f"Best cost: {best_cost:.2f}")
        else:
            best_text.set_text("Best cost: inf")
    else:
        best_lc.set_segments([])
        for arrow in best_arrows:
            arrow.set_visible(False)
        best_text.set_text("Best cost: N/A")

    return (pher_lc, best_lc, ant_scatter, iter_text, best_text, *best_arrows)


def on_click(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    new_dest = nearest_node(event.xdata, event.ydata)
    if new_dest in (None, ORIGIN) or new_dest == DESTINATION:
        return
    reset_simulation(new_dest)


fig.canvas.mpl_connect("button_press_event", on_click)
reset_simulation(DESTINATION)

if SAVE_MP4:
    ani.save(MP4_PATH, writer="ffmpeg", fps=12)
elif SAVE_GIF:
    ani.save(GIF_PATH, writer="pillow", fps=8)
else:
    plt.show()
