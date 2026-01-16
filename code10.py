import math
import random

import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

# =========================
# CONFIG
# =========================

USE_AGG = False
SAVE_GIF = False
SAVE_MP4 = False
GIF_PATH = "aco_dynamic.gif"
MP4_PATH = "aco_dynamic.mp4"

ANTS = 2
ORIGIN_NODE = "A"
DEST_NODE = "J"

SPEED = 1.0
DT = 0.2
INTERVAL_MS = 60

ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
RHO = 0.08
Q = 1.0
EPSILON = 0.2
ENDPOINT_BOOST = 2.5
TAU0 = 1.0
TAU0_RANGE = (0.2, 2.0)
RANDOM_INIT = True
SEED = 5

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

undirected_edges = []
seen = set()
for u, v in G.edges():
    key = tuple(sorted((u, v)))
    if key not in seen:
        seen.add(key)
        undirected_edges.append(key)

segments = [(pos[u], pos[v]) for u, v in undirected_edges]

# =========================
# ACO HELPERS
# =========================

def heuristic(u, v):
    return 1.0 / G[u][v]["weight"]


def compute_distance_to_target(target):
    return nx.single_source_dijkstra_path_length(G.reverse(), target, weight="weight")


def choose_next(current, last_node, tau, rng, target_dist):
    neighbors = list(G.neighbors(current))
    if not neighbors:
        return None

    candidates = list(neighbors)
    if last_node in candidates and len(candidates) > 1 and rng.random() < 0.7:
        candidates.remove(last_node)

    if rng.random() < EPSILON:
        return rng.choice(candidates)

    weights = []
    total = 0.0
    for v in candidates:
        goal = 1.0
        if target_dist is not None:
            dist = target_dist.get(v, math.inf)
            if math.isinf(dist):
                goal = 1e-6
            else:
                goal = (1.0 / (dist + 1e-6)) ** GAMMA
        value = (tau[(current, v)] ** ALPHA) * (heuristic(current, v) ** BETA) * goal
        weights.append(value)
        total += value

    r = rng.random() * total
    acc = 0.0
    for v, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return v
    return candidates[-1]


class Ant:
    def __init__(self, start_node, rng, offset, color):
        self.current = start_node
        self.next = None
        self.last = None
        self.progress = 0.0
        self.travel_time = 1.0
        self.rng = rng
        self.offset = offset
        self.color = color
        self.mode = "search"
        self.path = [start_node]
        self.return_idx = None

    def reset(self, start_node):
        self.current = start_node
        self.next = None
        self.last = None
        self.progress = 0.0
        self.travel_time = 1.0
        self.mode = "search"
        self.path = [start_node]
        self.return_idx = None

    def start_edge(self, nxt, tau):
        self.next = nxt
        weight = G[self.current][nxt]["weight"]
        self.travel_time = weight / SPEED
        self.progress = 0.0
        boost = ENDPOINT_BOOST if self.current in (ORIGIN_NODE, DEST_NODE) else 1.0
        tau[(self.current, nxt)] += (Q / weight) * boost

    def step(self, dt, tau, target_dist):
        if self.next is None:
            if self.mode == "return":
                if self.return_idx is None or self.return_idx < 0:
                    self.mode = "search"
                    self.path = [self.current]
                    self.return_idx = None
                else:
                    nxt = self.path[self.return_idx]
                    self.return_idx -= 1
                    self.start_edge(nxt, tau)
            else:
                nxt = choose_next(self.current, self.last, tau, self.rng, target_dist)
                if nxt is None:
                    return
                self.start_edge(nxt, tau)

        self.progress += dt / self.travel_time
        if self.progress >= 1.0:
            self.last = self.current
            self.current = self.next
            self.next = None
            self.progress = 0.0
            if self.mode == "search":
                if not self.path or self.current != self.path[-1]:
                    self.path.append(self.current)
                if self.current == DEST_NODE and DEST_NODE != ORIGIN_NODE:
                    self.mode = "return"
                    self.return_idx = len(self.path) - 2
            elif self.current == ORIGIN_NODE:
                self.mode = "search"
                self.path = [self.current]
                self.return_idx = None

    def position(self):
        if self.next is None:
            return np.array(pos[self.current]) + self.offset
        u = self.current
        v = self.next
        p = self.progress
        return (1 - p) * np.array(pos[u]) + p * np.array(pos[v]) + self.offset


# =========================
# SIMULATION STATE
# =========================

dist_to_dest = compute_distance_to_target(DEST_NODE)

tau = {}
for u, v in G.edges():
    if RANDOM_INIT:
        init_val = RNG.uniform(TAU0_RANGE[0], TAU0_RANGE[1])
    else:
        init_val = TAU0
    tau[(u, v)] = init_val
    tau[(v, u)] = init_val

ant_offsets = np.array(
    [(RNG.uniform(-0.02, 0.02), RNG.uniform(-0.02, 0.02)) for _ in range(ANTS)]
)
ant_colors = ["#1f77b4", "#ff7f0e"]

ants = []
for i in range(ANTS):
    ants.append(Ant(ORIGIN_NODE, random.Random(SEED + i + 1), ant_offsets[i], ant_colors[i % 2]))

time_elapsed = 0.0

# =========================
# ANIMATION
# =========================

fig, ax = plt.subplots(figsize=(9, 7))
ax.set_title("Dynamic ACO Simulation: Origin to Destination")
ax.axis("off")

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=520, node_color="lightblue")
nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)

origin_scatter = ax.scatter(
    [pos[ORIGIN_NODE][0]],
    [pos[ORIGIN_NODE][1]],
    s=680,
    c="#ffcc66",
    edgecolors="#cc7a00",
    linewidths=1.2,
    zorder=4,
)
dest_scatter = ax.scatter(
    [pos[DEST_NODE][0]],
    [pos[DEST_NODE][1]],
    s=680,
    c="#9be28f",
    edgecolors="#2f8f2f",
    linewidths=1.2,
    zorder=4,
)

base_lc = LineCollection(segments, colors="#888888", linewidths=1.0, alpha=0.25)
ax.add_collection(base_lc)

pher_lc = LineCollection(segments, cmap=plt.cm.Reds, linewidths=2.0, alpha=0.7)
ax.add_collection(pher_lc)

ant_scatter = ax.scatter(
    np.zeros(ANTS),
    np.zeros(ANTS),
    s=40,
    c=ant_colors,
    zorder=5,
)

time_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
)

dest_text = ax.text(
    0.02,
    0.92,
    f"Destination: {DEST_NODE} (click a node to change)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
)


def update(_):
    global time_elapsed

    for e in tau:
        tau[e] *= max(0.0, 1.0 - RHO * DT)

    for ant in ants:
        ant.step(DT, tau, dist_to_dest)

    pher = []
    for u, v in undirected_edges:
        pher.append(0.5 * (tau[(u, v)] + tau[(v, u)]))
    pher = np.array(pher)
    min_p = float(np.min(pher))
    max_p = float(np.max(pher))
    if abs(max_p - min_p) < 1e-6:
        max_p = min_p + 1.0
    norm = (pher - min_p) / (max_p - min_p)
    norm = 0.15 + 0.85 * norm
    pher_lc.set_array(norm)
    pher_lc.set_linewidths(2.0 + 6.0 * norm)

    positions = np.array([ant.position() for ant in ants])
    ant_scatter.set_offsets(positions)

    time_elapsed += DT
    time_text.set_text(f"Time: {time_elapsed:.1f}s")

    return pher_lc, ant_scatter, time_text, dest_text


ani = animation.FuncAnimation(
    fig,
    update,
    interval=INTERVAL_MS,
    blit=False,
)


def on_click(event):
    global DEST_NODE, dist_to_dest
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    click = np.array([event.xdata, event.ydata])
    closest = min(pos.keys(), key=lambda n: np.linalg.norm(click - np.array(pos[n])))
    if closest == ORIGIN_NODE:
        return
    if np.linalg.norm(click - np.array(pos[closest])) > 0.18:
        return
    DEST_NODE = closest
    dist_to_dest = compute_distance_to_target(DEST_NODE)
    dest_scatter.set_offsets([pos[DEST_NODE]])
    dest_text.set_text(f"Destination: {DEST_NODE} (click a node to change)")
    for ant in ants:
        ant.reset(ORIGIN_NODE)


fig.canvas.mpl_connect("button_press_event", on_click)

if SAVE_MP4:
    ani.save(MP4_PATH, writer="ffmpeg", fps=12)
elif SAVE_GIF:
    ani.save(GIF_PATH, writer="pillow", fps=8)
else:
    plt.show()
