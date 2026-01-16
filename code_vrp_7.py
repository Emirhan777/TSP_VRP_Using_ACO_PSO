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
GIF_PATH = "vrp_aco_animation.gif"

ANTS = 20
ITERATIONS = 15
FRAMES_PER_EDGE = 6

ALPHA = 1.0
BETA = 1.0
RHO = 0.3
Q = 1.0
EPSILON = 0.3
TAU0 = 1.0
TAU0_RANGE = (0.2, 2.0)
RANDOM_INIT = True
SEED = 1

START_NODES = ["A", "H"]

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

# =========================
# ACO HELPERS (VRP)
# =========================

def heuristic(u, v):
    return 1.0 / G[u][v]["weight"]


def choose_next(current, candidates, tau):
    if RNG.random() < EPSILON:
        return RNG.choice(candidates)
    weights = []
    total = 0.0
    for v in candidates:
        value = (tau[(current, v)] ** ALPHA) * (heuristic(current, v) ** BETA)
        weights.append(value)
        total += value
    r = RNG.random() * total
    acc = 0.0
    for v, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return v
    return candidates[-1]


def build_routes(starts, tau):
    routes = [[s] for s in starts]
    current = list(starts)
    visited = set(starts)
    k = len(starts)

    while len(visited) < len(NODES):
        progressed = False
        for i in range(k):
            if len(visited) >= len(NODES):
                break
            neighbors = [n for n in G.neighbors(current[i]) if n not in visited]
            if not neighbors:
                continue
            nxt = choose_next(current[i], neighbors, tau)
            routes[i].append(nxt)
            visited.add(nxt)
            current[i] = nxt
            progressed = True
        if not progressed:
            break

    if len(visited) != len(NODES):
        return routes, False

    for i in range(k):
        if not G.has_edge(current[i], starts[i]):
            return routes, False
        routes[i].append(starts[i])

    return routes, True


def routes_cost(routes):
    total = 0.0
    for route in routes:
        total += sum(G[u][v]["weight"] for u, v in zip(route, route[1:]))
    return total


# =========================
# SIMULATION + FRAME DATA
# =========================

undirected_edges = []
seen = set()
for u, v in G.edges():
    key = tuple(sorted((u, v)))
    if key not in seen:
        seen.add(key)
        undirected_edges.append(key)

segments = [(pos[u], pos[v]) for u, v in undirected_edges]

tau_total = {}
for u, v in G.edges():
    if RANDOM_INIT:
        init_val = RNG.uniform(TAU0_RANGE[0], TAU0_RANGE[1])
    else:
        init_val = TAU0
    tau_total[(u, v)] = init_val
    tau_total[(v, u)] = init_val

pheromone_frames = []
ant_frames = []
iter_frames = []
best_routes_by_iter = []
best_cost_by_iter = []

best_routes = None
best_cost = math.inf

for it in range(ITERATIONS):
    ant_solutions = []
    for _ in range(ANTS):
        routes, valid = build_routes(START_NODES, tau_total)
        ant_solutions.append((routes, valid))
        if valid:
            cost = routes_cost(routes)
            if cost < best_cost:
                best_cost = cost
                best_routes = [list(r) for r in routes]

    best_routes_by_iter.append(best_routes)
    best_cost_by_iter.append(best_cost)

    max_steps = 0
    for routes, _ in ant_solutions:
        for route in routes:
            max_steps = max(max_steps, len(route) - 1)

    for step in range(max_steps):
        for sub in range(FRAMES_PER_EDGE):
            t = (sub + 1) / FRAMES_PER_EDGE
            positions = []
            for routes, _ in ant_solutions:
                for route in routes:
                    if step < len(route) - 1:
                        u = route[step]
                        v = route[step + 1]
                        u_pos = np.array(pos[u])
                        v_pos = np.array(pos[v])
                        positions.append(u_pos * (1 - t) + v_pos * t)
                    else:
                        positions.append(np.array(pos[route[-1]]))

            pheromone_snapshot = []
            for u, v in undirected_edges:
                pheromone_snapshot.append(0.5 * (tau_total[(u, v)] + tau_total[(v, u)]))
            pheromone_frames.append(pheromone_snapshot)
            ant_frames.append(positions)
            iter_frames.append(it + 1)

    # Evaporate
    for e in tau_total:
        tau_total[e] *= (1 - RHO)

    # Deposit pheromone
    for routes, valid in ant_solutions:
        if not valid:
            continue
        cost = routes_cost(routes)
        deposit = Q / cost
        for route in routes:
            for u, v in zip(route, route[1:]):
                tau_total[(u, v)] += deposit
                tau_total[(v, u)] += deposit

    # Show updated pheromone with ants back at depots
    pheromone_snapshot = []
    for u, v in undirected_edges:
        pheromone_snapshot.append(0.5 * (tau_total[(u, v)] + tau_total[(v, u)]))
    pheromone_frames.append(pheromone_snapshot)
    reset_positions = []
    for _ in range(ANTS):
        for s in START_NODES:
            reset_positions.append(np.array(pos[s]))
    ant_frames.append(reset_positions)
    iter_frames.append(it + 1)


# =========================
# ANIMATION
# =========================

total_array = np.array(pheromone_frames)
total_min = float(np.min(total_array))
total_max = float(np.max(total_array))
if abs(total_max - total_min) < 1e-6:
    total_max = total_min + 1.0

fig, ax = plt.subplots(figsize=(9, 7))
fig.subplots_adjust(right=0.72)
ax.set_title("VRP ACO Simulation: Ants + Pheromone")
ax.axis("off")

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=520, node_color="lightblue")
nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)

base_lc = LineCollection(segments, colors="#888888", linewidths=1.0, alpha=0.25)
ax.add_collection(base_lc)

lc = LineCollection(segments, cmap=plt.cm.Reds, linewidths=2.0, alpha=0.7)
ax.add_collection(lc)

route_colors = ["#b30000", "#1f7a1f"]
best_lcs = []
best_arrows = [[], []]
for color in route_colors:
    best_lc = LineCollection([], colors=color, linewidths=4.5, alpha=0.6)
    ax.add_collection(best_lc)
    best_lcs.append(best_lc)

ant_scatter = ax.scatter(
    np.zeros(ANTS * len(START_NODES)),
    np.zeros(ANTS * len(START_NODES)),
    s=36,
    c="black",
    zorder=5
)

iter_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"}
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
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
)

pher_text = fig.text(
    0.74,
    0.92,
    "",
    ha="left",
    va="top",
    fontsize=8,
    family="monospace",
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4}
)


def update(frame_idx):
    total_pher = total_array[frame_idx]
    total_norm = (total_pher - total_min) / (total_max - total_min)
    total_norm = 0.15 + 0.85 * total_norm
    lc.set_array(total_norm)
    lc.set_linewidths(2.0 + 8.0 * total_norm)

    ants = np.array(ant_frames[frame_idx])
    ant_scatter.set_offsets(ants)
    iter_idx = iter_frames[frame_idx] - 1
    iter_text.set_text(f"Iteration {iter_frames[frame_idx]}")

    best_routes = best_routes_by_iter[iter_idx]
    best_cost = best_cost_by_iter[iter_idx]
    if best_routes:
        route_lines = []
        for r_idx, route in enumerate(best_routes):
            segments_route = [(pos[u], pos[v]) for u, v in zip(route, route[1:])]
            best_lcs[r_idx].set_segments(segments_route)

            while len(best_arrows[r_idx]) < len(segments_route):
                arrow = patches.FancyArrowPatch(
                    (0, 0),
                    (0, 0),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=route_colors[r_idx],
                    linewidth=1.2,
                    alpha=0.6
                )
                ax.add_patch(arrow)
                best_arrows[r_idx].append(arrow)

            for i, arrow in enumerate(best_arrows[r_idx]):
                if i < len(segments_route):
                    start, end = segments_route[i]
                    arrow.set_positions(start, end)
                    arrow.set_visible(True)
                else:
                    arrow.set_visible(False)

            route_lines.append(f"Route {START_NODES[r_idx]}: {'->'.join(route)}")

        best_text.set_text(
            f"Best cost: {best_cost:.2f}\n" + "\n".join(route_lines)
        )
    else:
        for r_idx, best_lc in enumerate(best_lcs):
            best_lc.set_segments([])
            for arrow in best_arrows[r_idx]:
                arrow.set_visible(False)
        best_text.set_text("Best cost: N/A\nRoutes: N/A")

    lines = ["Pheromone total (avg):"]
    for (u, v), value in zip(undirected_edges, total_pher):
        lines.append(f"{u}-{v}: {value:.2f}")
    pher_text.set_text("\n".join(lines))

    artists = [lc, ant_scatter, iter_text, best_text, pher_text]
    artists.extend(best_lcs)
    for arrows in best_arrows:
        artists.extend(arrows)
    return tuple(artists)


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(pheromone_frames),
    interval=120,
    blit=False
)

if SAVE_GIF:
    ani.save(GIF_PATH, writer="pillow", fps=8)
else:
    plt.show()
