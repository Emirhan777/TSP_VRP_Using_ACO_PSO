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
GIF_PATH = "pso_animation.gif"

NUM_PARTICLES = 80
ITERATIONS = 20
FRAMES_PER_EDGE = 6

W_START = 0.9
W_END = 0.4
C1 = 0.7
C2 = 0.9
MUTATION_RATE = 0.15
LOCAL_SEARCH = True
SEED = 7

START_NODE = "A"

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
NODES = [n for n in G.nodes() if n != START_NODE]

# =========================
# PSO HELPERS
# =========================

def edge_weight(u, v):
    if G.has_edge(u, v):
        return G[u][v]["weight"]
    return math.inf


def tour_cost(order):
    path = [START_NODE] + order + [START_NODE]
    cost = 0.0
    for u, v in zip(path, path[1:]):
        w = edge_weight(u, v)
        if not math.isfinite(w):
            return math.inf
        cost += w
    return cost


def two_opt(order):
    best = list(order)
    best_cost = tour_cost(best)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                cand_cost = tour_cost(candidate)
                if cand_cost < best_cost:
                    best = candidate
                    best_cost = cand_cost
                    improved = True
        if improved:
            continue
    return best, best_cost


def swaps_to_transform(src, target):
    src_list = list(src)
    swap_list = []
    index_map = {v: i for i, v in enumerate(src_list)}
    for i, v in enumerate(target):
        if src_list[i] != v:
            j = index_map[v]
            swap_list.append((i, j))
            src_list[i], src_list[j] = src_list[j], src_list[i]
            index_map[src_list[j]] = j
            index_map[src_list[i]] = i
    return swap_list


def apply_swaps(order, swaps):
    order = list(order)
    for i, j in swaps:
        order[i], order[j] = order[j], order[i]
    return order


class Particle:
    def __init__(self, init_order):
        self.position = list(init_order)
        self.velocity = []
        self.cost = tour_cost(self.position)
        self.best_position = list(self.position)
        self.best_cost = self.cost


def run_pso():
    particles = []
    for _ in range(NUM_PARTICLES):
        order = list(NODES)
        RNG.shuffle(order)
        particles.append(Particle(order))

    global_best = min(particles, key=lambda p: p.cost)
    gbest_pos = list(global_best.position)
    gbest_cost = global_best.cost

    particle_paths_by_iter = []
    gbest_path_by_iter = []
    gbest_cost_by_iter = []

    for it in range(ITERATIONS):
        w = W_START - (W_START - W_END) * (it / max(ITERATIONS - 1, 1))
        for p in particles:
            v1 = swaps_to_transform(p.position, p.best_position)
            v2 = swaps_to_transform(p.position, gbest_pos)

            keep = int(w * len(p.velocity))
            new_velocity = p.velocity[:keep]

            for sw in v1:
                if RNG.random() < C1:
                    new_velocity.append(sw)
            for sw in v2:
                if RNG.random() < C2:
                    new_velocity.append(sw)

            p.position = apply_swaps(p.position, new_velocity)
            p.velocity = new_velocity

            if RNG.random() < MUTATION_RATE:
                i, j = RNG.sample(range(len(p.position)), 2)
                p.position[i], p.position[j] = p.position[j], p.position[i]

            if LOCAL_SEARCH:
                p.position, p.cost = two_opt(p.position)
            else:
                p.cost = tour_cost(p.position)

            if p.cost < p.best_cost:
                p.best_cost = p.cost
                p.best_position = list(p.position)

        best_particle = min(particles, key=lambda p: p.cost)
        if best_particle.cost < gbest_cost:
            gbest_cost = best_particle.cost
            gbest_pos = list(best_particle.position)

        particle_paths = []
        for p in particles:
            particle_paths.append([START_NODE] + p.position + [START_NODE])
        particle_paths_by_iter.append(particle_paths)
        gbest_path_by_iter.append([START_NODE] + gbest_pos + [START_NODE])
        gbest_cost_by_iter.append(gbest_cost)

    return particle_paths_by_iter, gbest_path_by_iter, gbest_cost_by_iter


particle_paths_by_iter, gbest_path_by_iter, gbest_cost_by_iter = run_pso()

# =========================
# FRAME DATA
# =========================

particle_frames = []
iter_frames = []

for it, particle_paths in enumerate(particle_paths_by_iter):
    max_steps = max(len(path) - 1 for path in particle_paths)
    for step in range(max_steps):
        for sub in range(FRAMES_PER_EDGE):
            t = (sub + 1) / FRAMES_PER_EDGE
            positions = []
            for path in particle_paths:
                u = path[step]
                v = path[step + 1]
                u_pos = np.array(pos[u])
                v_pos = np.array(pos[v])
                positions.append(u_pos * (1 - t) + v_pos * t)
            particle_frames.append(positions)
            iter_frames.append(it + 1)

# =========================
# ANIMATION
# =========================

fig, ax = plt.subplots(figsize=(9, 7))
fig.subplots_adjust(right=0.72)
ax.set_title("PSO Simulation: Particles + Best Tour")
ax.axis("off")

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=520, node_color="lightblue")
nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)

base_lc = LineCollection(
    [(pos[u], pos[v]) for u, v in G.edges()],
    colors="#bbbbbb",
    linewidths=1.0,
    alpha=0.4
)
ax.add_collection(base_lc)

best_lc = LineCollection([], colors="#b30000", linewidths=4.5, alpha=0.6)
ax.add_collection(best_lc)
best_arrows = []

particle_scatter = ax.scatter(
    np.zeros(NUM_PARTICLES),
    np.zeros(NUM_PARTICLES),
    s=30,
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


def update(frame_idx):
    ants = np.array(particle_frames[frame_idx])
    particle_scatter.set_offsets(ants)
    iter_idx = iter_frames[frame_idx] - 1
    iter_text.set_text(f"Iteration {iter_frames[frame_idx]}")

    best_path = gbest_path_by_iter[iter_idx]
    best_cost = gbest_cost_by_iter[iter_idx]
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
                alpha=0.6
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
            best_text.set_text(
                f"Best cost: {best_cost:.2f}\n"
                f"Best tour: {'->'.join(best_path)}"
            )
        else:
            best_text.set_text("Best cost: inf\nBest tour: (infeasible)")
    else:
        best_lc.set_segments([])
        for arrow in best_arrows:
            arrow.set_visible(False)
        best_text.set_text("Best cost: N/A\nBest tour: N/A")

    return (particle_scatter, best_lc, iter_text, best_text, *best_arrows)


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(particle_frames),
    interval=120,
    blit=False
)

if SAVE_GIF:
    ani.save(GIF_PATH, writer="pillow", fps=8)
else:
    plt.show()
