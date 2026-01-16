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
GIF_PATH = "vrp_pso_animation.gif"

NUM_PARTICLES = 80
ITERATIONS = 40
FRAMES_PER_EDGE = 3
RESTARTS = 5
SELECT_RESTART = 2

W_START = 0.9
W_END = 0.4
C1 = 0.7
C2 = 0.9
MUTATION_RATE = 0.15
SEED = 7

STARTS = ["A", "H"]

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

NODES = [n for n in G.nodes() if n not in STARTS]

# =========================
# PSO HELPERS (VRP)
# =========================

def edge_weight(u, v):
    if G.has_edge(u, v):
        return G[u][v]["weight"]
    return math.inf


def routes_from_position(order, split_index):
    route_a = [STARTS[0]] + order[:split_index] + [STARTS[0]]
    route_b = [STARTS[1]] + order[split_index:] + [STARTS[1]]
    return [route_a, route_b]


def route_cost(route):
    cost = 0.0
    for u, v in zip(route, route[1:]):
        w = edge_weight(u, v)
        if not math.isfinite(w):
            return math.inf
        cost += w
    return cost


def vrp_cost(order, split_index):
    if split_index <= 0 or split_index >= len(order):
        return math.inf
    routes = routes_from_position(order, split_index)
    return route_cost(routes[0]) + route_cost(routes[1])


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
    def __init__(self, init_order, split_index):
        self.position = list(init_order)
        self.split = split_index
        self.velocity = []
        self.cost = vrp_cost(self.position, self.split)
        self.best_position = list(self.position)
        self.best_split = self.split
        self.best_cost = self.cost


def run_pso_vrp():
    selected_particles = None
    selected_gbest_routes = None
    selected_gbest_costs = None

    last_particles = None
    last_gbest_routes = None
    last_gbest_costs = None

    for restart_idx in range(1, RESTARTS + 1):
        particles = []
        for _ in range(NUM_PARTICLES):
            order = list(NODES)
            RNG.shuffle(order)
            split_index = RNG.randint(1, len(order) - 1)
            particles.append(Particle(order, split_index))

        global_best = min(particles, key=lambda p: p.cost)
        gbest_pos = list(global_best.position)
        gbest_split = global_best.split
        gbest_cost = global_best.cost

        particle_routes_by_iter = []
        gbest_routes_by_iter = []
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

                if RNG.random() < MUTATION_RATE:
                    p.split = RNG.randint(1, len(p.position) - 1)

                p.cost = vrp_cost(p.position, p.split)

                if p.cost < p.best_cost:
                    p.best_cost = p.cost
                    p.best_position = list(p.position)
                    p.best_split = p.split

            best_particle = min(particles, key=lambda p: p.cost)
            if best_particle.cost < gbest_cost:
                gbest_cost = best_particle.cost
                gbest_pos = list(best_particle.position)
                gbest_split = best_particle.split

            particle_routes = []
            for p in particles:
                particle_routes.append(routes_from_position(p.position, p.split))
            particle_routes_by_iter.append(particle_routes)
            gbest_routes_by_iter.append(routes_from_position(gbest_pos, gbest_split))
            gbest_cost_by_iter.append(gbest_cost)

        last_particles = particle_routes_by_iter
        last_gbest_routes = gbest_routes_by_iter
        last_gbest_costs = gbest_cost_by_iter

        if restart_idx == SELECT_RESTART:
            selected_particles = particle_routes_by_iter
            selected_gbest_routes = gbest_routes_by_iter
            selected_gbest_costs = gbest_cost_by_iter
            break

    if selected_particles is None:
        selected_particles = last_particles
        selected_gbest_routes = last_gbest_routes
        selected_gbest_costs = last_gbest_costs

    return selected_particles, selected_gbest_routes, selected_gbest_costs


particle_routes_by_iter, gbest_routes_by_iter, gbest_cost_by_iter = run_pso_vrp()

# =========================
# FRAME DATA
# =========================

particle_frames = []
iter_frames = []

for it, particle_routes in enumerate(particle_routes_by_iter):
    max_steps = 0
    for routes in particle_routes:
        for route in routes:
            max_steps = max(max_steps, len(route) - 1)
    for step in range(max_steps):
        for sub in range(FRAMES_PER_EDGE):
            t = (sub + 1) / FRAMES_PER_EDGE
            positions = []
            for routes in particle_routes:
                for route in routes:
                    if step < len(route) - 1:
                        u = route[step]
                        v = route[step + 1]
                        u_pos = np.array(pos[u])
                        v_pos = np.array(pos[v])
                        positions.append(u_pos * (1 - t) + v_pos * t)
                    else:
                        positions.append(np.array(pos[route[-1]]))
            particle_frames.append(positions)
            iter_frames.append(it + 1)

# =========================
# ANIMATION
# =========================

fig, ax = plt.subplots(figsize=(9, 7))
fig.subplots_adjust(right=0.72)
ax.set_title(f"VRP PSO Simulation: Particles + Best Routes (Restart {SELECT_RESTART})")
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

route_colors = ["#b30000", "#1f7a1f"]
best_lcs = []
best_arrows = [[], []]
for color in route_colors:
    best_lc = LineCollection([], colors=color, linewidths=4.5, alpha=0.6)
    ax.add_collection(best_lc)
    best_lcs.append(best_lc)

particle_scatter = ax.scatter(
    np.zeros(NUM_PARTICLES * 2),
    np.zeros(NUM_PARTICLES * 2),
    s=28,
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
    dots = np.array(particle_frames[frame_idx])
    particle_scatter.set_offsets(dots)
    iter_idx = iter_frames[frame_idx] - 1
    iter_text.set_text(f"Iteration {iter_frames[frame_idx]}")

    best_routes = gbest_routes_by_iter[iter_idx]
    best_cost = gbest_cost_by_iter[iter_idx]
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

            route_lines.append(f"Route {STARTS[r_idx]}: {'->'.join(route)}")

        if math.isfinite(best_cost):
            best_text.set_text(
                f"Best cost: {best_cost:.2f}\n" + "\n".join(route_lines)
            )
        else:
            best_text.set_text("Best cost: inf\nRoutes: (infeasible)")
    else:
        for r_idx, best_lc in enumerate(best_lcs):
            best_lc.set_segments([])
            for arrow in best_arrows[r_idx]:
                arrow.set_visible(False)
        best_text.set_text("Best cost: N/A\nRoutes: N/A")

    artists = [particle_scatter, iter_text, best_text]
    artists.extend(best_lcs)
    for arrows in best_arrows:
        artists.extend(arrows)
    return tuple(artists)


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(particle_frames),
    interval=60,
    blit=False
)

if SAVE_GIF:
    ani.save(GIF_PATH, writer="pillow", fps=8)
else:
    plt.show()
