import math
import io
import base64
import webbrowser
import random
import os

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template_string

# =========================
# GRAPH DEFINITION (TWO-WAY DIRECTED GRAPH)
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

# =========================
# PSO FOR TSP (PERMUTATION)
# =========================

START_NODE = "A"
RNG = random.Random(7)

nodes = [n for n in G.nodes() if n != START_NODE]


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


def pso_tsp(
    num_particles=80,
    iterations=20,
    restarts=5,
    w_start=0.9,
    w_end=0.4,
    c1=0.7,
    c2=0.9,
    mutation_rate=0.15,
    local_search=True
):
    best_overall = (None, math.inf)
    history = []

    for _ in range(restarts):
        particles = []
        for _ in range(num_particles):
            order = list(nodes)
            RNG.shuffle(order)
            particles.append(Particle(order))

        global_best = min(particles, key=lambda p: p.cost)
        gbest_pos = list(global_best.position)
        gbest_cost = global_best.cost

        restart_history = []
        for it in range(iterations):
            w = w_start - (w_start - w_end) * (it / max(iterations - 1, 1))
            for p in particles:
                v1 = swaps_to_transform(p.position, p.best_position)
                v2 = swaps_to_transform(p.position, gbest_pos)

                keep = int(w * len(p.velocity))
                new_velocity = p.velocity[:keep]

                for sw in v1:
                    if RNG.random() < c1:
                        new_velocity.append(sw)
                for sw in v2:
                    if RNG.random() < c2:
                        new_velocity.append(sw)

                p.position = apply_swaps(p.position, new_velocity)
                p.velocity = new_velocity

                if RNG.random() < mutation_rate:
                    i, j = RNG.sample(range(len(p.position)), 2)
                    p.position[i], p.position[j] = p.position[j], p.position[i]

                if local_search:
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

            restart_history.append((list(gbest_pos), gbest_cost))

        if gbest_cost < best_overall[1]:
            best_overall = (list(gbest_pos), gbest_cost)

        history.append(restart_history)

    return best_overall[0], best_overall[1], history


best_order, best_cost, history = pso_tsp()
best_path = [START_NODE] + best_order + [START_NODE]

print("PSO tour:", best_path)
print("Cost:", best_cost)


def render_solution_image(path):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color="lightblue",
        node_size=600,
        edge_color="#999999",
        width=2,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18
    )

    if path:
        best_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_edges,
            edge_color=(1.0, 0.0, 0.0, 0.4),
            width=12,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=28,
            ax=ax
        )

    ax.set_title("PSO TSP Tour")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


solution_image = render_solution_image(best_path)

# Show iterations 1 to 6 per restart.
step = 1
max_iterations_to_show = 6


def render_iteration_image(path, it_index):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color="lightblue",
        node_size=600,
        edge_color="#999999",
        width=2,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18
    )

    if path:
        best_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=best_edges,
            edge_color=(1.0, 0.0, 0.0, 0.4),
            width=12,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=28,
            ax=ax
        )

    ax.set_title(f"Iteration {it_index + 1}")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def save_cost_image(cost, path):
    fig, ax = plt.subplots(figsize=(2.4, 0.6))
    fig.patch.set_alpha(0.0)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Cost: {cost:.2f}",
        ha="center",
        va="center",
        fontsize=16,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2}
    )
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


OUTPUT_DIR = "pso_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

restart_blocks = []
for r_idx, restart_history in enumerate(history, start=1):
    count = min(max_iterations_to_show, len(restart_history))
    images = []
    for it in range(0, count, step):
        order, iter_cost = restart_history[it]
        path = [START_NODE] + order + [START_NODE]
        img_b64 = render_iteration_image(path, it)
        restart_dir = os.path.join(OUTPUT_DIR, f"restart_{r_idx}")
        os.makedirs(restart_dir, exist_ok=True)
        file_path = os.path.join(restart_dir, f"iter_{it + 1:03d}.png")
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        cost_path = os.path.join(restart_dir, f"cost_iter_{it + 1:03d}.png")
        save_cost_image(iter_cost, cost_path)
        images.append({"iteration": it + 1, "image": img_b64, "cost": iter_cost})
    restart_blocks.append({"restart": r_idx, "images": images})

app = Flask(__name__)


@app.route("/")
def index():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>PSO TSP Solution</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; }
        .card { background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        code { background: #eee; padding: 2px 6px; border-radius: 4px; }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>PSO Tour</h2>
        <p><strong>Start node:</strong> {{ start }}</p>
        <p><strong>Tour:</strong> <code>{{ path }}</code></p>
        <p><strong>Cost:</strong> {{ cost }}</p>
      </div>
      {% for block in restart_blocks %}
        <h3 style="margin-top:24px;">Restart {{ block.restart }}</h3>
        <div style="display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap:10px;">
          {% for item in block.images %}
            <div class="card">
              <img src="data:image/png;base64,{{ item.image }}" style="width:100%; height:auto;" />
              <div style="margin-top:6px; font-size:12px;">
                Cost: {{ "%.2f"|format(item.cost) }}
              </div>
            </div>
          {% endfor %}
        </div>
      {% endfor %}
    </body>
    </html>
    """
    return render_template_string(
        html,
        start=START_NODE,
        path=best_path,
        cost=best_cost,
        image=solution_image,
        restart_blocks=restart_blocks
    )


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)
