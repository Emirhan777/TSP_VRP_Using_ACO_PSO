import random
import math
import io
import base64
import webbrowser
import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from flask import Flask, render_template_string

# =========================
# ACO IMPLEMENTATION
# =========================

class ACO:
    def __init__(
        self,
        G,
        alpha=1.0,
        beta=1.0,
        rho=0.3,
        Q=1.0,
        ants=30,
        iterations=100,
        tau0=1.0,
        random_init=False,
        tau0_range=(0.5, 1.5),
        epsilon=0.3,
        seed=1
    ):
        self.G = G
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.ants = ants
        self.iterations = iterations
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        self.tau = defaultdict(lambda: tau0)
        for u, v in G.edges():
            if random_init:
                init_val = self.rng.uniform(tau0_range[0], tau0_range[1])
            else:
                init_val = tau0
            self.tau[(u, v)] = init_val
            self.tau[(v, u)] = init_val

        self.history_tau = []
        self.history_best = []

    def heuristic(self, u, v):
        return 1 / self.G[u][v]["weight"]

    def choose_next(self, u, neighbors):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(neighbors)
        weights = []
        total = 0
        for v in neighbors:
            value = (self.tau[(u, v)] ** self.alpha) * (self.heuristic(u, v) ** self.beta)
            weights.append(value)
            total += value

        r = self.rng.random() * total
        acc = 0
        for v, w in zip(neighbors, weights):
            acc += w
            if acc >= r:
                return v
        return neighbors[-1]

    def run(self, start):
        best_path = None
        best_cost = float("inf")

        for it in range(self.iterations):
            all_paths = []

            for _ in range(self.ants):
                path = [start]
                visited = {start}
                current = start

                while len(visited) < self.G.number_of_nodes():
                    neighbors = [n for n in self.G.neighbors(current) if n not in visited]
                    if not neighbors:
                        break
                    nxt = self.choose_next(current, neighbors)
                    path.append(nxt)
                    visited.add(nxt)
                    current = nxt

                if len(visited) == self.G.number_of_nodes() and self.G.has_edge(current, start):
                    path.append(start)
                    cost = sum(
                        self.G[u][v]["weight"]
                        for u, v in zip(path, path[1:])
                    )
                    all_paths.append((path, cost))
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path

            # Evaporation
            for e in self.tau:
                self.tau[e] *= (1 - self.rho)

            # Deposit pheromone
            for path, cost in all_paths:
                deposit = self.Q / cost
                for u, v in zip(path, path[1:]):
                    self.tau[(u, v)] += deposit
                    self.tau[(v, u)] += deposit

            # Store iteration state
            self.history_tau.append(dict(self.tau))
            self.history_best.append(best_path)

        return best_path, best_cost


# =========================
# GRAPH DEFINITION (LARGER COMPLETE GRAPH)
# =========================

G = nx.DiGraph()

coords = {
    "A": (32.38, 15.08),
    "B": (65.09, 7.24),
    "C": (53.59, 36.57),
    "D": (5.80, 50.74),
    "E": (3.75, 43.36),
    "F": (6.99, 9.07),
    "G": (42.45, 82.69),
    "H": (12.38, 22.32),
    "I": (62.74, 94.77),
    "J": (57.71, 39.67),
    "K": (97.63, 4.66),
    "L": (85.85, 28.96),
    "M": (14.43, 11.78),
    "N": (30.85, 81.61),
    "O": (18.07, 58.16),
    "P": (63.89, 37.24),
    "Q": (54.77, 6.28),
    "R": (5.96, 20.60),
}

nodes = list(coords.keys())
for u in nodes:
    for v in nodes:
        if u == v:
            continue
        dx = coords[u][0] - coords[v][0]
        dy = coords[u][1] - coords[v][1]
        w = int(round(math.hypot(dx, dy)))
        if w == 0:
            w = 1
        G.add_edge(u, v, weight=w)

pos = coords


# =========================
# RUN ACO
# =========================

restarts = 5
base_seed = 1
aco_runs = []
best_path = None
best_cost = float("inf")

for r in range(restarts):
    aco = ACO(
        G,
        ants=20,
        iterations=200,
        random_init=True,
        tau0_range=(0.2, 2.0),
        epsilon=0.3,
        seed=base_seed + r
    )
    path, cost = aco.run("A")
    aco_runs.append((aco, path, cost))
    if cost < best_cost:
        best_cost = cost
        best_path = path

print("Best tour:", best_path)
print("Cost:", best_cost)


# =========================
# VISUALIZE ITERATIONS IN FLASK (SCROLLABLE)
# =========================

# 🔧 ITERATIONS TO SHOW (ONLY WHEN COST CHANGES)
max_iterations_to_show = 200
OUTPUT_DIR = "aco_images_big"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def render_iteration_image(aco_obj, it, restart_idx):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    tau = aco_obj.history_tau[it]
    best_path = aco_obj.history_best[it]
    best_cost = None
    if best_path:
        best_cost = sum(
            G[u][v]["weight"]
            for u, v in zip(best_path, best_path[1:])
        )
    # No per-image cost label

    pher = np.array([tau[(u, v)] for u, v in G.edges()])
    pher_norm = pher / pher.max()

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color="lightblue",
        node_size=600,
        edge_color=pher_norm,
        edge_cmap=plt.cm.plasma,
        width=2 + 4 * pher_norm
    )

    if G.number_of_nodes() <= 10:
        edge_labels = {
            (u, v): f"{tau[(u, v)]:.2f}"
            for u, v in G.edges()
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax
        )

    if best_path:
        best_edges = list(zip(best_path, best_path[1:]))
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

    ax.set_title(f"Iteration {it + 1}")
    ax.axis("off")

    buf = io.BytesIO()
    restart_dir = os.path.join(OUTPUT_DIR, f"restart_{restart_idx + 1}")
    os.makedirs(restart_dir, exist_ok=True)
    file_path = os.path.join(restart_dir, f"iter_{it + 1:03d}.png")
    fig.savefig(file_path, format="png", dpi=200)
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    if best_cost is not None:
        cost_path = os.path.join(restart_dir, f"cost_iter_{it + 1:03d}.png")
        save_cost_image(best_cost, cost_path)
    return base64.b64encode(buf.read()).decode("ascii"), best_cost


def path_cost(path):
    if not path:
        return None
    return sum(
        G[u][v]["weight"]
        for u, v in zip(path, path[1:])
    )


def iterations_with_cost_changes(aco_obj):
    changes = []
    prev_cost = None
    limit = min(max_iterations_to_show, len(aco_obj.history_best))
    for it in range(limit):
        cost = path_cost(aco_obj.history_best[it])
        if cost is None:
            continue
        if prev_cost is None or not math.isclose(cost, prev_cost, rel_tol=1e-9, abs_tol=1e-9):
            changes.append(it)
            prev_cost = cost
    return changes


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


app = Flask(__name__)


@app.route("/")
def index():
    restart_blocks = []
    for r_idx, (aco_obj, _, _) in enumerate(aco_runs):
        iterations_to_show = iterations_with_cost_changes(aco_obj)
        images = []
        for it in iterations_to_show:
            img_b64, best_cost = render_iteration_image(aco_obj, it, r_idx)
            images.append({"image": img_b64, "cost": best_cost})
        restart_blocks.append({"restart": r_idx + 1, "images": images})
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>ACO Iterations (Big Graph)</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 16px; background: #f7f7f7; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
        .card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        img { width: 100%; height: auto; display: block; }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>ACO Tour (Big Graph)</h2>
        <p><strong>Start node:</strong> {{ start }}</p>
        <p><strong>Tour:</strong> <code>{{ path }}</code></p>
        <p><strong>Cost:</strong> {{ cost }}</p>
      </div>
      <h2>ACO Iterations (cost changes only)</h2>
      {% for block in restart_blocks %}
        <h3>Restart {{ block.restart }}</h3>
        <div class="grid">
          {% for img in block.images %}
            <div class="card">
              <img src="data:image/png;base64,{{ img.image }}" />
              <div style="margin-top:6px; font-size:12px;">
                Cost: {{ "%.2f"|format(img.cost) if img.cost is not none else "N/A" }}
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
        restart_blocks=restart_blocks,
        start="A",
        path=best_path,
        cost=best_cost
    )


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)
