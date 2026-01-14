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
# ACO IMPLEMENTATION (2-VEHICLE VRP)
# =========================

class ACO_VRP:
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

        self.history_tau = []
        self.history_best = []

    def heuristic(self, u, v):
        return 1 / self.G[u][v]["weight"]

    def choose_next(self, u, neighbors):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(neighbors)
        weights = []
        total = 0.0
        for v in neighbors:
            value = (self.tau[(u, v)] ** self.alpha) * (self.heuristic(u, v) ** self.beta)
            weights.append(value)
            total += value
        r = self.rng.random() * total
        acc = 0.0
        for v, w in zip(neighbors, weights):
            acc += w
            if acc >= r:
                return v
        return neighbors[-1]

    def run(self, starts):
        best_routes = None
        best_cost = float("inf")
        k = len(starts)

        for _ in range(self.iterations):
            all_routes = []

            for _ in range(self.ants):
                routes = [[s] for s in starts]
                current = list(starts)
                visited = set(starts)

                while len(visited) < self.G.number_of_nodes():
                    progressed = False
                    for i in range(k):
                        if len(visited) >= self.G.number_of_nodes():
                            break
                        neighbors = [n for n in self.G.neighbors(current[i]) if n not in visited]
                        if not neighbors:
                            continue
                        nxt = self.choose_next(current[i], neighbors)
                        routes[i].append(nxt)
                        visited.add(nxt)
                        current[i] = nxt
                        progressed = True
                    if not progressed:
                        break

                if len(visited) != self.G.number_of_nodes():
                    continue

                valid = True
                for i in range(k):
                    if not self.G.has_edge(current[i], starts[i]):
                        valid = False
                        break
                    routes[i].append(starts[i])

                if not valid:
                    continue

                cost = 0.0
                for route in routes:
                    cost += sum(
                        self.G[u][v]["weight"]
                        for u, v in zip(route, route[1:])
                    )

                all_routes.append((routes, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_routes = routes

            for e in self.tau:
                self.tau[e] *= (1 - self.rho)

            for routes, cost in all_routes:
                deposit = self.Q / cost
                for route in routes:
                    for u, v in zip(route, route[1:]):
                        self.tau[(u, v)] += deposit

            self.history_tau.append(dict(self.tau))
            self.history_best.append(best_routes)

        return best_routes, best_cost


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
# RUN ACO (VRP)
# =========================

starts = ["A", "H"]
restarts = 5
base_seed = 1
aco_runs = []
best_routes = None
best_cost = float("inf")

for r in range(restarts):
    aco = ACO_VRP(
        G,
        ants=20,
        iterations=20,
        random_init=True,
        tau0_range=(0.2, 2.0),
        epsilon=0.3,
        seed=base_seed + r
    )
    routes, cost = aco.run(starts)
    aco_runs.append((aco, routes, cost))
    if cost < best_cost:
        best_cost = cost
        best_routes = routes

print("Best routes:", best_routes)
print("Cost:", best_cost)

# =========================
# VISUALIZE ITERATIONS IN FLASK (SCROLLABLE)
# =========================

step = 3
max_iterations_to_show = 16
OUTPUT_DIR = "vrp_aco_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def render_iteration_image(aco_obj, it, restart_idx):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    tau = aco_obj.history_tau[it]
    best = aco_obj.history_best[it]
    best_cost = None
    if best:
        best_cost = 0.0
        for route in best:
            best_cost += sum(
                G[u][v]["weight"]
                for u, v in zip(route, route[1:])
            )

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
        width=2 + 4 * pher_norm,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18
    )

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

    if best:
        colors = [(1.0, 0.0, 0.0, 0.4), (0.0, 0.6, 0.0, 0.4)]
        for idx, route in enumerate(best):
            route_edges = list(zip(route, route[1:]))
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=route_edges,
                edge_color=colors[idx % len(colors)],
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
        limit = min(max_iterations_to_show, len(aco_obj.history_tau))
        iterations_to_show = list(range(0, limit, step))
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
      <title>VRP ACO Iterations</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 16px; background: #f7f7f7; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
        .card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        img { width: 100%; height: auto; display: block; }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>VRP ACO Routes</h2>
        <p><strong>Start nodes:</strong> {{ starts }}</p>
        <p><strong>Routes:</strong> <code>{{ routes }}</code></p>
        <p><strong>Cost:</strong> {{ cost }}</p>
      </div>
      <h2>ACO Iterations (scroll down)</h2>
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
        starts=starts,
        routes=best_routes,
        cost=best_cost
    )


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)
