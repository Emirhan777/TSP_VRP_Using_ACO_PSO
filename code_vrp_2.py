import math
import io
import base64
import webbrowser
import os
from itertools import combinations

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
# EXACT TSP (HELD-KARP)
# =========================


def edge_weight(u, v):
    if G.has_edge(u, v):
        return G[u][v]["weight"]
    return math.inf


def held_karp_tsp(graph, start, nodes_subset):
    if not nodes_subset:
        return None, math.inf

    nodes = list(nodes_subset)
    n = len(nodes)
    dp = {}

    for j, node in enumerate(nodes):
        w = edge_weight(start, node)
        if math.isfinite(w):
            dp[(1 << j, j)] = (w, None)

    for r in range(2, n + 1):
        for subset in combinations(range(n), r):
            mask = 0
            for idx in subset:
                mask |= 1 << idx
            for j in subset:
                best = (math.inf, None)
                prev_mask = mask & ~(1 << j)
                for k in subset:
                    if k == j:
                        continue
                    if (prev_mask, k) not in dp:
                        continue
                    prev_cost, _ = dp[(prev_mask, k)]
                    w = edge_weight(nodes[k], nodes[j])
                    cost = prev_cost + w
                    if cost < best[0]:
                        best = (cost, k)
                if math.isfinite(best[0]):
                    dp[(mask, j)] = best

    full_mask = (1 << n) - 1
    best_end = (math.inf, None)
    for j in range(n):
        if (full_mask, j) not in dp:
            continue
        cost_to_j, _ = dp[(full_mask, j)]
        w = edge_weight(nodes[j], start)
        total_cost = cost_to_j + w
        if total_cost < best_end[0]:
            best_end = (total_cost, j)

    if not math.isfinite(best_end[0]):
        return None, math.inf

    path = [start]
    mask = full_mask
    j = best_end[1]
    order = []
    while j is not None:
        order.append(nodes[j])
        _, prev = dp[(mask, j)]
        mask &= ~(1 << j)
        j = prev
    order.reverse()
    path.extend(order)
    path.append(start)
    return path, best_end[0]


# =========================
# EXACT VRP (2 VEHICLES)
# =========================


def exact_vrp_two_vehicles(graph, start_a, start_b):
    nodes = [n for n in graph.nodes() if n not in (start_a, start_b)]
    n = len(nodes)
    full_mask = (1 << n) - 1

    cache_a = {}
    cache_b = {}
    for mask in range(1 << n):
        subset = [nodes[i] for i in range(n) if mask & (1 << i)]
        cache_a[mask] = held_karp_tsp(graph, start_a, subset)
        cache_b[mask] = held_karp_tsp(graph, start_b, subset)

    best_total = math.inf
    best_routes = None

    for mask in range(1 << n):
        if mask == 0 or mask == full_mask:
            continue
        path_a, cost_a = cache_a[mask]
        path_b, cost_b = cache_b[full_mask ^ mask]
        if not math.isfinite(cost_a) or not math.isfinite(cost_b):
            continue
        total = cost_a + cost_b
        if total < best_total:
            best_total = total
            best_routes = [path_a, path_b]

    return best_routes, best_total


def route_cost(route):
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(route, route[1:]):
        w = edge_weight(u, v)
        if not math.isfinite(w):
            return math.inf
        total += w
    return total


# =========================
# RUN EXACT VRP
# =========================

STARTS = ["A", "H"]

best_routes, best_cost = exact_vrp_two_vehicles(G, STARTS[0], STARTS[1])

print("Exact VRP routes:", best_routes)
print("Cost:", best_cost)


# =========================
# VISUALIZE SOLUTION
# =========================


def render_solution_image(routes):
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

    if routes:
        colors = [(1.0, 0.0, 0.0, 0.4), (0.0, 0.6, 0.0, 0.4)]
        for idx, route in enumerate(routes):
            if route and len(route) > 1:
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

    ax.set_title("Exact VRP Solution")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


solution_image = render_solution_image(best_routes)
OUTPUT_DIR = "vrp_exact_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "exact_solution.png"), "wb") as f:
    f.write(base64.b64decode(solution_image))


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


save_cost_image(best_cost, os.path.join(OUTPUT_DIR, "cost_exact.png"))


# =========================
# SERVE WITH FLASK
# =========================

app = Flask(__name__)


@app.route("/")
def index():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Exact VRP Solution</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; }
        .card { background: white; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        code { background: #eee; padding: 2px 6px; border-radius: 4px; }
      </style>
    </head>
    <body>
      <div class="card">
        <h2>Exact VRP (2 Vehicles)</h2>
        <p><strong>Start nodes:</strong> {{ starts }}</p>
        <p><strong>Route A:</strong> <code>{{ route_a }}</code></p>
        <p><strong>Route H:</strong> <code>{{ route_b }}</code></p>
        <p><strong>Cost A:</strong> {{ cost_a }}</p>
        <p><strong>Cost H:</strong> {{ cost_b }}</p>
        <p><strong>Total cost:</strong> {{ total_cost }}</p>
        <img src="data:image/png;base64,{{ image }}" style="width:320px; height:auto; margin-top:12px;" />
      </div>
    </body>
    </html>
    """
    route_a = best_routes[0] if best_routes else None
    route_b = best_routes[1] if best_routes else None
    return render_template_string(
        html,
        starts=STARTS,
        route_a=route_a,
        route_b=route_b,
        cost_a=f"{route_cost(route_a):.2f}" if route_a else "N/A",
        cost_b=f"{route_cost(route_b):.2f}" if route_b else "N/A",
        total_cost=f"{best_cost:.2f}" if math.isfinite(best_cost) else "N/A",
        image=solution_image
    )


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)
