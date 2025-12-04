"""
Experiments: run comparisons between A*, Weighted A*, and Q*-A*.
"""

import random
from typing import Dict, Any, List
import csv
import os
import time

from .geometry import build_grid_graph
from .astar import astar_search
from .qlearning import train_q_learning
from .qstar import q_star_search
from . import config
from .config import BETA
from .config import Q_EPISODES


def generate_random_obstacles(width: int, height: int, density: float) -> List[tuple]:
    obstacles = []
    for y in range(height):
        for x in range(width):
            if (x, y) in [(0, 0), (width - 1, height - 1)]:
                continue
            if random.random() < density:
                obstacles.append((x, y))
    return obstacles


def run_single_experiment(width: int = config.GRID_WIDTH,
                          height: int = config.GRID_HEIGHT,
                          density: float = config.OBSTACLE_DENSITY,
                          q_episodes: int = config.Q_EPISODES) -> Dict[str, Any]:

    obstacles = generate_random_obstacles(width, height, density)
    graph, start, goal = build_grid_graph(width, height, obstacles, diagonal=True)

    # A* óptimo
    res_astar = astar_search(graph, start, goal, weight=1.0)

    # Weighted A* (baseline más rápido / menos óptimo)
    res_wastar = astar_search(graph, start, goal, weight=1.5)

    # Entrenamos Q-learning en el mismo mapa (solo 4 direcciones)
    t_train0 = time.time()
    agent = train_q_learning(graph, start, goal,
                             episodes= q_episodes,
                             max_steps=width * height)
    train_time_ms = (time.time() - t_train0) * 1000.0

    # Q*-A*
    res_qstar = q_star_search(graph, start, goal, agent, beta = BETA)

    def cost_ratio(cost, optimal):
        if optimal == float("inf") or cost == float("inf"):
            return float("inf")
        return cost / optimal

    return {
        "astar_cost": res_astar.cost,
        "astar_expanded": res_astar.expanded,
        "astar_time_ms": res_astar.runtime_ms,

        "wastar_cost": res_wastar.cost,
        "wastar_expanded": res_wastar.expanded,
        "wastar_time_ms": res_wastar.runtime_ms,
        "wastar_cost_ratio": cost_ratio(res_wastar.cost, res_astar.cost),

        "qstar_cost": res_qstar.cost,
        "qstar_expanded": res_qstar.expanded,
        "qstar_time_ms": res_qstar.runtime_ms,
        "qstar_cost_ratio": cost_ratio(res_qstar.cost, res_astar.cost),

        "q_train_time_ms": train_time_ms,

        "width": width,
        "height": height,
        "density": density,
        "q_episodes": q_episodes,
    }


def run_batch_experiments(n_runs: int = config.N_RUNS,
                          width: int = config.GRID_WIDTH,
                          height: int = config.GRID_HEIGHT,
                          density: float = config.OBSTACLE_DENSITY,
                          q_episodes: int = config.Q_EPISODES,
                          out_csv: str = "results/experiments.csv"):

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = [
        "astar_cost", "astar_expanded", "astar_time_ms",
        "wastar_cost", "wastar_expanded", "wastar_time_ms", "wastar_cost_ratio",
        "qstar_cost", "qstar_expanded", "qstar_time_ms", "qstar_cost_ratio",
        "q_train_time_ms",
        "width", "height", "density", "q_episodes",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_runs):
            res = run_single_experiment(width, height, density, q_episodes)
            writer.writerow(res)
            print(f"Run {i+1}/{n_runs} done.")


if __name__ == "__main__":
    # smoke test simple
    print("Running single experiment (smoke test)...")
    result = run_single_experiment()
    for k, v in result.items():
        print(f"{k}: {v}")
