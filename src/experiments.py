"""
Experiments: run comparisons between A*, Weighted A*, and Q*-A*.
"""

import random
from typing import Dict, Any, List
import csv
import os

from .geometry import build_grid_graph
from .astar import astar_search
from .qlearning import train_q_learning
from .qstar import q_star_search
from .config import GRID_WIDTH, GRID_HEIGHT, OBSTACLE_DENSITY, Q_EPISODES, BETA


def generate_random_obstacles(width: int, height: int, density: float) -> List[tuple]:
    obstacles = []
    for y in range(height):
        for x in range(width):
            if (x, y) in [(0,0), (width-1, height-1)]:
                continue
            if random.random() < density:
                obstacles.append((x, y))
    return obstacles


def run_single_experiment(width: int = 40,
                          height: int = 40,
                          density: float = 0.2,
                          q_episodes: int = 500,
                          beta: float = 0.5) -> Dict[str, Any]:
    obstacles = generate_random_obstacles(width, height, density)
    graph, start, goal = build_grid_graph(width, height, obstacles, diagonal=True)

    # A* optimal
    res_astar = astar_search(graph, start, goal, weight=1.0)

    # Weighted A* (e.g., w=1.5)
    res_wastar = astar_search(graph, start, goal, weight=1.5)

    # Train Q-learning on the same map
    agent = train_q_learning(graph, start, goal, episodes=q_episodes, max_steps=width*height)

    # Q*-A*
    res_qstar = q_star_search(graph, start, goal, agent, beta=beta)

    def cost_ratio(cost, optimal):
        if optimal == float("inf") or cost == float("inf") or optimal == 0:
            return float("inf")
        return cost / optimal

    def expansion_reduction(expanded, baseline):
        if baseline == 0:
            return 0.0
        return (1.0 - expanded / baseline) * 100.0

    return {
        "astar_cost": res_astar.cost,
        "astar_expanded": res_astar.expanded,
        "astar_time_ms": res_astar.runtime_ms,

        "wastar_cost": res_wastar.cost,
        "wastar_expanded": res_wastar.expanded,
        "wastar_time_ms": res_wastar.runtime_ms,
        "wastar_cost_ratio": cost_ratio(res_wastar.cost, res_astar.cost),
        "wastar_expansion_reduction": expansion_reduction(res_wastar.expanded, res_astar.expanded),

        "qstar_cost": res_qstar.cost,
        "qstar_expanded": res_qstar.expanded,
        "qstar_time_ms": res_qstar.runtime_ms,
        "qstar_cost_ratio": cost_ratio(res_qstar.cost, res_astar.cost),
        "qstar_expansion_reduction": expansion_reduction(res_qstar.expanded, res_astar.expanded),

        "width": width,
        "height": height,
        "density": density,
        "q_episodes": q_episodes,
        "beta": beta,
    }


def run_batch_experiments(n_runs: int = 30,
                          width: int = 40,
                          height: int = 40,
                          density: float = 0.2,
                          q_episodes: int = 500,
                          beta: float = 0.5,
                          out_csv: str = "results/experiments.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = [
        "astar_cost", "astar_expanded", "astar_time_ms",
        "wastar_cost", "wastar_expanded", "wastar_time_ms", "wastar_cost_ratio", "wastar_expansion_reduction",
        "qstar_cost", "qstar_expanded", "qstar_time_ms", "qstar_cost_ratio", "qstar_expansion_reduction",
        "width", "height", "density", "q_episodes", "beta"
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_runs):
            res = run_single_experiment(width, height, density, q_episodes, beta)
            writer.writerow(res)
            print(f"Run {i+1}/{n_runs} done.")


if __name__ == "__main__":
    from .config import GRID_WIDTH, GRID_HEIGHT, OBSTACLE_DENSITY, Q_EPISODES, N_RUNS, BETA
    run_batch_experiments(
        n_runs=N_RUNS,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        density=OBSTACLE_DENSITY,
        q_episodes=Q_EPISODES,
        beta=BETA,
        out_csv="results/experiments.csv"
    )

