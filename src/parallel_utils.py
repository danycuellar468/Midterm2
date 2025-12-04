"""
Parallel execution utilities for HPC benchmarking.
"""

from multiprocessing import Pool
from functools import partial
from typing import List, Dict, Any
import os
import csv

from .experiments import run_single_experiment


def _run_indexed_exp(idx: int, width: int, height: int, density: float, q_episodes: int, beta: float) -> Dict[str, Any]:
    res = run_single_experiment(width, height, density, q_episodes, beta)
    res["run_id"] = idx
    return res


def run_batch_parallel(n_runs: int = 30,
                       width: int = 40,
                       height: int = 40,
                       density: float = 0.2,
                       q_episodes: int = 500,
                       beta: float = 0.5,
                       out_csv: str = "results/experiments_parallel.csv",
                       n_workers: int = 4):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with Pool(processes=n_workers) as pool:
        func = partial(_run_indexed_exp, width=width, height=height,
                       density=density, q_episodes=q_episodes, beta=beta)
        results: List[Dict[str, Any]] = pool.map(func, range(n_runs))

    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"✓ Parallel experiments completed: {n_runs} runs with {n_workers} workers")


if __name__ == "__main__":
    from .config import GRID_WIDTH, GRID_HEIGHT, OBSTACLE_DENSITY, Q_EPISODES, N_RUNS, N_WORKERS, BETA
    run_batch_parallel(
        n_runs=N_RUNS,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        density=OBSTACLE_DENSITY,
        q_episodes=Q_EPISODES,
        beta=BETA,
        n_workers=N_WORKERS,
        out_csv="results/experiments_parallel.csv"
    )

