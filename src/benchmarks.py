"""
Benchmark runner for Midterm 2 project.

- Corre un batch secuencial de experimentos
- Corre un batch paralelo con multiprocessing
- Compara tiempos y genera CSVs

Se ejecuta con:
    python -m src.benchmarks
"""

import time
from . import config
from .experiments import run_batch_experiments
from .parallel_utils import run_batch_parallel


def run_sequential_benchmark():
    print("=== Sequential benchmark ===")
    t0 = time.time()
    run_batch_experiments(
        n_runs=config.N_RUNS,
        width=config.GRID_WIDTH,
        height=config.GRID_HEIGHT,
        density=config.OBSTACLE_DENSITY,
        q_episodes=config.Q_EPISODES,
        out_csv="results/experiments_sequential.csv"
    )
    elapsed = time.time() - t0
    print(f"Sequential total time: {elapsed:.2f} s")
    return elapsed


def run_parallel_benchmark():
    print("=== Parallel benchmark ===")
    t0 = time.time()
    run_batch_parallel(
        n_runs=config.N_RUNS,
        width=config.GRID_WIDTH,
        height=config.GRID_HEIGHT,
        density=config.OBSTACLE_DENSITY,
        q_episodes=config.Q_EPISODES,
        out_csv="results/experiments_parallel.csv",
        n_workers=config.N_WORKERS
    )
    elapsed = time.time() - t0
    print(f"Parallel total time: {elapsed:.2f} s")
    return elapsed


if __name__ == "__main__":
    seq_time = run_sequential_benchmark()
    par_time = run_parallel_benchmark()
    speedup = seq_time / par_time if par_time > 0 else float("inf")
    print(f"Approximate speedup (sequential / parallel): {speedup:.2f}x")
