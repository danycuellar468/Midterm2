"""
Main entry point for Q*-A* experiments.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.experiments import run_batch_experiments
from src.parallel_utils import run_batch_parallel
from src.config import *

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Q*-A* Pathfinding Experiments')
    parser.add_argument('--parallel', action='store_true', help='Use parallel execution')
    parser.add_argument('--runs', type=int, default=N_RUNS, help='Number of runs')
    parser.add_argument('--width', type=int, default=GRID_WIDTH, help='Grid width')
    parser.add_argument('--height', type=int, default=GRID_HEIGHT, help='Grid height')
    parser.add_argument('--density', type=float, default=OBSTACLE_DENSITY, help='Obstacle density')
    parser.add_argument('--episodes', type=int, default=Q_EPISODES, help='Q-learning episodes')
    parser.add_argument('--beta', type=float, default=BETA, help='Q*-A* bias weight')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Q*-A* Pathfinding Experiments")
    print("="*60)
    print(f"Grid: {args.width}x{args.height}")
    print(f"Obstacle density: {args.density}")
    print(f"Q-Learning episodes: {args.episodes}")
    print(f"Beta (bias weight): {args.beta}")
    print(f"Runs: {args.runs}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    print("="*60)
    
    if args.parallel:
        run_batch_parallel(
            n_runs=args.runs,
            width=args.width,
            height=args.height,
            density=args.density,
            q_episodes=args.episodes,
            beta=args.beta,
            n_workers=N_WORKERS,
            out_csv="results/experiments_parallel.csv"
        )
    else:
        run_batch_experiments(
            n_runs=args.runs,
            width=args.width,
            height=args.height,
            density=args.density,
            q_episodes=args.episodes,
            beta=args.beta,
            out_csv="results/experiments.csv"
        )
    
    print("\n✓ Experiments completed!")
    print("Results saved to results/experiments.csv")

