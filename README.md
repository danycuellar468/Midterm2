# Q*-A*: Hybrid A* with Q-Learning Bias

## Project Overview

This project implements a hybrid pathfinding algorithm that combines A* with Q-Learning to accelerate geometric pathfinding while maintaining near-optimal solution quality.

**Research Question:** Can a reinforcement-learning-based bias, learned through Q-Learning, be integrated into A* to reduce node expansions and runtime by at least 30% while keeping path-cost deviation within 5-10% of optimal A*?

## Problem Statement

Pathfinding in geometric environments with obstacles is computationally expensive. While A* guarantees optimality, it can expand thousands of nodes in large or cluttered maps. This project investigates whether learned navigation preferences from Q-Learning can guide A* more intelligently, reducing computational effort without sacrificing solution quality.

## Algorithms Implemented

1. **A*** - Classical optimal pathfinding with admissible heuristic
2. **Weighted A*** - Faster but suboptimal variant (baseline)
3. **Q-Learning** - Reinforcement learning agent that learns action-values
4. **Q*-A*** - Hybrid algorithm combining A* with Q-Learning bias

## Core Approach

Q*-A* uses a modified priority function:
```
f'(n) = g(n) + h(n) - β · bias(n)
```

Where:
- `g(n)` = cost from start to node n
- `h(n)` = admissible heuristic (Euclidean distance)
- `bias(n)` = max Q-value from node n (learned navigation preference)
- `β` = bias weight parameter (tunable)

## Project Structure

```
midterm/
├── src/
│   ├── graph.py          # Graph data structures
│   ├── geometry.py       # Grid map generation
│   ├── astar.py          # A* and Weighted A*
│   ├── qlearning.py      # Q-Learning agent
│   ├── qstar.py          # Q*-A* hybrid algorithm
│   ├── experiments.py    # Experiment runner
│   ├── parallel_utils.py # HPC parallel execution
│   └── config.py         # Configuration parameters
├── results/
│   ├── logs/             # Experiment logs
│   └── plots/            # Generated plots
├── notebooks/
│   └── analysis.ipynb     # Data analysis
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Sequential Experiments

```bash
python -m src.experiments
```

### Run Parallel Experiments (HPC)

```bash
python -m src.parallel_utils
```

### Configuration

Edit `src/config.py` to adjust:
- Grid size (width, height)
- Obstacle density
- Q-Learning episodes
- Number of runs
- Number of parallel workers

## Success Criteria

1. **Efficiency:** At least 15% reduction in node expansions and runtime
2. **Quality:** Path cost deviation ≤ 10-20% from optimal A*

## Baseline Comparison

- **A*** (optimal): Guarantees optimality but computationally expensive
- **Weighted A***: Faster but suboptimal baseline

## Complexity Analysis

- **Time:** O(E log V) for all algorithms
- **Space:** O(V + E) for graph and search structures
- **Q-Learning Training:** O(Ep · L) where Ep = episodes, L = max steps

## Results

Results are saved to `results/experiments.csv` and `results/experiments_parallel.csv` with metrics:
- Node expansions
- Runtime (ms)
- Path cost
- Cost ratio vs optimal
- Expansion reduction percentage

## References

- InClassAssignment: Algorithmic proposal document
- Article: "Reinforcement Learning with A* and a Deep Heuristic" (recommended reading)

## Authors

Midterm 2 Project - Advanced Algorithms Course

