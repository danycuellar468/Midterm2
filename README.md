# Q*-A*: Hybrid A* with Q-Learning Bias

## Project Overview

This project implements and evaluates a hybrid pathfinding algorithm that combines A* with Q-Learning (Q*-A*) and compares it against standard A* and Weighted A*. The goal is to investigate whether a reinforcement-learning-based bias can accelerate geometric pathfinding while preserving near-optimal solution quality.

**Research Question:**  
Can a reinforcement-learning-based bias, learned through Q-Learning, be integrated into A* to reduce node expansions and runtime by at least 30% while keeping path-cost deviation within 5–10% of optimal A*?

---

## Problem Statement

Pathfinding in geometric environments with obstacles is computationally expensive. While A* guarantees optimality, it may expand thousands of nodes in large or cluttered maps. This project evaluates whether learned navigation preferences from Q-Learning can guide A* more intelligently, reducing computational overhead while preserving high-quality solutions.

---

## Algorithms Implemented

1. **A\*** — Optimal pathfinding using an admissible Euclidean heuristic  
2. **Weighted A\*** — Faster but suboptimal baseline  
3. **Q-Learning** — Tabular RL agent using 4 cardinal actions  
4. **Q*-A\*** — Hybrid algorithm that biases A* with Q-value guidance  

---

## Core Approach

Q*-A* modifies the A* priority function:
 f'(n) = g(n) + h(n) - β · bias(n)

Where:  
- `g(n)` = cost from start to n  
- `h(n)` = Euclidean heuristic  
- `bias(n)` = max Q-value among cardinal neighbors  
- `β` = bias weight parameter  

This bias helps A* explore more promising regions of the grid.

---

## Architecture

### System Components

1. **Geometry / Grid Module**  
   - Builds the grid environment  
   - Creates nodes, edges, and adjacency lists  
   - Provides Euclidean heuristic  

2. **Q-Learning Module**  
   - Learns action-value estimates (Q-table)  
   - Restricted to 4 cardinal directions to reduce branching  
   - Provides learned bias for A*  

3. **Search Module**  
   - A*, Weighted A*, Q*-A*  
   - Priority queue based on modified f(n)  

### System Diagram

 ┌───────────────────┐
 │  Geometry / Grid  │
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │   Q-Learning      │
 │   (Q-table)       │
 └─────────┬─────────┘
           │ bias(n)
           ▼
 ┌─────────────────────────┐
 │          Q*-A*          │
 │   f'(n)=g+h-β·bias(n)   │
 └─────────┬──────────────┘
           │
           ▼
       Best Path

---

## Project Structure

midterm2/
├── src/
│ ├── graph.py
│ ├── geometry.py
│ ├── astar.py
│ ├── qlearning.py
│ ├── qstar.py
│ ├── experiments.py
│ ├── parallel_utils.py
│ ├── testing.py
│ └── config.py
├── data/maps/
├── results/
│ ├── logs/
│ └── plots/
├── notebooks/
│ └── analysis.ipynb
└── README.md

---

## Installation

To install dependencies:

pip install -r requirements.txt

---

## Usage

### Run sequential experiments:

 python -m src.experiments

### Run parallel experiments:

python -m src.benchmarks

### Run correctness tests:

python -m src.testing

### Modify configuration:

src/config.py

Parameters include:
- Grid size  
- Obstacle density  
- Q-learning episodes  
- Number of runs  
- Number of workers  

---

## Dataset / Input Description

This project uses procedurally generated geometric grids with:

- Customizable width/height  
- Obstacle density  
- Start/goal positions  
- Controlled random seeds  

Optional fixed maps can be placed in:

data/maps/

All map metadata is logged for reproducibility.

---

## Reproducibility

1. Run sequential experiments  
2. Run parallel experiments  
3. Open the notebook: `notebooks/analysis.ipynb`  
4. Generated CSVs appear in:  
   - `results/experiments.csv`  
   - `results/experiments_parallel.csv`  
5. Plots are saved to:  
   - `results/plots/`  

---

## Evaluation Criteria

We evaluate the three algorithms using:

- **Search effort:** number of node expansions  
- **Runtime:** wall-clock time in milliseconds  
- **Solution quality:** path cost and cost ratio vs optimal A*  

Our expectation was that Q*-A* might reduce expansions and runtime compared to vanilla A*, while keeping the cost close to optimal. The experiments show that Weighted A* achieves large efficiency gains at the expense of slightly higher cost, whereas Q*-A* preserves optimal cost but does not significantly reduce expansions or runtime in this static setting.

---

## Baseline Comparison

- **A\*** — optimal baseline  
- **Weighted A\*** — faster baseline  
- **Q*-A\*** — guided but near-optimal hybrid  

---

## Complexity Analysis

- **A\*, WA\*, Q\*-A\***: `O(E log V)`  
- **Storage**: `O(V + E)`  
- **Q-learning**: `O(Ep × L)` (episodes × steps per episode)

---

## High-Performance Computing (HPC)

Parallel execution uses Python's `multiprocessing` module.

We measure:
- Sequential runtime  
- Parallel runtime  
- Speedup: `S = T_seq / T_par`  

This demonstrates performance scalability across CPU cores.

---

## Results & Discussion

The main CSV (`results/experiments.csv`) stores, for each run:

- Node expansions
- Runtime (ms)
- Path cost
- Cost ratio vs optimal A*
- Grid parameters (width, height, obstacle density, Q episodes)

From the aggregated results we observe:

- **A***: Serves as the optimal baseline. It consistently finds the lowest-cost path, with moderate search effort and runtime.
- **Weighted A***: Achieves a large reduction in node expansions and runtime (often more than 50% fewer expansions) at the cost of a small increase in path cost (≈1–2% above optimal).
- **Q*-A***: Preserves optimal path cost (cost ratio ≈ 1.0) but does **not** significantly reduce expansions or runtime compared to standard A*. In some cases, it is slightly slower due to the overhead of computing the Q-based bias.

In other words, in our static grid setting with a good Euclidean heuristic, the learned Q-bias behaves similarly to the heuristic itself and does not provide a strong additional signal. A* is already very efficient, and the hybrid Q*-A* mostly matches its behavior rather than improving on it.

---

## Limitations & Future Work

- The environment is **static and fully known**. In this setting, A* with a good heuristic is already extremely efficient, which limits the potential gains of adding a Q-learning bias.
- Q-learning uses a tabular representation and is trained offline on each specific map; the learned bias does not generalize across different grids.
- The reward function is very simple (−1 per step, large positive reward at the goal), which tends to reproduce a “move towards the goal” signal already captured by the heuristic.
- Only 4 cardinal actions are used during Q-training.

Future work could explore:

- More informative reward shaping and state features, or neural heuristics.
- Sharing a single Q-function across multiple maps (transfer learning).
- **Dynamic or partially observable environments**, where A* must replan frequently and a learned policy could adapt better to changes; in such scenarios, a hybrid Q*-A* approach might offer clearer benefits over pure A*.
---

## References

- InClassAssignment: Algorithmic Improvement Proposal  
- *Reinforcement Learning with A\* and a Deep Heuristic*  
- Russell & Norvig, *Artificial Intelligence: A Modern Approach*  

---

## Authors

Midterm 2 Project — Advanced Algorithms Course
