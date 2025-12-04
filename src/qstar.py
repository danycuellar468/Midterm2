"""
Q*-A*: Hybrid A* with Q-Learning bias.
"""

import heapq
import time
from typing import Dict, Optional, List, Tuple
from .graph import Graph, NodeId
from .qlearning import QLearningAgent, cardinal_neighbors, all_neighbors
from .astar import reconstruct_path, AStarResult
from .config import BETA

def q_star_search(graph: Graph, start: NodeId, goal: NodeId, q_agent: QLearningAgent, beta = BETA) -> AStarResult:
    t0 = time.time()
    open_heap: List[Tuple[float, NodeId]] = []
    g_cost: Dict[NodeId, float] = {start: 0.0}
    parent: Dict[NodeId, Optional[NodeId]] = {start: None}
    expanded = 0
    
    def q_bias(u: NodeId) -> float:
        """Get Q-bias value (uses pre-computed cache if available)."""
        return q_agent.get_bias(u)
    
    def compute_weight(u: NodeId) -> float:
        """
        Compute adaptive weight based on Q-values (Weighted A* style).
        Uses cached bias to minimize overhead.
        """
        bias = q_bias(u)
        # If no Q-value info, use standard A* (weight=1.0)
        if bias == 0.0:
            return 1.0
        # Q-values are negative. More negative = further from goal = use higher weight
        # Map to weight range [1.0, 1.0 + beta] similar to Weighted A*
        # Typical Q-values: -10 (far) to -5 (close), map to weights 1.0+beta to 1.0
        # Invert: more negative Q -> higher weight (faster search, less optimal)
        normalized = max(0.0, min(1.0, -bias / 10.0))  # Normalize to [0, 1]
        weight = 1.0 + normalized * beta
        return weight

    h_start = graph.euclidean_heuristic(start, goal)
    weight_start = compute_weight(start)
    f_start = g_cost[start] + weight_start * h_start
    heapq.heappush(open_heap, (f_start, start))

    closed = set()

    while open_heap:
        f_u, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1

        if u == goal:
            path = reconstruct_path(parent, start, goal)
            runtime_ms = (time.time() - t0) * 1000.0
            return AStarResult(path, g_cost[goal], expanded, runtime_ms)

        for v, cost_uv in graph.neighbors(u):
            tentative = g_cost[u] + cost_uv
            if v not in g_cost or tentative < g_cost[v]:
                g_cost[v] = tentative
                parent[v] = u
                h_v = graph.euclidean_heuristic(v, goal)
                weight_v = compute_weight(v)
                f_v = tentative + weight_v * h_v
                heapq.heappush(open_heap, (f_v, v))

    runtime_ms = (time.time() - t0) * 1000.0
    return AStarResult([], float("inf"), expanded, runtime_ms)