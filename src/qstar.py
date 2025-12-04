"""
Q*-A*: Hybrid A* with Q-Learning bias.
"""

import heapq
import time
from typing import Dict, Optional, List, Tuple
from .graph import Graph, NodeId
from .qlearning import QLearningAgent
from .astar import reconstruct_path, AStarResult


def q_star_search(graph: Graph,
                  start: NodeId,
                  goal: NodeId,
                  q_agent: QLearningAgent,
                  beta: float = 0.5) -> AStarResult:
    """
    Q*-A*: A* guided by Q-learning bias.
    Does NOT guarantee exact optimality, but aims to reduce expansions and runtime
    with cost deviation <= 5-10%.
    """
    t0 = time.time()
    open_heap: List[Tuple[float, NodeId]] = []
    g_cost: Dict[NodeId, float] = {start: 0.0}
    parent: Dict[NodeId, Optional[NodeId]] = {start: None}
    expanded = 0

    def q_bias(u: NodeId) -> float:
        """Compute Q-bias: use Q-value to guide search toward promising regions"""
        neighbors = list(graph.neighbors(u))
        if not neighbors:
            return 0.0
        
        # Get Q-values and find the best Q-value (most promising action)
        q_values = []
        for v, _ in neighbors:
            q_val = q_agent.get_q(u, v)
            q_values.append(q_val)
        
        if not q_values:
            return 0.0
        
        max_q = max(q_values)
        avg_q = sum(q_values) / len(q_values)
        
        # Use the difference between max and average as bias strength
        # This captures how "confident" the Q-learning is about the best action
        q_spread = max_q - avg_q
        
        # Normalize: if Q-values are in typical range [-100, 100], scale appropriately
        # Higher max_q relative to average = more promising node
        if q_spread > 1.0:
            # Significant difference, use it
            return min(1.0, q_spread / 20.0)  # Cap at 1.0, scale by 20
        elif max_q > 0:
            # Positive Q-values, use scaled version
            return min(0.5, max_q / 50.0)
        else:
            # All negative or similar, minimal bias
            return 0.0

    h_start = graph.euclidean_heuristic(start, goal)
    f_start = g_cost[start] + h_start - beta * q_bias(start)
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
                bias = q_bias(v)
                # Apply bias: f'(v) = g(v) + h(v) - β·bias(v)
                # Higher bias (better Q) reduces f', making node expand earlier
                f_v = tentative + h_v - beta * bias * h_v  # Scale bias by heuristic
                heapq.heappush(open_heap, (f_v, v))

    runtime_ms = (time.time() - t0) * 1000.0
    return AStarResult([], float("inf"), expanded, runtime_ms)

