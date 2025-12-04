"""
Q*-A*: Hybrid A* with Q-Learning bias.
"""

import heapq
import time
from typing import Dict, Optional, List, Tuple
from .graph import Graph, NodeId
from .qlearning import QLearningAgent, cardinal_neighbors
from .astar import reconstruct_path, AStarResult
from .config import BETA

def q_star_search(graph: Graph, start: NodeId, goal: NodeId, q_agent: QLearningAgent, beta = BETA) -> AStarResult:
    t0 = time.time()
    open_heap: List[Tuple[float, NodeId]] = []
    g_cost: Dict[NodeId, float] = {start: 0.0}
    parent: Dict[NodeId, Optional[NodeId]] = {start: None}
    expanded = 0

    def q_bias(u: NodeId) -> float:
        neighbors = cardinal_neighbors(graph, u)
        if not neighbors:
            return 0.0
        return max(q_agent.get_q(u, v) for v, _ in neighbors)

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
                f_v = tentative + h_v - beta * q_bias(v)
                heapq.heappush(open_heap, (f_v, v))

    runtime_ms = (time.time() - t0) * 1000.0
    return AStarResult([], float("inf"), expanded, runtime_ms)