"""
A* and Weighted A* implementations.
"""

import heapq
from typing import Dict, Optional, List, Tuple
from .graph import Graph, NodeId


class AStarResult:
    def __init__(self,
                 path: List[NodeId],
                 cost: float,
                 expanded: int,
                 runtime_ms: float):
        self.path = path
        self.cost = cost
        self.expanded = expanded
        self.runtime_ms = runtime_ms


def reconstruct_path(parent: Dict[NodeId, Optional[NodeId]],
                     start: NodeId, goal: NodeId) -> List[NodeId]:
    if goal not in parent:
        return []
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


def astar_search(graph: Graph, start: NodeId, goal: NodeId,
                 weight: float = 1.0) -> AStarResult:
    """
    Standard A*. If weight > 1 acts as Weighted A* (less optimal, faster).
    """
    import time
    t0 = time.time()

    open_heap: List[Tuple[float, NodeId]] = []
    g_cost: Dict[NodeId, float] = {start: 0.0}
    parent: Dict[NodeId, Optional[NodeId]] = {start: None}
    expanded = 0

    h_start = graph.euclidean_heuristic(start, goal)
    heapq.heappush(open_heap, (h_start, start))

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
                f_v = tentative + weight * h_v
                heapq.heappush(open_heap, (f_v, v))

    runtime_ms = (time.time() - t0) * 1000.0
    return AStarResult([], float("inf"), expanded, runtime_ms)

