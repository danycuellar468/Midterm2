"""
Geometric map generation for pathfinding experiments.
"""

from typing import List, Tuple
from .graph import Graph


def build_grid_graph(width: int, height: int,
                     obstacles: List[Tuple[int, int]],
                     diagonal: bool = True) -> Tuple[Graph, int, int]:
    """
    Builds a graph from free cells of a width x height grid.
    Returns: (graph, start_id, goal_id)
    By default: start=(0,0), goal=(width-1, height-1).
    """
    g = Graph(directed=False)
    blocked = set(obstacles)

    def idx(x, y):
        return y * width + x

    # Create nodes
    for y in range(height):
        for x in range(width):
            if (x, y) in blocked:
                continue
            node_id = idx(x, y)
            g.add_node(node_id, float(x), float(y))

    # Neighbors (4 or 8 directions)
    directions_4 = [(1,0), (-1,0), (0,1), (0,-1)]
    directions_diag = [(1,1), (1,-1), (-1,1), (-1,-1)]
    directions = directions_4 + (directions_diag if diagonal else [])

    import math
    for y in range(height):
        for x in range(width):
            if (x, y) in blocked:
                continue
            u = idx(x, y)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked:
                    v = idx(nx, ny)
                    cost = math.hypot(dx, dy)
                    g.add_edge(u, v, cost, bidirectional=True)

    start = idx(0, 0)
    goal = idx(width-1, height-1)
    return g, start, goal

