"""
Geometric map generation for pathfinding experiments.
"""

from typing import List, Tuple
from .graph import Graph

def build_grid_graph(width: int, height: int,
                     obstacles: List[Tuple[int, int]],
                     diagonal: bool = True) -> Tuple[Graph, int, int]:
    """
    Construye un grafo de celdas libres de un grid width x height.
    Evita que start y goal sean obstáculos.
    Devuelve: (grafo, start_id, goal_id)
    """
    g = Graph(directed=False)
    blocked = set(obstacles)

    start_coord = (0, 0)
    goal_coord = (width - 1, height - 1)
    # aseguramos que no estén bloqueados
    blocked.discard(start_coord)
    blocked.discard(goal_coord)

    def idx(x, y):
        return y * width + x

    # Crear nodos
    for y in range(height):
        for x in range(width):
            if (x, y) in blocked:
                continue
            node_id = idx(x, y)
            g.add_node(node_id, float(x), float(y))

    # Vecinos (4 u 8)
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

    start = idx(*start_coord)
    goal = idx(*goal_coord)
    return g, start, goal
