"""
Correctness testing and baseline comparison.

Se corre con:
    python -m src.testing
"""

from typing import List, Tuple
import math

from .geometry import build_grid_graph
from .astar import astar_search
from .qlearning import train_q_learning
from .qstar import q_star_search
from . import config


def path_is_valid(path: List[int],
                  width: int,
                  height: int,
                  obstacles: List[Tuple[int, int]]) -> bool:
    """Verifica que el camino no salga del grid ni pase por obstáculos."""
    blocked = set(obstacles)
    if not path:
        return False

    def coord(node_id: int) -> Tuple[int, int]:
        x = node_id % width
        y = node_id // width
        return x, y

    for node in path:
        x, y = coord(node)
        if not (0 <= x < width and 0 <= y < height):
            return False
        if (x, y) in blocked:
            return False
    return True


def run_correctness_tests():
    print("=== Correctness & baseline tests ===")

    width = config.GRID_WIDTH
    height = config.GRID_HEIGHT
    density = config.OBSTACLE_DENSITY

    # mapa simple: sin obstáculos
    obstacles: List[Tuple[int, int]] = []
    graph, start, goal = build_grid_graph(width, height, obstacles, diagonal=True)

    print("Test 1: A* optimality on empty grid")

    res_astar = astar_search(graph, start, goal, weight=1.0)
    res_wastar = astar_search(graph, start, goal, weight=1.5)

    assert res_astar.path, "A* should find a path."
    assert path_is_valid(res_astar.path, width, height, obstacles)
    assert path_is_valid(res_wastar.path, width, height, obstacles)

    # En un grid vacío, A* con weight=1 debería ser óptimo
    assert res_wastar.cost >= res_astar.cost - 1e-6, \
        "Weighted A* should not produce a path cheaper than optimal A*."

    print("  OK: A* finds a valid, optimal path; Weighted A* is >= optimal cost.")

    print("Test 2: Q*-A* correctness on simple map")

    # entrenamos Q en el mismo mapa
    agent = train_q_learning(graph, start, goal,
                             episodes=200,  # menos para pruebas
                             max_steps=width * height)

    res_qstar = q_star_search(graph, start, goal, agent, beta=config.BETA)

    assert res_qstar.path, "Q*-A* should also find a path."
    assert path_is_valid(res_qstar.path, width, height, obstacles)

    print("  OK: Q*-A* finds a valid path.")

    print("All tests passed.")


if __name__ == "__main__":
    run_correctness_tests()
