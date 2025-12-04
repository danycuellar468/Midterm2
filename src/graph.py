"""
Graph data structures for geometric pathfinding.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import math

NodeId = int


@dataclass
class Node:
    id: NodeId
    x: float
    y: float


@dataclass
class Edge:
    u: NodeId
    v: NodeId
    cost: float


class Graph:
    """
    Directed/undirected graph with adjacency list representation.
    """
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.nodes: Dict[NodeId, Node] = {}
        self.adj: Dict[NodeId, List[Tuple[NodeId, float]]] = {}

    def add_node(self, node_id: NodeId, x: float, y: float):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, x, y)
            self.adj[node_id] = []

    def add_edge(self, u: NodeId, v: NodeId, cost: float, bidirectional: bool = True):
        self.adj[u].append((v, cost))
        if bidirectional and not self.directed:
            self.adj[v].append((u, cost))

    def neighbors(self, u: NodeId) -> Iterable[Tuple[NodeId, float]]:
        return self.adj.get(u, [])

    def euclidean_heuristic(self, u: NodeId, goal: NodeId) -> float:
        """h(u) = Euclidean distance to goal (admissible in this environment)."""
        nu = self.nodes[u]
        ng = self.nodes[goal]
        return math.hypot(nu.x - ng.x, nu.y - ng.y)

