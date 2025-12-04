"""
Q-Learning agent for pathfinding bias.
"""

import random
from typing import Dict, Tuple, List
from .graph import Graph, NodeId
from .config import Q_EPISODES

QTable = Dict[Tuple[NodeId, NodeId], float]

def cardinal_neighbors(graph: Graph, s: NodeId) -> List[Tuple[NodeId, float]]:
    ns = []
    node_s = graph.nodes[s]
    xs, ys = node_s.x, node_s.y

    for v, cost in graph.neighbors(s):
        node_v = graph.nodes[v]
        xv, yv = node_v.x, node_v.y
        dx = xv - xs
        dy = yv - ys
        # movimiento cardinal <=> uno de dx,dy es 0 y el otro es ±1
        if (abs(dx) + abs(dy)) == 1:
            ns.append((v, cost))
    return ns

class QLearningAgent:
    def __init__(self,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: QTable = {}

    def get_q(self, s: NodeId, a: NodeId) -> float:
        return self.Q.get((s, a), 0.0)

    def best_action(self, graph: Graph, s: NodeId) -> NodeId:
        neighbors = cardinal_neighbors(graph, s)
        if not neighbors:
            return s
        best_a = None
        best_q = float("-inf")
        for v, _ in neighbors:
            q = self.get_q(s, v)
            if q > best_q:
                best_q = q
                best_a = v
        return best_a if best_a is not None else neighbors[0][0]

    def epsilon_greedy(self, graph: Graph, s: NodeId) -> NodeId:
        neighbors = cardinal_neighbors(graph, s)
        if not neighbors:
            return s
        if random.random() < self.epsilon:
            return random.choice(neighbors)[0]
        return self.best_action(graph, s)

    def update(self, s: NodeId, a: NodeId, r: float, s_next: NodeId, graph: Graph):
        # Q(s,a) <- (1-alpha)Q + alpha(r + gamma * max_a' Q(s',a'))
        old = self.get_q(s, a)
        neighbors_next = cardinal_neighbors(graph, s_next)
        if neighbors_next:
            max_next = max(self.get_q(s_next, v) for v, _ in neighbors_next)
        else:
            max_next = 0.0
        target = r + self.gamma * max_next
        self.Q[(s, a)] = (1 - self.alpha) * old + self.alpha * target

def train_q_learning(graph: Graph, start: NodeId, goal: NodeId, episodes: int = Q_EPISODES, max_steps: int = 400) -> QLearningAgent:
    agent = QLearningAgent()
    for ep in range(episodes):
        s = start
        steps = 0
        while steps < max_steps and s != goal:
            a = agent.epsilon_greedy(graph, s)
            s_next = a  # acción = vecino cardinal
            # recompensa por paso:
            r = -1.0
            if s_next == goal:
                r = 100.0
            steps += 1
            agent.update(s, a, r, s_next, graph)
            s = s_next

        # annealing de epsilon:
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    return agent