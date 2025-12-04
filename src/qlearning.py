"""
Q-Learning agent for pathfinding bias.
"""

import random
from typing import Dict, Tuple
from .graph import Graph, NodeId

QTable = Dict[Tuple[NodeId, NodeId], float]


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
        neighbors = list(graph.neighbors(s))
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
        neighbors = list(graph.neighbors(s))
        if not neighbors:
            return s
        if random.random() < self.epsilon:
            return random.choice(neighbors)[0]
        return self.best_action(graph, s)

    def update(self, s: NodeId, a: NodeId, r: float, s_next: NodeId, graph: Graph):
        # Q(s,a) <- (1-alpha)Q + alpha(r + gamma * max_a' Q(s',a'))
        old = self.get_q(s, a)
        neighbors_next = list(graph.neighbors(s_next))
        if neighbors_next:
            max_next = max(self.get_q(s_next, v) for v, _ in neighbors_next)
        else:
            max_next = 0.0
        target = r + self.gamma * max_next
        self.Q[(s, a)] = (1 - self.alpha) * old + self.alpha * target


def train_q_learning(graph: Graph,
                     start: NodeId,
                     goal: NodeId,
                     episodes: int = 500,
                     max_steps: int = 500) -> QLearningAgent:
    agent = QLearningAgent(alpha=0.2, gamma=0.95, epsilon=0.3)
    
    # Pre-compute heuristic distances for better rewards
    from .astar import astar_search
    optimal_result = astar_search(graph, start, goal, weight=1.0)
    optimal_cost = optimal_result.cost if optimal_result.cost != float('inf') else max_steps
    
    for ep in range(episodes):
        s = start
        steps = 0
        path_cost = 0.0
        
        while steps < max_steps and s != goal:
            a = agent.epsilon_greedy(graph, s)
            s_next = a
            
            # Find edge cost
            edge_cost = 1.0
            for v, cost in graph.neighbors(s):
                if v == s_next:
                    edge_cost = cost
                    break
            
            path_cost += edge_cost
            
            # Reward: negative step cost, bonus for reaching goal, penalty for long paths
            r = -edge_cost
            if s_next == goal:
                r = 100.0 - path_cost  # Bonus minus path cost
            elif path_cost > optimal_cost * 1.5:
                r = -10.0  # Penalty for very long paths
            
            steps += 1
            agent.update(s, a, r, s_next, graph)
            s = s_next
        
        # Epsilon annealing
        agent.epsilon = max(0.05, agent.epsilon * 0.998)
    
    return agent

