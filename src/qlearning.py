"""
Q-Learning agent for pathfinding bias.
"""

import random
from typing import Dict, Tuple, List
from .graph import Graph, NodeId
from .config import Q_EPISODES

QTable = Dict[Tuple[NodeId, NodeId], float]

def cardinal_neighbors(graph: Graph, s: NodeId) -> List[Tuple[NodeId, float]]:
    """Get only cardinal (4-directional) neighbors."""
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


def all_neighbors(graph: Graph, s: NodeId) -> List[Tuple[NodeId, float]]:
    """Get all neighbors (8-directional including diagonals)."""
    return list(graph.neighbors(s))

class QLearningAgent:
    def __init__(self,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 use_all_directions: bool = False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_all_directions = use_all_directions
        self.Q: QTable = {}
        # Pre-computed bias cache for faster lookup during search
        self.bias_cache: Dict[NodeId, float] = {}
        # Select neighbor function based on mode
        self._get_neighbors = all_neighbors if use_all_directions else cardinal_neighbors
    
    def precompute_biases(self, graph: Graph):
        """Pre-compute all bias values to avoid computation during search."""
        self.bias_cache.clear()
        for node_id in graph.nodes.keys():
            neighbors = self._get_neighbors(graph, node_id)
            if neighbors:
                bias = max(self.get_q(node_id, v) for v, _ in neighbors)
            else:
                bias = 0.0
            self.bias_cache[node_id] = bias
    
    def get_bias(self, u: NodeId) -> float:
        """Get pre-computed bias value (fast lookup)."""
        return self.bias_cache.get(u, 0.0)

    def get_q(self, s: NodeId, a: NodeId) -> float:
        return self.Q.get((s, a), 0.0)

    def best_action(self, graph: Graph, s: NodeId) -> NodeId:
        neighbors = self._get_neighbors(graph, s)
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
        neighbors = self._get_neighbors(graph, s)
        if not neighbors:
            return s
        if random.random() < self.epsilon:
            return random.choice(neighbors)[0]
        return self.best_action(graph, s)

    def update(self, s: NodeId, a: NodeId, r: float, s_next: NodeId, graph: Graph):
        # Q(s,a) <- (1-alpha)Q + alpha(r + gamma * max_a' Q(s',a'))
        old = self.get_q(s, a)
        neighbors_next = self._get_neighbors(graph, s_next)
        if neighbors_next:
            max_next = max(self.get_q(s_next, v) for v, _ in neighbors_next)
        else:
            max_next = 0.0
        target = r + self.gamma * max_next
        self.Q[(s, a)] = (1 - self.alpha) * old + self.alpha * target

def train_q_learning(graph: Graph, start: NodeId, goal: NodeId, 
                     episodes: int = Q_EPISODES, max_steps: int = 400,
                     use_all_directions: bool = False) -> QLearningAgent:
    """
    Train Q-learning agent with improved reward shaping.
    
    Args:
        graph: The graph to train on
        start: Start node
        goal: Goal node
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        use_all_directions: If True, use 8-directional movement (including diagonals)
                          If False, use only 4 cardinal directions (default)
    """
    agent = QLearningAgent(use_all_directions=use_all_directions)
    
    # Pre-compute goal position for distance-based rewards
    goal_node = graph.nodes[goal]
    
    for ep in range(episodes):
        s = start
        steps = 0
        prev_dist_to_goal = graph.euclidean_heuristic(start, goal)
        
        while steps < max_steps and s != goal:
            a = agent.epsilon_greedy(graph, s)
            s_next = a
            
            # Improved reward shaping
            dist_to_goal = graph.euclidean_heuristic(s_next, goal)
            
            # Base step cost
            r = -1.0
            
            # Reward for getting closer to goal (encourages efficient paths)
            if dist_to_goal < prev_dist_to_goal:
                r += 0.5  # Small bonus for progress
            elif dist_to_goal > prev_dist_to_goal:
                r -= 0.5  # Penalty for moving away
            
            # Large reward for reaching goal
            if s_next == goal:
                r = 100.0
            
            steps += 1
            agent.update(s, a, r, s_next, graph)
            s = s_next
            prev_dist_to_goal = dist_to_goal

        # annealing de epsilon:
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    return agent