"""
Diagnostic tools to analyze Q*A* behavior and identify performance issues.
"""

from typing import Dict, List, Tuple
from .graph import Graph, NodeId
from .qlearning import QLearningAgent, cardinal_neighbors, all_neighbors


def analyze_q_coverage(graph: Graph, q_agent: QLearningAgent, sample_nodes: List[NodeId] = None) -> Dict:
    """
    Analyze how many neighbors Q-learning covers vs how many A* considers.
    
    Returns statistics about:
    - Average number of neighbors per node
    - Average number of cardinal neighbors (Q-learning coverage)
    - Percentage of neighbors covered by Q-learning
    - Number of nodes with Q-values
    """
    if sample_nodes is None:
        sample_nodes = list(graph.nodes.keys())[:100]  # Sample first 100 nodes
    
    total_neighbors = 0
    total_cardinal = 0
    nodes_with_q_values = 0
    q_value_count = 0
    
    for node_id in sample_nodes:
        all_neighbors_list = list(graph.neighbors(node_id))
        # Use appropriate neighbor function based on agent mode
        if q_agent.use_all_directions:
            q_neighbors = all_neighbors(graph, node_id)
        else:
            q_neighbors = cardinal_neighbors(graph, node_id)
        
        total_neighbors += len(all_neighbors_list)
        total_cardinal += len(q_neighbors)
        
        # Check Q-values for this node
        has_q_values = False
        for neighbor_id, _ in q_neighbors:
            q_val = q_agent.get_q(node_id, neighbor_id)
            if q_val != 0.0:  # Has learned Q-value
                has_q_values = True
                q_value_count += 1
        
        if has_q_values:
            nodes_with_q_values += 1
    
    n_samples = len(sample_nodes)
    avg_all_neighbors = total_neighbors / n_samples if n_samples > 0 else 0
    avg_cardinal = total_cardinal / n_samples if n_samples > 0 else 0
    coverage_pct = (avg_cardinal / avg_all_neighbors * 100) if avg_all_neighbors > 0 else 0
    
    return {
        'avg_all_neighbors': avg_all_neighbors,
        'avg_cardinal_neighbors': avg_cardinal,
        'coverage_percentage': coverage_pct,
        'nodes_with_q_values': nodes_with_q_values,
        'total_q_values': q_value_count,
        'sample_size': n_samples
    }


def analyze_q_bias_distribution(graph: Graph, q_agent: QLearningAgent, 
                                sample_nodes: List[NodeId] = None) -> Dict:
    """
    Analyze the distribution of Q-bias values.
    """
    if sample_nodes is None:
        sample_nodes = list(graph.nodes.keys())[:100]
    
    bias_values = []
    zero_bias_count = 0
    
    for node_id in sample_nodes:
        # Use appropriate neighbor function based on agent mode
        if q_agent.use_all_directions:
            neighbors = all_neighbors(graph, node_id)
        else:
            neighbors = cardinal_neighbors(graph, node_id)
        if neighbors:
            bias = max(q_agent.get_q(node_id, v) for v, _ in neighbors)
            bias_values.append(bias)
            if bias == 0.0:
                zero_bias_count += 1
        else:
            bias_values.append(0.0)
            zero_bias_count += 1
    
    import statistics
    return {
        'mean_bias': statistics.mean(bias_values) if bias_values else 0.0,
        'median_bias': statistics.median(bias_values) if bias_values else 0.0,
        'std_bias': statistics.stdev(bias_values) if len(bias_values) > 1 else 0.0,
        'min_bias': min(bias_values) if bias_values else 0.0,
        'max_bias': max(bias_values) if bias_values else 0.0,
        'zero_bias_count': zero_bias_count,
        'zero_bias_percentage': (zero_bias_count / len(sample_nodes) * 100) if sample_nodes else 0
    }


def compare_directions(graph: Graph, node_id: NodeId) -> Dict:
    """
    Compare all neighbors vs cardinal neighbors for a specific node.
    """
    all_neighbors = list(graph.neighbors(node_id))
    cardinal_only = cardinal_neighbors(graph, node_id)
    
    # Identify which are diagonal
    node = graph.nodes[node_id]
    diagonal_neighbors = []
    cardinal_list = []
    
    for v, cost in all_neighbors:
        neighbor_node = graph.nodes[v]
        dx = neighbor_node.x - node.x
        dy = neighbor_node.y - node.y
        
        if (abs(dx) + abs(dy)) == 1:
            cardinal_list.append((v, cost, (dx, dy)))
        else:
            diagonal_neighbors.append((v, cost, (dx, dy)))
    
    return {
        'node_id': node_id,
        'node_coords': (node.x, node.y),
        'total_neighbors': len(all_neighbors),
        'cardinal_count': len(cardinal_only),
        'diagonal_count': len(diagonal_neighbors),
        'cardinal_neighbors': cardinal_list,
        'diagonal_neighbors': diagonal_neighbors,
        'coverage': len(cardinal_only) / len(all_neighbors) * 100 if all_neighbors else 0
    }


def print_diagnostic_report(graph: Graph, q_agent: QLearningAgent, 
                           start: NodeId, goal: NodeId):
    """
    Print a comprehensive diagnostic report.
    """
    print("=" * 80)
    print("Q*A* DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Coverage analysis
    print("\n1. NEIGHBOR COVERAGE ANALYSIS")
    print("-" * 80)
    coverage = analyze_q_coverage(graph, q_agent)
    print(f"Average neighbors per node (A* considers): {coverage['avg_all_neighbors']:.2f}")
    print(f"Average cardinal neighbors (Q-learning covers): {coverage['avg_cardinal_neighbors']:.2f}")
    print(f"Coverage percentage: {coverage['coverage_percentage']:.1f}%")
    print(f"Nodes with Q-values: {coverage['nodes_with_q_values']}/{coverage['sample_size']}")
    print(f"Total Q-values learned: {coverage['total_q_values']}")
    
    # Q-bias distribution
    print("\n2. Q-BIAS DISTRIBUTION")
    print("-" * 80)
    bias_stats = analyze_q_bias_distribution(graph, q_agent)
    print(f"Mean bias: {bias_stats['mean_bias']:.4f}")
    print(f"Median bias: {bias_stats['median_bias']:.4f}")
    print(f"Std deviation: {bias_stats['std_bias']:.4f}")
    print(f"Min bias: {bias_stats['min_bias']:.4f}")
    print(f"Max bias: {bias_stats['max_bias']:.4f}")
    print(f"Zero bias nodes: {bias_stats['zero_bias_count']} ({bias_stats['zero_bias_percentage']:.1f}%)")
    
    # Example node analysis
    print("\n3. EXAMPLE NODE ANALYSIS (Start Node)")
    print("-" * 80)
    start_analysis = compare_directions(graph, start)
    print(f"Node {start} at {start_analysis['node_coords']}")
    print(f"Total neighbors: {start_analysis['total_neighbors']}")
    print(f"Cardinal neighbors: {start_analysis['cardinal_count']}")
    print(f"Diagonal neighbors: {start_analysis['diagonal_count']}")
    
    # Show which neighbors Q-learning covers
    if q_agent.use_all_directions:
        print(f"Q-learning mode: 8-directional (covers ALL neighbors)")
        covered_neighbors = start_analysis['cardinal_neighbors'] + start_analysis['diagonal_neighbors']
        print("\nAll neighbors (Q-learning covers):")
        for v, cost, (dx, dy) in covered_neighbors:
            q_val = q_agent.get_q(start, v)
            neighbor_type = "cardinal" if (abs(dx) + abs(dy)) == 1 else "diagonal"
            print(f"  -> Node {v}: cost={cost:.2f}, direction=({dx:+.0f},{dy:+.0f}), type={neighbor_type}, Q={q_val:.4f}")
    else:
        print(f"Q-learning mode: 4-directional (covers ONLY cardinal)")
        print(f"Coverage: {start_analysis['coverage']:.1f}%")
        print("\nCardinal neighbors (Q-learning covers):")
        for v, cost, (dx, dy) in start_analysis['cardinal_neighbors']:
            q_val = q_agent.get_q(start, v)
            print(f"  -> Node {v}: cost={cost:.2f}, direction=({dx:+.0f},{dy:+.0f}), Q={q_val:.4f}")
        print("\nDiagonal neighbors (Q-learning MISSING):")
        for v, cost, (dx, dy) in start_analysis['diagonal_neighbors']:
            print(f"  -> Node {v}: cost={cost:.2f}, direction=({dx:+.0f},{dy:+.0f}), Q=NOT_LEARNED")
    
    print("\n" + "=" * 80)
    print("KEY FINDING:")
    print(f"Q-learning only covers {coverage['coverage_percentage']:.1f}% of neighbors!")
    print("This means when A* considers diagonal moves, Q-learning has NO information.")
    print("=" * 80)

