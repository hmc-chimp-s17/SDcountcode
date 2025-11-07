"""
Count boundaries (faces) of ribbon graph from 3-cycle configuration.

After merging:
- Vertices a1,a2,a3 → single vertex 'a' with 3 incident edges
- Vertices b1,b2,b3 → single vertex 'b' with 3 incident edges
- Vertices c1,c2,c3 → single vertex 'c' with 3 incident edges
- Vertices t1,t2,t3 → single vertex 't' with 3 incident edges

The rotation system specifies the cyclic order of edges at each vertex.
For example, at vertex 'a', if edges e1, e2, e3 are incident (originally at a1, a2, a3),
the rotation a1→a2→a3 means e1→e2→e3→e1 cyclically.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np
import os

# Two configurations to analyze
CONFIGS = {
    '6J': [
        ('t', 'a'),  # originally (t1, a1)
        ('t', 'b'),  # originally (t2, b2)
        ('t', 'c'),  # originally (t3, c3)
        ('a', 'b'),  # originally (a2, b1)
        ('a', 'c'),  # originally (a3, c1)
        ('b', 'c'),  # originally (b3, c2)
    ],
    'PILLOW': [
        ('t', 'a'),  # originally (t1, a1)
        ('t', 'a'),  # originally (t2, a2) - DOUBLE EDGE
        ('t', 'c'),  # originally (t3, c3)
        ('a', 'b'),  # originally (a3, b1)
        ('b', 'c'),  # originally (b2, c2)
        ('b', 'c'),  # originally (b3, c1) - DOUBLE EDGE
    ]
}

# Original port labels (before merging) for each edge
# This tells us the rotation order at each vertex
ORIGINAL_PORTS = {
    '6J': [
        (('t', 1), ('a', 1)),
        (('t', 2), ('b', 2)),
        (('t', 3), ('c', 3)),
        (('a', 2), ('b', 1)),
        (('a', 3), ('c', 1)),
        (('b', 3), ('c', 2)),
    ],
    'PILLOW': [
        (('t', 1), ('a', 1)),
        (('t', 2), ('a', 2)),
        (('t', 3), ('c', 3)),
        (('a', 3), ('b', 1)),
        (('b', 2), ('c', 2)),
        (('b', 3), ('c', 1)),
    ]
}


def build_rotation_system(original_ports):
    """
    Build rotation system at each vertex based on original port labels.
    Returns dict: vertex → list of (edge_id, port_number) in cyclic order
    """
    rotation = {'a': [], 'b': [], 'c': [], 't': []}

    for edge_id, (u, v) in enumerate(original_ports):
        # Add edge to rotation at vertex u
        vertex_u = u[0]
        port_u = u[1]
        rotation[vertex_u].append((edge_id, port_u))

        # Add edge to rotation at vertex v
        vertex_v = v[0]
        port_v = v[1]
        rotation[vertex_v].append((edge_id, port_v))

    # Sort edges at each vertex by port number (1, 2, 3)
    for vertex in rotation:
        rotation[vertex].sort(key=lambda x: x[1])

    return rotation


def build_dart_structure(edges, rotation_system):
    """
    Build dart (half-edge) structure for ribbon graph.
    Each edge becomes two darts (directed half-edges).
    Returns: darts list and dart_map dictionary
    """
    darts = []
    dart_map = {}

    for edge_id, (u, v) in enumerate(edges):
        idx = len(darts)
        # Forward dart: u → v
        dart_fwd = (u, v, edge_id)
        # Backward dart: v → u
        dart_bwd = (v, u, edge_id)

        darts.append(dart_fwd)
        darts.append(dart_bwd)

        dart_map[dart_fwd] = idx
        dart_map[dart_bwd] = idx + 1

    return darts, dart_map


def build_permutations(darts, dart_map, rotation_system):
    """
    Build α (alpha - edge reversal) and σ (sigma - vertex rotation) permutations.
    Face permutation φ = σ ∘ α.
    """
    D = len(darts)

    # α: edge flip (swap to opposite dart on same edge)
    alpha = [None] * D
    for i, (u, v, eid) in enumerate(darts):
        opposite = (v, u, eid)
        if opposite in dart_map:
            alpha[i] = dart_map[opposite]

    # σ: vertex rotation
    sigma = [None] * D
    for vertex, edges_at_v in rotation_system.items():
        n = len(edges_at_v)
        if n == 0:
            continue

        for k in range(n):
            eid_k, port_k = edges_at_v[k]
            eid_next, port_next = edges_at_v[(k + 1) % n]

            # Find outgoing dart for current edge
            for u, v, eid in darts:
                if u == vertex and eid == eid_k:
                    dart_k = (u, v, eid)
                    break

            # Find outgoing dart for next edge
            for u, v, eid in darts:
                if u == vertex and eid == eid_next:
                    dart_next = (u, v, eid)
                    break

            if dart_k in dart_map and dart_next in dart_map:
                sigma[dart_map[dart_k]] = dart_map[dart_next]

    # φ = σ ∘ α (composition: first α, then σ)
    phi = [None] * D
    for i in range(D):
        if alpha[i] is not None and sigma[alpha[i]] is not None:
            phi[i] = sigma[alpha[i]]

    return alpha, sigma, phi


def count_boundary_cycles(darts, dart_map, rotation_system):
    """
    Count boundary cycles (faces) using φ = σ ∘ α.
    """
    alpha, sigma, phi = build_permutations(darts, dart_map, rotation_system)

    D = len(darts)
    visited = [False] * D
    boundaries = []

    for i in range(D):
        if visited[i] or phi[i] is None:
            continue

        # Trace φ-cycle (boundary)
        boundary = []
        j = i
        while not visited[j] and phi[j] is not None:
            visited[j] = True
            boundary.append(darts[j])
            j = phi[j]

        if len(boundary) > 0:
            boundaries.append(boundary)

    return boundaries


def print_boundary(boundary, boundary_num, edges):
    """Print a boundary cycle."""
    print(f"\nBoundary {boundary_num}:")
    print(f"  Length: {len(boundary)} darts")

    # Extract vertex and edge sequence
    vertices = []
    edge_ids = []
    for dart in boundary:
        u, v, edge_id = dart
        vertices.append(u)
        edge_ids.append(edge_id)

    print(f"  Vertex sequence: {' → '.join(vertices)}")
    print(f"  Edge sequence: {' → '.join([f'e{i}' for i in edge_ids])}")


def draw_ribbon_diagram(name, edges, original_ports, boundaries, output_file):
    """Draw ribbon graph with boundaries highlighted."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Vertex positions (arranged in a square-like layout)
    positions = {
        't': (0, 1),
        'a': (-0.8, 0),
        'b': (0.8, 0),
        'c': (0, -0.8)
    }

    # Build rotation system
    rotation_system = build_rotation_system(original_ports)

    # Count edges between each pair of vertices
    edge_counts = {}
    for i, (u, v) in enumerate(edges):
        key = frozenset([u, v])
        if key not in edge_counts:
            edge_counts[key] = []
        edge_counts[key].append(i)

    # Calculate edge label positions (midpoints)
    edge_label_positions = {}
    for i, (u, v) in enumerate(edges):
        x1, y1 = positions[u]
        x2, y2 = positions[v]

        key = frozenset([u, v])
        edge_list = edge_counts[key]
        num_parallel = len(edge_list)
        idx_in_parallel = edge_list.index(i)

        # Calculate offset for parallel edges
        if num_parallel == 1:
            offset = 0
        else:
            offset = (idx_in_parallel - (num_parallel - 1) / 2) * 0.15

        # Calculate midpoint
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        perp_x, perp_y = -dy / length, dx / length

        mid_x = (x1 + x2) / 2 + offset * perp_x
        mid_y = (y1 + y2) / 2 + offset * perp_y

        edge_label_positions[i] = (mid_x, mid_y)

    # Draw edge labels
    for i, (mid_x, mid_y) in edge_label_positions.items():
        ax.text(mid_x, mid_y, f'e{i}', fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'),
               zorder=5)

    # Draw boundaries as colored paths
    boundary_colors = ['red', 'blue', 'green', 'orange', 'purple']

    for b_idx, boundary in enumerate(boundaries):
        color = boundary_colors[b_idx % len(boundary_colors)]

        # Extract path of darts
        for dart_idx, dart in enumerate(boundary):
            u, v, edge_id = dart
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Get edge position
            key = frozenset([u, v])
            edge_list = edge_counts[key]
            num_parallel = len(edge_list)
            idx_in_parallel = edge_list.index(edge_id)

            # Base offset for parallel edges
            if num_parallel == 1:
                parallel_offset = 0
            else:
                parallel_offset = (idx_in_parallel - (num_parallel - 1) / 2) * 0.15

            # Draw arrow along the dart direction
            if u != v:
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                perp_x, perp_y = -dy / length, dx / length

                # Determine which "side" of the edge this dart goes
                # Create a consistent ordering: forward is u<v lexicographically
                u_str = str(u)
                v_str = str(v)
                is_forward = u_str < v_str

                # Offset perpendicular to edge: forward on one side, backward on other
                # Small offset to separate arrows slightly
                dart_offset = 0.04 if is_forward else -0.04

                # Start and end points with perpendicular offset
                start_x = x1 + 0.12 * dx / length + (parallel_offset + dart_offset) * perp_x
                start_y = y1 + 0.12 * dy / length + (parallel_offset + dart_offset) * perp_y
                end_x = x2 - 0.12 * dx / length + (parallel_offset + dart_offset) * perp_x
                end_y = y2 - 0.12 * dy / length + (parallel_offset + dart_offset) * perp_y

                # Control point for curve
                mid_x = (x1 + x2) / 2 + (parallel_offset + dart_offset) * perp_x
                mid_y = (y1 + y2) / 2 + (parallel_offset + dart_offset) * perp_y

                # Draw arrow with curved path
                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    arrowstyle='->', mutation_scale=15, linewidth=2.5,
                    color=color, alpha=0.7, zorder=3,
                    connectionstyle=f"arc3,rad={parallel_offset * 0.3}"
                )
                ax.add_patch(arrow)

    # Draw vertices
    for vertex, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.12, color='lightblue', ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, vertex, fontsize=14, ha='center', va='center',
               fontweight='bold', zorder=11)

        # Show rotation order
        rot_info = rotation_system[vertex]
        rot_text = ' → '.join([f'e{eid}' for eid, port in rot_info])
        ax.text(x, y - 0.25, rot_text, fontsize=8, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7),
               zorder=4)

    # Create legend for boundaries
    legend_elements = []
    for b_idx, boundary in enumerate(boundaries):
        color = boundary_colors[b_idx % len(boundary_colors)]
        vertices = [dart[0] for dart in boundary]
        vertex_seq = ' → '.join(vertices)
        label = f"Boundary {b_idx + 1} ({len(boundary)} darts): {vertex_seq}"
        legend_elements.append(mpatches.Patch(facecolor=color, alpha=0.6, label=label))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             bbox_to_anchor=(0.02, 0.98))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Ribbon Graph: {name}\n(Boundaries highlighted with colored arrows)",
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved diagram to {output_file}")
    plt.close()


def analyze_configuration(name, edges, original_ports, draw_diagram=False, output_dir=None):
    """Analyze one configuration."""
    print("\n" + "="*60)
    print(f"CONFIGURATION: {name}")
    print("="*60)

    print("\nEdges (after merging):")
    for i, (u, v) in enumerate(edges):
        orig_u, orig_v = original_ports[i]
        print(f"  Edge {i}: {u} -- {v}  (originally {orig_u[0]}{orig_u[1]} -- {orig_v[0]}{orig_v[1]})")

    print("\n" + "="*60)
    print("Building rotation system...")
    print("="*60)

    rotation_system = build_rotation_system(original_ports)

    print("\nRotation at each vertex:")
    for vertex in ['a', 'b', 'c', 't']:
        edges_info = []
        for edge_id, port in rotation_system[vertex]:
            orig_u, orig_v = original_ports[edge_id]
            edges_info.append(f"e{edge_id}(port {vertex}{port})")
        print(f"  Vertex '{vertex}': {' → '.join(edges_info)}")

    print("\n" + "="*60)
    print("Building dart structure...")
    print("="*60)

    darts, dart_map = build_dart_structure(edges, rotation_system)
    print(f"Total darts: {len(darts)}")

    print("\n" + "="*60)
    print("Counting boundary cycles...")
    print("="*60)

    boundaries = count_boundary_cycles(darts, dart_map, rotation_system)

    print(f"\nNumber of boundaries (faces): F = {len(boundaries)}")

    for i, boundary in enumerate(boundaries, 1):
        print_boundary(boundary, i, edges)

    print("\n" + "="*60)
    print("EULER CHARACTERISTIC")
    print("="*60)

    V = 4  # Merged vertices: a, b, c, t
    E = len(edges)
    F = len(boundaries)

    chi = V - E + F

    print(f"\nV (vertices): {V}")
    print(f"E (edges): {E}")
    print(f"F (faces/boundaries): {F}")
    print(f"\nχ = V - E + F = {V} - {E} + {F} = {chi}")

    # Genus formula
    genus = (2 - chi) / 2
    print(f"\nGenus g = (2 - χ) / 2 = (2 - {chi}) / 2 = {genus}")

    # Draw diagram if requested
    if draw_diagram and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{name}_ribbon_diagram.png")
        draw_ribbon_diagram(name, edges, original_ports, boundaries, output_file)

    return V, E, F, chi, genus, boundaries


def main():
    print("="*60)
    print("3-Cycle Ribbon Graph Boundary Counter")
    print("="*60)

    results = {}
    output_dir = "ribbon_diagrams"

    for config_name in CONFIGS:
        edges = CONFIGS[config_name]
        original_ports = ORIGINAL_PORTS[config_name]
        V, E, F, chi, genus, boundaries = analyze_configuration(
            config_name, edges, original_ports,
            draw_diagram=True, output_dir=output_dir
        )
        results[config_name] = (V, E, F, chi, genus)

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    print("\n{:<15} {:<5} {:<5} {:<5} {:<8} {:<8}".format("Config", "V", "E", "F", "χ", "Genus"))
    print("-" * 60)
    for config_name, (V, E, F, chi, genus) in results.items():
        print("{:<15} {:<5} {:<5} {:<5} {:<8} {:<8}".format(
            config_name, V, E, F, chi, genus))

if __name__ == "__main__":
    main()
