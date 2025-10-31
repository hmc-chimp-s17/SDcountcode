from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Define all ports ----
# Each vertex has 3 ports
S = [("s", i) for i in (1, 2, 3)]  # Input ports
T = [("t", i) for i in (1, 2, 3)]  # Output ports
A = [("a", i) for i in (1, 2, 3)]  # Intermediate ports
B = [("b", i) for i in (1, 2, 3)]  # Intermediate ports
C = [("c", i) for i in (1, 2, 3)]  # Intermediate ports

# All ports that can be on the RHS (everything except S)
RHS_PORTS = T + A + B + C  # 12 ports total

# ---- Fixed LHS edges ----
FIXED = [
    (("s", 1), ("a", 1)),
    (("s", 2), ("b", 2)),
    (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)),
    (("b", 3), ("c", 2)),
    (("c", 1), ("a", 3)),
]


# ---- Helper function: build adjacency list ----
def build_adjacency(edges):
    """Build adjacency list from list of edges."""
    adj = defaultdict(set)
    for edge in edges:
        if isinstance(edge, frozenset):
            u, v = tuple(edge)
        else:
            u, v = edge
        adj[u].add(v)
        adj[v].add(u)
    return adj


# ---- Check port degrees (valency) ----
def check_port_degrees(edges):
    """
    Check that each port has the correct degree:
    - S ports (input): exactly 1 edge
    - T ports (output): exactly 1 edge
    - A, B, C ports (intermediate): exactly 2 edges
    """
    adj = build_adjacency(edges)

    # Check S ports: degree 1
    for port in S:
        if len(adj[port]) != 1:
            return False

    # Check T ports: degree 1
    for port in T:
        if len(adj[port]) != 1:
            return False

    # Check A, B, C ports: degree 2
    for port in A + B + C:
        if len(adj[port]) != 2:
            return False

    return True


# ---- Find a cycle in the graph ----
def find_cycle(edges):
    """Find any cycle in the graph. Returns list of edges in the cycle, or empty list."""
    adj = build_adjacency(edges)
    visited = set()
    parent = {}

    def dfs(node, par):
        """DFS to find cycle. Returns cycle edges if found, else None."""
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor == par:  # Skip edge back to parent
                continue
            if neighbor in visited:
                # Found a cycle! Reconstruct it
                cycle = []
                current = node
                while current != neighbor:
                    cycle.append((current, parent[current]))
                    current = parent[current]
                cycle.append((neighbor, node))
                return cycle
            else:
                parent[neighbor] = node
                result = dfs(neighbor, node)
                if result:
                    return result
        return None

    # Try starting from each unvisited node
    for start_node in adj:
        if start_node not in visited:
            parent[start_node] = None
            cycle = dfs(start_node, None)
            if cycle:
                return cycle

    return []


# ---- Remove all cycles ----
def remove_all_cycles(edges):
    """Remove all cycles by repeatedly finding and deleting one cycle."""
    # Convert all edges to frozensets for consistency
    edge_list = [frozenset(e) if not isinstance(e, frozenset) else e for e in edges]

    while True:
        cycle = find_cycle(edge_list)
        if not cycle:
            break  # No more cycles

        # Remove the edges in this cycle
        cycle_set = {frozenset(e) for e in cycle}
        edge_list = [e for e in edge_list if e not in cycle_set]

    return edge_list


# ---- Check if graph forms exactly 3 disconnected paths from S to T ----
def check_three_paths(edges):
    """
    Check if the graph forms exactly 3 DISCONNECTED paths from S to T.
    Requirements:
    1. Exactly 3 connected components
    2. Each component is a path from one S port to one T port
    3. Paths must connect s_i to t_i (same index i) - no permutation
    4. Each component is disconnected from the others
    """
    adj = build_adjacency(edges)

    # Find all connected components
    visited = set()
    components = []

    def bfs_component(start):
        """BFS to find all nodes in the connected component containing start."""
        component = set()
        queue = [start]
        component.add(start)
        visited.add(start)

        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        return component

    # Find all components starting from S ports
    for s_port in S:
        if s_port not in visited:
            component = bfs_component(s_port)
            components.append(component)

    # Must have exactly 3 components (one per S port)
    if len(components) != 3:
        return False

    # Check each component and verify correct S-T pairing
    for component in components:
        # Find which S and T ports are in this component
        s_in_component = [port for port in component if port in S]
        t_in_component = [port for port in component if port in T]

        # Each component must have exactly 1 S and 1 T
        if len(s_in_component) != 1 or len(t_in_component) != 1:
            return False

        # Get the specific S and T ports
        s_port = s_in_component[0]
        t_port = t_in_component[0]

        # CRITICAL: Check that s_i connects to t_i (same port number)
        # s_port = ("s", i), t_port = ("t", j)
        # We require i == j
        s_index = s_port[1]  # port number (1, 2, or 3)
        t_index = t_port[1]  # port number (1, 2, or 3)

        if s_index != t_index:
            return False  # Not the correct pairing!

        # Check that this component forms a simple path (no branches)
        # In a path, all nodes have degree ≤ 2, and exactly 2 endpoints have degree 1
        degrees = []
        for node in component:
            degree = len([n for n in adj[node] if n in component])
            degrees.append(degree)

        # Count degree-1 nodes (endpoints)
        endpoints = sum(1 for d in degrees if d == 1)

        # All nodes should have degree 1 or 2
        if not all(d in [1, 2] for d in degrees):
            return False

        # Should have exactly 2 endpoints (the path ends)
        if endpoints != 2:
            return False

    return True


# ---- Generate all valid RHS configurations ----
def generate_all_rhs_matchings():
    """
    Generate all ways to pair up the 12 RHS ports into 6 edges.
    Constraints:
    - Each port used exactly once (perfect matching)
    - No t-t connections (no two T ports connected)
    """
    all_matchings = []

    def build_matching(remaining_ports, current_matching):
        """Recursively build perfect matchings."""
        if len(current_matching) == 6:
            # We have 6 edges, check if all ports are used
            if len(remaining_ports) == 0:
                all_matchings.append(list(current_matching))
            return

        if len(remaining_ports) < 2:
            return

        # Take the first remaining port and try pairing it with others
        first = remaining_ports[0]
        for i in range(1, len(remaining_ports)):
            second = remaining_ports[i]

            # Constraint: no t-t connection
            if first in T and second in T:
                continue

            # Create edge and recurse
            edge = (first, second)
            new_remaining = [p for j, p in enumerate(remaining_ports) if j != 0 and j != i]
            build_matching(new_remaining, current_matching + [edge])

    build_matching(RHS_PORTS, [])
    return all_matchings


# ---- Visualization ----
def draw_graph(edges, title="Graph", filename=None):
    """Draw the graph with ports arranged vertically with horizontal offset to avoid overlaps."""
    fig, ax = plt.subplots(figsize=(12, 16))

    # Port positions
    positions = {}

    # S ports - left column (vertically aligned)
    for i, port in enumerate([("s", 1), ("s", 2), ("s", 3)]):
        positions[port] = (0, 8 - i * 3.5)

    # A, B, C ports - middle, each arranged as a triangle
    # Triangles are vertically stacked: A at top, B in middle, C at bottom
    # Triangle orientation: port 1 at top, ports 2 and 3 at bottom corners

    # A ports - triangle at top (centered at y=8)
    triangle_size = 0.6  # radius of triangle
    a_center_y = 8
    positions[("a", 1)] = (4.5, a_center_y + triangle_size)      # top
    positions[("a", 2)] = (4.5 - triangle_size, a_center_y - triangle_size * 0.5)  # bottom left
    positions[("a", 3)] = (4.5 + triangle_size, a_center_y - triangle_size * 0.5)  # bottom right

    # B ports - triangle in middle (centered at y=4.5)
    b_center_y = 4.5
    positions[("b", 1)] = (4.5, b_center_y + triangle_size)      # top
    positions[("b", 2)] = (4.5 - triangle_size, b_center_y - triangle_size * 0.5)  # bottom left
    positions[("b", 3)] = (4.5 + triangle_size, b_center_y - triangle_size * 0.5)  # bottom right

    # C ports - triangle at bottom (centered at y=1)
    c_center_y = 1
    positions[("c", 1)] = (4.5, c_center_y + triangle_size)      # top
    positions[("c", 2)] = (4.5 - triangle_size, c_center_y - triangle_size * 0.5)  # bottom left
    positions[("c", 3)] = (4.5 + triangle_size, c_center_y - triangle_size * 0.5)  # bottom right

    # T ports - right column (vertically aligned)
    for i, port in enumerate([("t", 1), ("t", 2), ("t", 3)]):
        positions[port] = (9, 8 - i * 3.5)

    # Colors
    colors = {
        "s": "lightblue",
        "a": "lightgreen",
        "b": "lightyellow",
        "c": "lightcoral",
        "t": "plum"
    }

    # Draw ports
    for port, (x, y) in positions.items():
        vertex, num = port
        color = colors[vertex]
        circle = plt.Circle((x, y), 0.2, color=color, ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f"{vertex}{num}", ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=4)

    # Draw edges
    edge_set = {frozenset(e) if not isinstance(e, frozenset) else e for e in edges}

    # Separate fixed edges from others
    fixed_edges = {frozenset(e) for e in FIXED}

    for edge in edge_set:
        u, v = tuple(edge)
        x1, y1 = positions[u]
        x2, y2 = positions[v]

        # Use blue for fixed LHS edges, black for RHS edges
        if edge in fixed_edges:
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2.5, zorder=1, alpha=0.8)
        else:
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

    # Draw vertex labels (centered on each triangle)
    ax.text(-0.7, 4.5, "S", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(4.5, a_center_y, "A", ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.text(4.5, b_center_y, "B", ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.text(4.5, c_center_y, "C", ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.text(9.7, 4.5, "T", ha='center', va='center', fontsize=16, fontweight='bold')

    # Set plot properties
    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-0.5, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=colors[v], edgecolor='black', label=f'Vertex {v.upper()}')
        for v in ['s', 'a', 'b', 'c', 't']
    ]
    # Add edge color legend
    legend_elements.extend([
        Line2D([0], [0], color='blue', linewidth=2.5, label='Fixed LHS edges'),
        Line2D([0], [0], color='black', linewidth=2, label='RHS edges')
    ])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    else:
        plt.show()

    plt.close()


# ---- Main counting function ----
def count_valid_configurations():
    """Count all valid configurations."""
    print("Generating all RHS matchings...")
    all_rhs = generate_all_rhs_matchings()
    print(f"Total RHS matchings to check: {len(all_rhs)}")

    valid_count = 0
    valid_examples = []
    examples_with_cycles = []  # Track examples where cycles were removed

    for idx, rhs_edges in enumerate(all_rhs):
        if (idx + 1) % 1000 == 0:
            print(f"Checked {idx + 1}/{len(all_rhs)}...")

        # Combine LHS + RHS
        combined = FIXED + rhs_edges

        # Check port degrees
        if not check_port_degrees(combined):
            continue

        # Remove cycles
        after_removal = remove_all_cycles(combined)

        # Check if it forms 3 paths
        if check_three_paths(after_removal):
            valid_count += 1

            # Prioritize examples where cycles were actually removed
            had_cycle = len(combined) != len(after_removal)

            if had_cycle and len(examples_with_cycles) < 5:
                examples_with_cycles.append((rhs_edges, combined, after_removal))
            elif len(valid_examples) < 5:
                valid_examples.append((rhs_edges, combined, after_removal))

    # Prefer examples with cycles, fall back to examples without cycles
    if examples_with_cycles:
        return valid_count, examples_with_cycles
    return valid_count, valid_examples


# ---- Main execution ----
if __name__ == "__main__":
    import sys

    draw_mode = "--draw" in sys.argv

    print("="*60)
    print("6j Symbol Configuration Counter")
    print("="*60)
    print(f"Fixed LHS edges: {len(FIXED)}")
    print(f"RHS ports: {len(RHS_PORTS)}")
    print()

    # Draw the fixed LHS
    if draw_mode:
        print("Drawing LHS configuration...")
        draw_graph(FIXED, title="Fixed LHS Configuration", filename="lhs_fixed.png")
        print()

    # Count valid configurations
    count, examples = count_valid_configurations()

    print()
    print("="*60)
    print(f"TOTAL VALID CONFIGURATIONS: {count}")
    print("="*60)

    # Show examples
    if examples:
        print(f"\nFirst {len(examples)} valid examples:\n")
        for i, (rhs, combined, final) in enumerate(examples, 1):
            print(f"Example {i}:")
            print(f"  RHS edges: {len(rhs)}")
            for edge in rhs:
                print(f"    {edge}")
            print(f"  Combined: {len(combined)} edges")
            print(f"  After cycle removal: {len(final)} edges")
            print()

            if draw_mode:
                draw_graph(combined, f"Example {i}: Initial Combined (LHS + RHS)", f"ex{i}_initial.png")
                draw_graph(final, f"Example {i}: Final (After Cycle Removal)", f"ex{i}_final.png")

    if draw_mode:
        print("\nAll diagrams saved!")
    else:
        print("\nTo generate diagrams, run: python3 SDcount.py --draw")
