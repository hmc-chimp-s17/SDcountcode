from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define ports
S = [("s", i) for i in (1, 2, 3)]
T = [("t", i) for i in (1, 2, 3)]
A = [("a", i) for i in (1, 2, 3)]
B = [("b", i) for i in (1, 2, 3)]
C = [("c", i) for i in (1, 2, 3)]

RHS_PORTS = T + A + B + C

# Fixed LHS edges
FIXED = [
    (("s", 1), ("a", 1)),
    (("s", 2), ("b", 2)),
    (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)),
    (("b", 3), ("c", 2)),
    (("c", 1), ("a", 3)),
]


def build_adjacency(edges):
    """Build adjacency list (ignores parallel edges)."""
    adj = defaultdict(set)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        adj[u].add(v)
        adj[v].add(u)
    return adj


def check_port_degrees(edges):
    """Check S/T have degree 1, A/B/C have degree 2 (counting parallel edges)."""
    adj = defaultdict(list)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        adj[u].append(v)
        adj[v].append(u)

    for port in S + T:
        if len(adj[port]) != 1:
            return False
    for port in A + B + C:
        if len(adj[port]) != 2:
            return False
    return True


def find_cycle(edges):
    """Find any cycle, including 2-cycles from parallel edges."""
    # Check for parallel edges first
    edge_counts = defaultdict(int)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        edge_counts[frozenset([u, v])] += 1

    for key, count in edge_counts.items():
        if count >= 2:
            u, v = tuple(key)
            return [(u, v), (v, u)]

    # Find regular cycles using DFS
    adj = build_adjacency(edges)
    visited, parent = set(), {}

    def dfs(node, par):
        visited.add(node)
        for nb in adj[node]:
            if nb == par:
                continue
            if nb in visited:
                cyc = []
                cur = node
                while cur != nb:
                    cyc.append((cur, parent[cur]))
                    cur = parent[cur]
                cyc.append((nb, node))
                return cyc
            parent[nb] = node
            res = dfs(nb, node)
            if res:
                return res
        return None

    for s in adj:
        if s not in visited:
            parent[s] = None
            cyc = dfs(s, None)
            if cyc:
                return cyc
    return []


def remove_all_cycles(edges):
    """Remove all cycles. Returns (remaining_edges, num_cycles_removed)."""
    E = [frozenset(e) if not isinstance(e, frozenset) else e for e in edges]
    num_cycles = 0

    while True:
        cyc = find_cycle(E)
        if not cyc:
            break
        E = [e for e in E if frozenset(e) not in {frozenset(c) for c in cyc}]
        num_cycles += 1

    return E, num_cycles


def check_three_paths(edges):
    """Check for 3 disjoint paths s_i → t_i."""
    adj = build_adjacency(edges)
    visited = set()

    def bfs(start):
        q, comp = [start], {start}
        visited.add(start)
        while q:
            x = q.pop(0)
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    comp.add(y)
                    q.append(y)
        return comp

    comps = [bfs(sp) for sp in S if sp not in visited]
    if len(comps) != 3:
        return False

    for comp in comps:
        s_list = [p for p in comp if p in S]
        t_list = [p for p in comp if p in T]
        if len(s_list) != 1 or len(t_list) != 1 or s_list[0][1] != t_list[0][1]:
            return False
        degs = [len([n for n in adj[v] if n in comp]) for v in comp]
        if not all(d in (1, 2) for d in degs) or sum(1 for d in degs if d == 1) != 2:
            return False
    return True


def generate_all_rhs_matchings():
    """Generate all perfect matchings on RHS_PORTS (no T-T edges)."""
    all_matchings = []

    def rec(remaining, cur):
        if len(cur) == 6:
            if not remaining:
                all_matchings.append(list(cur))
            return
        if len(remaining) < 2:
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            second = remaining[i]
            if first in T and second in T:
                continue
            rec([p for j, p in enumerate(remaining) if j not in (0, i)], cur + [(first, second)])

    rec(RHS_PORTS, [])
    return all_matchings


def get_all_cycles(edges):
    """Get all cycles that will be removed. Returns list of cycles."""
    E = [frozenset(e) if not isinstance(e, frozenset) else e for e in edges]
    all_cycles = []

    while True:
        cyc = find_cycle(E)
        if not cyc:
            break
        all_cycles.append(cyc)
        E = [e for e in E if frozenset(e) not in {frozenset(c) for c in cyc}]

    return all_cycles


def draw_graph_with_cycles(edges, cycles, title, filename):
    """Draw graph with cycles highlighted."""
    positions = {
        ('s', 1): (-0.3, 5.3), ('s', 2): (0, 5), ('s', 3): (-0.3, 4.7),
        ('t', 1): (10.3, 5.3), ('t', 2): (10, 5), ('t', 3): (10.3, 4.7),
        ('a', 1): (4, 8), ('a', 2): (3.5, 7), ('a', 3): (4.5, 7),
        ('b', 1): (5, 5), ('b', 2): (4.5, 4), ('b', 3): (5.5, 4),
        ('c', 1): (4, 2), ('c', 2): (3.5, 1), ('c', 3): (4.5, 1),
    }

    vertex_colors = {
        **{p: 'lightblue' for p in S},
        **{p: 'plum' for p in T},
        **{p: 'lightgreen' for p in A},
        **{p: 'lightyellow' for p in B},
        **{p: 'lightcoral' for p in C},
    }

    # Identify cycle edges
    cycle_edges = []
    for i, cyc in enumerate(cycles):
        for edge in cyc:
            cycle_edges.append((frozenset(edge), i))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw vertices
    for v, (x, y) in positions.items():
        r = 0.15
        ax.add_patch(plt.Circle((x, y), r, color=vertex_colors[v], ec='black', lw=2, zorder=3))
        label = f"{v[0]}{v[1]}"
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    # Count edge multiplicities
    edge_counts = defaultdict(int)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        key = frozenset([u, v])
        edge_counts[key] += 1

    # Cycle colors
    cycle_colors = ['red', 'orange', 'purple']

    # Draw edges
    edge_drawn = defaultdict(int)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        if u not in positions or v not in positions:
            continue

        x1, y1 = positions[u]
        x2, y2 = positions[v]

        key = frozenset([u, v])
        total_parallel = edge_counts[key]
        current_idx = edge_drawn[key]
        edge_drawn[key] += 1

        # Check if this edge is in a cycle
        cycle_idx = None
        for ck, cidx in cycle_edges:
            if ck == key:
                cycle_idx = cidx
                break

        if cycle_idx is not None:
            color = cycle_colors[cycle_idx % len(cycle_colors)]
            lw = 3
            alpha = 0.9
        else:
            color = 'gray'
            lw = 2
            alpha = 0.5

        if total_parallel > 1:
            offset = (current_idx - (total_parallel - 1) / 2) * 0.3
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            norm = np.sqrt(dx**2 + dy**2)
            perp_x, perp_y = -dy / norm, dx / norm

            ctrl_x = mid_x + perp_x * offset
            ctrl_y = mid_y + perp_y * offset

            t = np.linspace(0, 1, 50)
            bx = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
            by = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
            ax.plot(bx, by, color=color, lw=lw, alpha=alpha, zorder=1)

            if cycle_idx is not None:
                ax.text(ctrl_x, ctrl_y, f"{u[0]}{u[1]}-{v[0]}{v[1]}", fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='pink', alpha=0.9),
                       ha='center', va='center', zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            if cycle_idx is not None:
                ax.text(mid_x, mid_y, f"{u[0]}{u[1]}-{v[0]}{v[1]}", fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='pink', alpha=0.9),
                       ha='center', va='center', zorder=2)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add cycle information
    cycle_text = ""
    for i, cyc in enumerate(cycles):
        if len(cyc) == 2:
            u, v = cyc[0]
            cycle_text += f"Cycle {i+1} (2-cycle): {u[0]}{u[1]} ⇄ {v[0]}{v[1]}\n"
        else:
            vertices = [u for u, _ in cyc]
            vertices.append(vertices[0])
            cycle_text += f"Cycle {i+1} ({len(cyc)}-cycle): {' → '.join([f'{v[0]}{v[1]}' for v in vertices])}\n"

    ax.text(0.02, 0.98, cycle_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def check_t_connections(rhs):
    """
    Check if all t1, t2, t3 connect to same intermediate vertex type.
    Returns: 'a' if all connect to a1/a2/a3, 'b' if all to b1/b2/b3,
             'c' if all to c1/c2/c3, or None otherwise.
    """
    t_connections = {}
    for e in rhs:
        u, v = e
        # Check if one end is a t vertex
        if u[0] == 't':
            t_vertex = u
            other_vertex = v
        elif v[0] == 't':
            t_vertex = v
            other_vertex = u
        else:
            continue

        # Record what intermediate vertex type this t connects to
        if other_vertex[0] in ['a', 'b', 'c']:
            t_connections[t_vertex] = other_vertex[0]

    # Check if we have all three t vertices
    if len(t_connections) != 3:
        return None

    # Check if all connect to the same type
    connection_types = set(t_connections.values())
    if len(connection_types) == 1:
        return connection_types.pop()

    return None


def count_valid_configurations():
    """Count all valid configurations."""
    print("Generating all RHS matchings...")
    all_rhs = generate_all_rhs_matchings()
    print(f"Total RHS matchings: {len(all_rhs)}")

    cycle_stats = defaultdict(list)
    t_connection_stats = {'a': [], 'b': [], 'c': []}

    for idx, rhs in enumerate(all_rhs):
        if (idx + 1) % 1000 == 0:
            print(f"  Checked {idx + 1}/{len(all_rhs)}...")

        combined = FIXED + rhs

        if not check_port_degrees(combined):
            continue

        after, num_cycles = remove_all_cycles(combined)

        if check_three_paths(after):
            all_cycles = get_all_cycles(combined)
            cycle_stats[num_cycles].append((rhs, combined, after, all_cycles))

            # Check t connections
            t_type = check_t_connections(rhs)
            if t_type:
                t_connection_stats[t_type].append((rhs, combined, after, all_cycles))

    return cycle_stats, t_connection_stats


if __name__ == "__main__":
    print("="*60)
    print("6j Symbol Configuration Counter")
    print("="*60)

    cycle_stats, t_connection_stats = count_valid_configurations()

    total = sum(len(v) for v in cycle_stats.values())

    print()
    print("="*60)
    print(f"TOTAL VALID CONFIGURATIONS: {total}")
    print("="*60)

    print("\nCycle removal statistics:")
    for num in sorted(cycle_stats.keys()):
        print(f"  {num} cycle(s) removed: {len(cycle_stats[num])} configurations")

    print("\nT-connection statistics (all t vertices connect to same type):")
    for vertex_type in ['a', 'b', 'c']:
        count = len(t_connection_stats[vertex_type])
        print(f"  All t's → {vertex_type}: {count} configurations")
    total_same_type = sum(len(v) for v in t_connection_stats.values())
    print(f"  Total with same-type connections: {total_same_type}")

    # Draw examples
    print("\nDrawing examples...")

    import os
    output_dir = "cycle_examples"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating output directory: {output_dir}/")

    # Draw ALL examples with 2 cycles
    if 2 in cycle_stats:
        print(f"Drawing all {len(cycle_stats[2])} examples with 2 cycles removed:")
        for i, (rhs, combined, final, all_cycles) in enumerate(cycle_stats[2]):
            title = f"2 Cycles Example {i+1}/{len(cycle_stats[2])}"
            filename = os.path.join(output_dir, f"2cycles_ex{i+1}.png")
            draw_graph_with_cycles(combined, all_cycles, title, filename)

    # Draw ALL examples with 3 cycles
    if 3 in cycle_stats:
        print(f"Drawing all {len(cycle_stats[3])} example(s) with 3 cycles removed:")
        for i, (rhs, combined, final, all_cycles) in enumerate(cycle_stats[3]):
            title = f"3 Cycles Example {i+1}/{len(cycle_stats[3])}"
            filename = os.path.join(output_dir, f"3cycles_ex{i+1}.png")
            draw_graph_with_cycles(combined, all_cycles, title, filename)

    # Draw examples of t-connections to same type
    t_dir = "t_connection_examples"
    os.makedirs(t_dir, exist_ok=True)
    print(f"\nCreating directory for t-connection examples: {t_dir}/")

    for vertex_type in ['a', 'b', 'c']:
        configs = t_connection_stats[vertex_type]
        if configs:
            print(f"Drawing all {len(configs)} examples where all t's → {vertex_type}:")
            for i, (rhs, combined, final, all_cycles) in enumerate(configs):
                title = f"All t's → {vertex_type} (Example {i+1}/{len(configs)})"
                filename = os.path.join(t_dir, f"t_to_{vertex_type}_ex{i+1}.png")
                draw_graph_with_cycles(combined, all_cycles, title, filename)

    print(f"\nAll drawings saved to {output_dir}/ and {t_dir}/")

    # Show text examples
    print("\nExamples (prioritized by cycle count):")
    count = 0
    for num in sorted(cycle_stats.keys(), reverse=True):
        for rhs, combined, final, all_cycles in cycle_stats[num][:3]:
            count += 1
            print(f"\nExample {count} ({num} cycle(s) removed):")
            print(f"  RHS edges:")
            for e in rhs:
                print(f"    {e}")
            print(f"  Total edges: {len(combined)} → {len(final)} after removal")

            if count >= 5:
                break
        if count >= 5:
            break
