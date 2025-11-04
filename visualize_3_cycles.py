from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define ports
S = [("s", i) for i in (1, 2, 3)]
T = [("t", i) for i in (1, 2, 3)]
A = [("a", i) for i in (1, 2, 3)]
B = [("b", i) for i in (1, 2, 3)]
C = [("c", i) for i in (1, 2, 3)]

# The special example with 3 parallel edges
FIXED = [
    (("s", 1), ("a", 1)),
    (("s", 2), ("b", 2)),
    (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)),  # PARALLEL 1
    (("b", 3), ("c", 2)),  # PARALLEL 2
    (("c", 1), ("a", 3)),  # PARALLEL 3
]

RHS = [
    (("t", 1), ("a", 1)),
    (("t", 2), ("b", 2)),
    (("t", 3), ("c", 3)),
    (("a", 2), ("b", 1)),  # PARALLEL 1
    (("a", 3), ("c", 1)),  # PARALLEL 3
    (("b", 3), ("c", 2)),  # PARALLEL 2
]


def build_adjacency(edges):
    """Build adjacency list (ignores multiplicity for cycle/path checking)."""
    adj = defaultdict(set)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_cycle(edges):
    """Find any cycle, including 2-cycles from parallel edges."""
    # First check for parallel edges (which form 2-cycles)
    edge_counts = defaultdict(list)
    for i, edge in enumerate(edges):
        if isinstance(edge, frozenset):
            u, v = tuple(edge)
        else:
            u, v = edge
        key = frozenset([u, v])
        edge_counts[key].append(i)

    # If any edge appears more than once, return it as a 2-cycle
    for key, indices in edge_counts.items():
        if len(indices) >= 2:
            u, v = tuple(key)
            return [(u, v), (v, u)]

    # Otherwise, find cycles using DFS
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
    """Repeatedly remove cycles and track them."""
    E = [frozenset(e) if not isinstance(e, frozenset) else e for e in edges]
    removed_cycles = []

    while True:
        cyc = find_cycle(E)
        if not cyc:
            break
        removed_cycles.append(cyc)
        E = [e for e in E if frozenset(e) not in {frozenset(c) for c in cyc}]

    return E, removed_cycles


def draw_graph_with_cycles(before_edges, after_edges, removed_cycles):
    """Draw before and after cycle removal, highlighting parallel edges."""

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

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Identify parallel edges (2-cycles)
    parallel_edges = set()
    for cyc in removed_cycles:
        if len(cyc) == 2:  # 2-cycle = parallel edge
            parallel_edges.add(frozenset([cyc[0][0], cyc[0][1]]))

    for ax_idx, (edges, title) in enumerate([
        (before_edges, f"BEFORE Removing 3 Cycles ({len(before_edges)} edges)"),
        (after_edges, f"AFTER Removing 3 Cycles ({len(after_edges)} edges)")
    ]):
        ax = axes[ax_idx]

        # Draw vertices
        for v, (x, y) in positions.items():
            r = 0.15
            ax.add_patch(plt.Circle((x, y), r, color=vertex_colors[v], ec='black', lw=2, zorder=3))
            label = f"{v[0]}{v[1]}"
            ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

        # Count edge multiplicities for offset
        edge_counts = defaultdict(int)
        for e in edges:
            u, v = tuple(e) if isinstance(e, frozenset) else e
            if u in positions and v in positions:
                key = frozenset([u, v])
                edge_counts[key] += 1

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

            # Check if this is a parallel edge
            is_parallel = (key in parallel_edges)

            if ax_idx == 0:  # Before
                if is_parallel:
                    color = 'red'
                    lw = 3
                    alpha = 0.9
                else:
                    color = 'black'
                    lw = 2
                    alpha = 0.6
            else:  # After
                color = 'darkgreen'
                lw = 2.5
                alpha = 0.7

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

                if ax_idx == 0 and is_parallel:
                    label_text = f"{u[0]}{u[1]}-{v[0]}{v[1]}"
                    ax.text(ctrl_x, ctrl_y, label_text, fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='pink', alpha=0.9),
                           ha='center', va='center', zorder=2)
            else:
                ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)

        ax.set_xlim(-1.5, 11.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # Add legend
        if ax_idx == 0:
            legend_text = "Red = Parallel edges\n(Each forms a 2-cycle)\nBlack = Other edges"
        else:
            legend_text = "Green = Remaining edges\n(After removing all\nparallel edges)"

        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add cycle information
    cycle_text = f"3 Parallel Edges Removed (3 × 2-cycles)\n\n"
    for i, cyc in enumerate(removed_cycles, 1):
        u, v = cyc[0]
        cycle_text += f"Cycle {i}: {u[0]}{u[1]} ⇄ {v[0]}{v[1]}\n"

    fig.text(0.5, 0.02, cycle_text, ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    plt.suptitle("The Unique 3-Cycle Configuration (3 Parallel Edges)",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.savefig("3_cycles_example.png", dpi=150, bbox_inches='tight')
    print("Saved: 3_cycles_example.png")
    plt.close()


# Process the example
combined = FIXED + RHS
after, removed = remove_all_cycles(combined)

print("="*60)
print("THE 3-CYCLE EXAMPLE (3 PARALLEL EDGES)")
print("="*60)

print(f"\nBefore: {len(combined)} edges")
print(f"After: {len(after)} edges")
print(f"Cycles removed: {len(removed)}")

print("\nFIXED edges:")
for i, e in enumerate(FIXED, 1):
    print(f"  {i}. {e}")

print("\nRHS edges:")
for i, e in enumerate(RHS, 1):
    print(f"  {i}. {e}")

# Identify parallel edges
edge_counts = defaultdict(int)
for e in combined:
    key = frozenset(e)
    edge_counts[key] += 1

print("\nParallel edges (appear in both FIXED and RHS):")
for key, count in edge_counts.items():
    if count > 1:
        u, v = tuple(key)
        print(f"  {u} ⇄ {v}: appears {count} times")

print("\nCycles removed (all are 2-cycles from parallel edges):")
for i, cyc in enumerate(removed, 1):
    u, v = cyc[0]
    print(f"  Cycle {i}: {u[0]}{u[1]} ⇄ {v[0]}{v[1]}")

print("\nAfter removing all parallel edges:")
for i, e in enumerate(after, 1):
    u, v = tuple(e) if isinstance(e, frozenset) else e
    print(f"  {i}. {u} - {v}")

print("\nThis configuration is special because:")
print("  - RHS contains 3 edges that exactly match FIXED edges")
print("  - These 3 overlaps create 3 parallel edges")
print("  - Each parallel edge forms a 2-cycle")
print("  - After removing all 3 cycles, we get 3 disconnected paths:")
print("    • s1 → a1 → t1")
print("    • s2 → b2 → t2")
print("    • s3 → c3 → t3")

# Draw visualization
draw_graph_with_cycles(combined, after, removed)

print("\nVisualization saved!")
