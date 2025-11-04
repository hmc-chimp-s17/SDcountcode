from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define ports
S = [("s", i) for i in (1, 2, 3)]
T = [("t", i) for i in (1, 2, 3)]
A = [("a", i) for i in (1, 2, 3)]
B = [("b", i) for i in (1, 2, 3)]
C = [("c", i) for i in (1, 2, 3)]

# Example with parallel edges
FIXED = [
    (("s", 1), ("a", 1)),
    (("s", 2), ("b", 2)),
    (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)),
    (("b", 3), ("c", 2)),
    (("c", 1), ("a", 3)),
]

RHS = [
    (("t", 1), ("a", 1)),
    (("t", 2), ("a", 2)),
    (("t", 3), ("a", 3)),
    (("b", 1), ("b", 2)),
    (("b", 3), ("c", 2)),  # PARALLEL with FIXED
    (("c", 1), ("c", 3)),
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
    """Find any cycle using DFS."""
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


def merge_ST(p):
    """Merge s1,s2,s3 → 'S' and t1,t2,t3 → 'T'."""
    return 'S' if p in S else ('T' if p in T else p)


def draw_graph_comparison(before_edges, after_edges, removed_cycles):
    """Draw before and after cycle removal."""

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

    # Identify parallel edge
    parallel_edge = frozenset([("b", 3), ("c", 2)])

    # Identify cycle edges
    cycle_edge_sets = set()
    for cyc in removed_cycles:
        for edge in cyc:
            cycle_edge_sets.add(frozenset(edge))

    for ax_idx, (edges, title) in enumerate([
        (before_edges, f"BEFORE Cycle Removal ({len(before_edges)} edges)"),
        (after_edges, f"AFTER Cycle Removal ({len(after_edges)} edges)")
    ]):
        ax = axes[ax_idx]

        # Draw vertices
        for v, (x, y) in positions.items():
            r = 0.15
            ax.add_patch(plt.Circle((x, y), r, color=vertex_colors[v], ec='black', lw=2, zorder=3))
            label = f"{v[0]}{v[1]}"
            ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

        # Count parallel edges for offset
        edge_counts = defaultdict(int)
        for e in edges:
            u, v = tuple(e) if isinstance(e, frozenset) else e
            if u in positions and v in positions:
                key = frozenset([u, v])
                edge_counts[key] += 1

        # Draw edges with offsets for parallel edges
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

            # Determine edge style
            is_parallel = (key == parallel_edge)
            is_in_cycle = (key in cycle_edge_sets)

            if ax_idx == 0:  # Before
                if is_parallel:
                    color = 'red'
                    lw = 3
                    alpha = 0.8
                elif is_in_cycle:
                    color = 'orange'
                    lw = 2.5
                    alpha = 0.7
                else:
                    color = 'black'
                    lw = 2
                    alpha = 0.6
            else:  # After
                if is_parallel:
                    color = 'red'
                    lw = 3
                    alpha = 0.8
                else:
                    color = 'darkgreen'
                    lw = 2.5
                    alpha = 0.7

            if total_parallel > 1:
                offset = (current_idx - (total_parallel - 1) / 2) * 0.25
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

                # Add label
                label_text = f"{u[0]}{u[1]}-{v[0]}{v[1]}"
                ax.text(ctrl_x, ctrl_y, label_text, fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                       ha='center', va='center', zorder=2)
            else:
                ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                label_text = f"{u[0]}{u[1]}-{v[0]}{v[1]}"
                ax.text(mid_x, mid_y, label_text, fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7),
                       ha='center', va='center', zorder=2)

        ax.set_xlim(-1.5, 11.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # Add legend
        if ax_idx == 0:
            legend_text = "Red = Parallel edge\nOrange = Cycle edges\nBlack = Other edges"
        else:
            legend_text = "Red = Parallel edge\nGreen = Remaining edges"

        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add cycle information
    if removed_cycles:
        cycle_text = f"Cycles Removed: {len(removed_cycles)}\n\n"
        for i, cyc in enumerate(removed_cycles):
            cycle_text += f"Cycle {i+1}: "
            edge_strs = [f"{u[0]}{u[1]}-{v[0]}{v[1]}" for u, v in cyc[:3]]
            if len(cyc) > 3:
                edge_strs.append("...")
            cycle_text += " → ".join(edge_strs) + "\n"
    else:
        cycle_text = "No cycles removed!\nGraph is already acyclic."

    fig.text(0.5, 0.02, cycle_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    plt.suptitle("Cycle Removal Process - Parallel Edges Example",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig("before_after_cycles.png", dpi=150, bbox_inches='tight')
    print("Saved: before_after_cycles.png")
    plt.close()


# Process the example
combined = FIXED + RHS
after, removed = remove_all_cycles(combined)

print("="*60)
print("CYCLE REMOVAL ANALYSIS")
print("="*60)

print(f"\nBefore: {len(combined)} edges")
print(f"After: {len(after)} edges")
print(f"Cycles removed: {len(removed)}")

if removed:
    print("\nCycles found and removed:")
    for i, cyc in enumerate(removed):
        print(f"\nCycle {i+1}:")
        for u, v in cyc:
            print(f"  {u} - {v}")
else:
    print("\nNo cycles found! The graph is already acyclic.")

print("\n" + "="*60)
print("BEFORE (all edges):")
for i, e in enumerate(combined):
    u, v = e
    is_parallel = frozenset(e) == frozenset([("b", 3), ("c", 2)])
    marker = " ← PARALLEL" if is_parallel else ""
    print(f"  {i+1}. {u} - {v}{marker}")

print("\n" + "="*60)
print("AFTER (cycles removed):")
for i, e in enumerate(after):
    u, v = tuple(e) if isinstance(e, frozenset) else e
    is_parallel = frozenset(e) == frozenset([("b", 3), ("c", 2)])
    marker = " ← PARALLEL" if is_parallel else ""
    print(f"  {i+1}. {u} - {v}{marker}")

# Draw visualization
draw_graph_comparison(combined, after, removed)

print("\nVisualization saved!")
