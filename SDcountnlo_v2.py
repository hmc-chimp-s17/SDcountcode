"""
SDcountnlo_v2.py - Simple and Intuitive Version

Find matchings where the s2-t3 path has more than one cycle after merging a,b,c vertices.

Steps:
1. Find non-special 1-cycle and special 0-cycle matchings with NLO pairing (s1-t1, s2-t3, s3-t2)
2. Remove cycles
3. Extract the s2-t3 path
4. Merge vertices: a1,a2,a3 → a; b1,b2,b3 → b; c1,c2,c3 → c
5. Count cycles in the merged s2-t3 path
6. Keep only matchings with >1 cycles
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import os

from SDcount_v3 import (
    S, T, A, B, C, FIXED_6J, FIXED_PILLOW,
    generate_rhs_matchings, check_degrees, remove_cycles,
    all_t_same_type, build_adj
)

# NLO pairing: s1-t1, s2-t3, s3-t2
NLO_PAIRING = {1: 1, 2: 3, 3: 2}


def check_nlo_paths(edges):
    """Check if graph has 3 disjoint paths with NLO pairing: s1→t1, s2→t3, s3→t2."""
    adj = build_adj(edges)
    visited = set()

    # Find all connected components
    components = []
    for s in S:
        if s in visited:
            continue
        # BFS from s
        comp = {s}
        queue = [s]
        visited.add(s)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    comp.add(nb)
                    queue.append(nb)
        components.append(comp)

    if len(components) != 3:
        return False

    # Check each component
    for comp in components:
        s_nodes = [v for v in comp if v in S]
        t_nodes = [v for v in comp if v in T]

        if len(s_nodes) != 1 or len(t_nodes) != 1:
            return False

        # Check NLO pairing
        s_idx = s_nodes[0][1]
        t_idx = t_nodes[0][1]
        if NLO_PAIRING[s_idx] != t_idx:
            return False

        # Check path structure (degree 1 or 2)
        for v in comp:
            deg = len([nb for nb in adj[v] if nb in comp])
            if v in S or v in T:
                if deg != 1:
                    return False
            else:
                if deg != 2:
                    return False

    return True


def extract_s2_t3_path(edges):
    """Get all edges in the s2-t3 connected component."""
    adj = build_adj(edges)

    # BFS from s2 to find component
    s2 = ('s', 2)
    component = {s2}
    queue = [s2]
    visited = {s2}

    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                component.add(nb)
                queue.append(nb)

    # Extract edges in this component
    path_edges = [(u, v) for u, v in edges if u in component and v in component]
    return path_edges


def merge_vertices(edges):
    """Merge a1,a2,a3→a, b1,b2,b3→b, c1,c2,c3→c."""
    merged = []
    for u, v in edges:
        # Merge intermediate vertices
        u_new = (u[0],) if u[0] in ['a', 'b', 'c'] else u
        v_new = (v[0],) if v[0] in ['a', 'b', 'c'] else v
        merged.append((u_new, v_new))
    return merged


def find_cycle(edges):
    """Find one cycle in the graph."""
    if not edges:
        return None

    # Check self-loops
    for u, v in edges:
        if u == v:
            return [(u, v)]

    # Check parallel edges (2-cycles)
    edge_count = defaultdict(int)
    for u, v in edges:
        edge_count[frozenset([u, v])] += 1
    for endpoints, count in edge_count.items():
        if count >= 2:
            u, v = tuple(endpoints)
            return [(u, v), (v, u)]

    # Find longer cycles with DFS
    adj = build_adj(edges)
    visited = set()
    parent = {}

    def dfs(node, par):
        visited.add(node)
        for nb in adj[node]:
            if nb == par:
                continue
            if nb in visited:
                # Found cycle - reconstruct it
                cycle = []
                cur = node
                while cur != nb:
                    cycle.append((cur, parent[cur]))
                    cur = parent[cur]
                cycle.append((nb, node))
                return cycle
            parent[nb] = node
            result = dfs(nb, node)
            if result:
                return result
        return None

    for start in adj:
        if start not in visited:
            parent[start] = None
            cycle = dfs(start, None)
            if cycle:
                return cycle

    return None


def count_cycles(edges):
    """Count how many cycles are in the graph."""
    num_cycles = 0
    remaining = edges[:]

    while True:
        cycle = find_cycle(remaining)
        if not cycle:
            break
        num_cycles += 1
        # Remove cycle edges
        cycle_set = {frozenset(e) for e in cycle}
        remaining = [e for e in remaining if frozenset(e) not in cycle_set]

    return num_cycles


def find_matchings_with_multiple_cycles(fixed_edges, label):
    """Find all matchings where s2-t3 path has >1 cycles after merging."""
    print(f"\n{'='*60}")
    print(f"Processing {label}")
    print(f"{'='*60}")

    # Generate all RHS matchings
    all_rhs = generate_rhs_matchings()

    # Find matchings that pass NLO path test
    one_cycle_nonspecial = []
    zero_cycle_special = []

    for idx, rhs in enumerate(all_rhs):
        combined = fixed_edges + rhs

        if not check_degrees(combined):
            continue

        # Remove cycles
        after_removal, num_cycles = remove_cycles(combined)

        if not check_nlo_paths(after_removal):
            continue

        # Save 1-cycle non-special or 0-cycle special (save both combined and after_removal)
        if num_cycles == 1 and not all_t_same_type(rhs):
            one_cycle_nonspecial.append((idx, rhs, combined, after_removal, num_cycles))
        elif num_cycles == 0 and all_t_same_type(rhs):
            t_type = all_t_same_type(rhs)
            zero_cycle_special.append((idx, rhs, combined, after_removal, num_cycles, t_type))

    print(f"Found {len(one_cycle_nonspecial)} non-special 1-cycle matchings")
    print(f"Found {len(zero_cycle_special)} special 0-cycle matchings")

    # Check which have >1 cycles in s2-t3 after merging
    results = []

    print(f"\nChecking s2-t3 path after merging...")

    for idx, rhs, combined, after_removal, cycles_removed in one_cycle_nonspecial:
        s2_t3_path = extract_s2_t3_path(after_removal)
        merged_path = merge_vertices(s2_t3_path)
        num_cycles_in_path = count_cycles(merged_path)

        if num_cycles_in_path > 1:
            results.append({
                'type': '1-cycle non-special',
                'idx': idx,
                'rhs': rhs,
                'combined': combined,
                'after_removal': after_removal,
                'cycles_removed': cycles_removed,
                's2_t3_path': s2_t3_path,
                'merged_path': merged_path,
                'num_cycles': num_cycles_in_path,
                't_type': None
            })

    for idx, rhs, combined, after_removal, cycles_removed, t_type in zero_cycle_special:
        s2_t3_path = extract_s2_t3_path(after_removal)
        merged_path = merge_vertices(s2_t3_path)
        num_cycles_in_path = count_cycles(merged_path)

        if num_cycles_in_path > 1:
            results.append({
                'type': '0-cycle special',
                'idx': idx,
                'rhs': rhs,
                'combined': combined,
                'after_removal': after_removal,
                'cycles_removed': cycles_removed,
                's2_t3_path': s2_t3_path,
                'merged_path': merged_path,
                'num_cycles': num_cycles_in_path,
                't_type': t_type
            })

    # Count by type
    one_cycle_count = sum(1 for r in results if r['type'] == '1-cycle non-special')
    zero_cycle_count = sum(1 for r in results if r['type'] == '0-cycle special')

    print(f"\nResults (s2-t3 path with >1 cycles after merging):")
    print(f"  1-cycle non-special: {one_cycle_count}")
    print(f"  0-cycle special: {zero_cycle_count}")
    print(f"  Total: {len(results)}")

    return results


def draw_comparison(result, label, filename):
    """Draw side-by-side comparison: before merge vs after merge."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Vertex positions
    pos_orig = {
        ('s', 1): (-0.3, 5.3), ('s', 2): (0, 5), ('s', 3): (-0.3, 4.7),
        ('t', 1): (10.3, 5.3), ('t', 2): (10, 5), ('t', 3): (10.3, 4.7),
        ('a', 1): (4, 8), ('a', 2): (3.5, 7), ('a', 3): (4.5, 7),
        ('b', 1): (5, 5), ('b', 2): (4.5, 4), ('b', 3): (5.5, 4),
        ('c', 1): (4, 2), ('c', 2): (3.5, 1), ('c', 3): (4.5, 1),
    }

    pos_merged = {
        ('s', 2): (0, 5), ('t', 3): (10, 5),
        ('a',): (3, 7), ('b',): (5, 5), ('c',): (5, 3),
    }

    vcolors = {
        **{v: 'lightblue' for v in [('s', 2)]},
        **{v: 'plum' for v in [('t', 3)]},
        ('a',): 'lightgreen', ('b',): 'lightyellow', ('c',): 'lightcoral',
    }

    # LEFT: Complete graph before cycle removal (highlight removed cycles and s2-t3 path)
    s2_t3_vertices = set([u for u, _ in result['s2_t3_path']] + [v for _, v in result['s2_t3_path']])
    s2_t3_edge_set = {frozenset([u, v]) for u, v in result['s2_t3_path']}

    # Find removed cycle edges (edges in combined but not in after_removal)
    after_removal_set = {frozenset([u, v]) for u, v in result['after_removal']}
    removed_cycle_edges = set()
    for u, v in result['combined']:
        edge = frozenset([u, v])
        if edge not in after_removal_set:
            removed_cycle_edges.add(edge)

    # Draw all edges from original combined graph
    for u, v in result['combined']:
        if u in pos_orig and v in pos_orig:
            x1, y1 = pos_orig[u]
            x2, y2 = pos_orig[v]
            edge = frozenset([u, v])

            # Color priority: removed cycles (red) > s2-t3 path (blue) > other (gray)
            if edge in removed_cycle_edges:
                color = 'red'
                lw = 3
                alpha = 0.9
            elif edge in s2_t3_edge_set:
                color = 'blue'
                lw = 2.5
                alpha = 0.8
            else:
                color = 'lightgray'
                lw = 1.5
                alpha = 0.4

            ax1.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)

    # Draw all vertices
    all_vertices = set([u for u, _ in result['combined']] + [v for _, v in result['combined']])
    for v in all_vertices:
        if v in pos_orig:
            x, y = pos_orig[v]
            color = 'lightblue' if v[0] == 's' else 'plum' if v[0] == 't' else 'lightgreen' if v[0] == 'a' else 'lightyellow' if v[0] == 'b' else 'lightcoral'
            # Highlight s2-t3 path vertices with thicker border
            ec_width = 3 if v in s2_t3_vertices else 2
            ax1.add_patch(plt.Circle((x, y), 0.15, color=color, ec='black', lw=ec_width, zorder=3))
            ax1.text(x, y, f"{v[0]}{v[1]}", ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    ax1.set_xlim(-1.5, 11.5)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    title_text = f"Complete Graph ({result['cycles_removed']} cycle removed, s2-t3 in blue)"
    ax1.set_title(title_text, fontsize=12, fontweight='bold')

    # RIGHT: merged s2-t3 path with cycles
    # Get all cycles
    cycles = []
    remaining = result['merged_path'][:]
    while True:
        cycle = find_cycle(remaining)
        if not cycle:
            break
        cycles.append(cycle)
        cycle_set = {frozenset(e) for e in cycle}
        remaining = [e for e in remaining if frozenset(e) not in cycle_set]

    cycle_edges = set()
    for cycle in cycles:
        for e in cycle:
            cycle_edges.add(frozenset(e))

    # Draw edges
    edge_counts = defaultdict(int)
    for u, v in result['merged_path']:
        edge_counts[frozenset([u, v])] += 1

    edge_drawn = defaultdict(int)
    for u, v in result['merged_path']:
        if u in pos_merged and v in pos_merged:
            key = frozenset([u, v])
            is_cycle = key in cycle_edges
            color = 'red' if is_cycle else 'gray'
            lw = 3 if is_cycle else 2

            x1, y1 = pos_merged[u]
            x2, y2 = pos_merged[v]

            # Handle parallel edges
            if edge_counts[key] > 1:
                offset = (edge_drawn[key] - (edge_counts[key] - 1) / 2) * 0.4
                import numpy as np
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                norm = (dx**2 + dy**2)**0.5
                if norm > 0:
                    perp_x, perp_y = -dy / norm, dx / norm
                    ctrl_x = mid_x + perp_x * offset
                    ctrl_y = mid_y + perp_y * offset
                    t = np.linspace(0, 1, 50)
                    bx = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
                    by = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
                    ax2.plot(bx, by, color=color, lw=lw, zorder=1)
                edge_drawn[key] += 1
            else:
                ax2.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1)

    for v in set([u for u, _ in result['merged_path']] + [v for _, v in result['merged_path']]):
        if v in pos_merged:
            x, y = pos_merged[v]
            ax2.add_patch(plt.Circle((x, y), 0.2, color=vcolors.get(v, 'white'), ec='black', lw=2, zorder=3))
            label = f"{v[0]}" if len(v) == 1 else f"{v[0]}{v[1]}"
            ax2.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

    ax2.set_xlim(-1, 11)
    ax2.set_ylim(1.5, 8)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f"After Merge ({result['num_cycles']} cycles)", fontsize=12, fontweight='bold')

    # Title
    type_str = result['type']
    if result['t_type']:
        type_str += f" (all t→{result['t_type']})"
    fig.suptitle(f"{label} - {type_str} - idx {result['idx']}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_results(label, results, output_dir):
    """Save results and drawings."""
    os.makedirs(output_dir, exist_ok=True)

    # Text summary
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"{label} - s2-t3 Path with >1 Cycles After Merging\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total: {len(results)} matchings\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"Matching {i} - {r['type']} (idx {r['idx']})\n")
            if r['t_type']:
                f.write(f"  Type: all t→{r['t_type']}\n")
            f.write(f"  Cycles in s2-t3: {r['num_cycles']}\n")
            f.write(f"  RHS edges: ")
            f.write(", ".join([f"{u[0]}{u[1]}-{v[0]}{v[1]}" for u, v in r['rhs']]))
            f.write("\n\n")

    # Drawings
    print(f"\nDrawing {len(results)} examples...")
    for i, r in enumerate(results, 1):
        type_prefix = "1cycle" if r['type'] == '1-cycle non-special' else "0cycle_special"
        suffix = f"_type{r['t_type']}" if r['t_type'] else ""
        filename = f"{output_dir}/{type_prefix}_{i:03d}_idx{r['idx']}{suffix}.png"
        draw_comparison(r, label, filename)

    print(f"\nResults saved to {output_dir}/")


def main():
    print("="*60)
    print("SDcountnlo_v2: s2-t3 Path Cycle Analysis")
    print("="*60)

    # Process 6J
    results_6j = find_matchings_with_multiple_cycles(FIXED_6J, "6J")
    save_results("6J", results_6j, "nlo_v2_results/6J")

    # Process PILLOW
    results_pillow = find_matchings_with_multiple_cycles(FIXED_PILLOW, "PILLOW")
    save_results("PILLOW", results_pillow, "nlo_v2_results/PILLOW")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
