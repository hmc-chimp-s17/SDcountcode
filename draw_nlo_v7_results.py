"""
Draw results from SDcountnlo_v7.py showing s1-t1 path with a-c splitting.

LEFT: Complete graph with removed cycles highlighted in red, s1-t1 path in green
RIGHT: Merged+split s1-t1 path showing cycles in red
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

from SDcount_v3 import (
    S, T, A, B, C, FIXED_6J, FIXED_PILLOW,
    generate_rhs_matchings, check_degrees, remove_cycles,
    all_t_same_type, build_adj
)

from SDcountnlo_v7 import (
    check_nlo_paths, extract_s1_t1_path, merge_vertices, split_ac_edges, find_cycle, count_cycles
)


def draw_comparison(result, label, filename):
    """Draw side-by-side comparison: complete graph vs merged+split s1-t1 path."""
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
        ('s', 1): (0, 5), ('t', 1): (10, 5),
        ('a',): (3, 7), ('b',): (5, 5), ('c',): (5, 3),
    }

    vcolors = {
        **{v: 'lightblue' for v in [('s', 1)]},
        **{v: 'plum' for v in [('t', 1)]},
        ('a',): 'lightgreen', ('b',): 'lightyellow', ('c',): 'lightcoral',
    }

    # LEFT: Complete graph before cycle removal (highlight removed cycles and s1-t1 path)
    s1_t1_vertices = set([u for u, _ in result['s1_t1_path']] + [v for _, v in result['s1_t1_path']])
    s1_t1_edge_set = {frozenset([u, v]) for u, v in result['s1_t1_path']}

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

            # Color priority: removed cycles (red) > s1-t1 path (green) > other (gray)
            if edge in removed_cycle_edges:
                color = 'red'
                lw = 3
                alpha = 0.9
            elif edge in s1_t1_edge_set:
                color = 'green'
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
            # Highlight s1-t1 path vertices with thicker border
            ec_width = 3 if v in s1_t1_vertices else 2
            ax1.add_patch(plt.Circle((x, y), 0.15, color=color, ec='black', lw=ec_width, zorder=3))
            ax1.text(x, y, f"{v[0]}{v[1]}", ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    ax1.set_xlim(-1.5, 11.5)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    title_text = f"Complete Graph ({result['cycles_removed']} cycle removed, s1-t1 in green)"
    ax1.set_title(title_text, fontsize=12, fontweight='bold')

    # RIGHT: merged+split s1-t1 path with cycles
    # Get all cycles
    cycles = []
    remaining = result['split_path'][:]
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
    for u, v in result['split_path']:
        if u in pos_merged and v in pos_merged:
            edge_counts[frozenset([u, v])] += 1

    # Draw each edge
    for u, v in result['split_path']:
        if u in pos_merged and v in pos_merged:
            x1, y1 = pos_merged[u]
            x2, y2 = pos_merged[v]
            edge = frozenset([u, v])

            if edge in cycle_edges:
                edge_color = 'red'
                lw = 3
                alpha = 0.9
            else:
                edge_color = 'green'
                lw = 2
                alpha = 0.6

            # For multiple edges, draw them with slight offset
            count = edge_counts[edge]
            if count > 1:
                # Draw curved edges for parallel edges
                offset_idx = 0
                for i, (u2, v2) in enumerate(result['split_path']):
                    if frozenset([u2, v2]) == edge:
                        if (u2, v2) == (u, v):
                            # Calculate curve offset
                            dx = x2 - x1
                            dy = y2 - y1
                            norm = np.sqrt(dx**2 + dy**2)
                            if norm > 0:
                                perp_x = -dy / norm
                                perp_y = dx / norm
                                offset = 0.3 * (offset_idx - (count-1)/2)
                                mid_x = (x1 + x2) / 2 + perp_x * offset
                                mid_y = (y1 + y2) / 2 + perp_y * offset

                                t = np.linspace(0, 1, 20)
                                curve_x = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
                                curve_y = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
                                ax2.plot(curve_x, curve_y, color=edge_color, lw=lw, alpha=alpha, zorder=1)
                            offset_idx += 1
                            break
            else:
                # Single edge - draw straight
                ax2.plot([x1, x2], [y1, y2], color=edge_color, lw=lw, alpha=alpha, zorder=1)

    # Draw vertices
    all_vertices_in_path = set()
    for u, v in result['split_path']:
        all_vertices_in_path.add(u)
        all_vertices_in_path.add(v)

    for v in all_vertices_in_path:
        if v in pos_merged:
            x, y = pos_merged[v]
            ax2.add_patch(plt.Circle((x, y), 0.15, color=vcolors[v], ec='black', lw=2, zorder=3))

            # Label formatting
            if v[0] in ['s', 't']:
                label_text = f"{v[0]}{v[1]}"
            else:
                label_text = f"{v[0]}"

            ax2.text(x, y, label_text, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    ax2.set_xlim(-1, 11)
    ax2.set_ylim(1, 9)
    ax2.set_aspect('equal')
    ax2.axis('off')
    title_text = f"Merged+Split s1-t1 Path ({result['num_cycles']} cycles, cycles in red)"
    ax2.set_title(title_text, fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle(label, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def draw_all_v7_results(label, results, output_dir):
    """Draw all matchings from v7 results."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Drawing {label} results")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        if result['type'] == '1-cycle non-special':
            filename = f"{output_dir}/1cycle_nonspecial_{i:03d}_idx{result['idx']}.png"
            title = f"{label} - 1-cycle non-special {i}/{len(results)} (idx {result['idx']})"
        else:
            t_type = result['t_type']
            filename = f"{output_dir}/0cycle_special_{i:03d}_idx{result['idx']}_type{t_type}.png"
            title = f"{label} - 0-cycle special (all t→{t_type}) {i}/{len(results)} (idx {result['idx']})"

        draw_comparison(result, title, filename)

    print(f"\nSaved {len(results)} drawings to {output_dir}/")


def main():
    print("="*60)
    print("Drawing all SDcountnlo_v7 results")
    print("="*60)

    # Load results from v7
    from SDcountnlo_v7 import find_matchings_with_s1_t1_cycles

    # Process 6J
    results_6j = find_matchings_with_s1_t1_cycles(FIXED_6J, "6J")
    draw_all_v7_results("6J", results_6j, "nlo_v7_results/6J/drawings")

    # Process PILLOW
    results_pillow = find_matchings_with_s1_t1_cycles(FIXED_PILLOW, "PILLOW")
    draw_all_v7_results("PILLOW", results_pillow, "nlo_v7_results/PILLOW/drawings")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
