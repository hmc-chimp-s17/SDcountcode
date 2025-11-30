"""
Draw results from SDcountnlo_v4.py showing all three paths.

LEFT: Complete graph with removed cycles highlighted in red, all three paths color-coded
RIGHT: All three merged+split paths showing cycles in each
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

from SDcountnlo_v4 import (
    check_nlo_paths, extract_path, merge_vertices, split_ac_edges, find_cycle, count_cycles
)


def draw_three_paths_comparison(result, label, filename):
    """Draw comparison with only paths that have cycles on the right."""
    # Determine which paths have cycles
    paths_with_cycles = []
    if result['cycles_s1_t1'] > 0:
        paths_with_cycles.append('s1_t1')
    if result['cycles_s2_t3'] > 0:
        paths_with_cycles.append('s2_t3')
    if result['cycles_s3_t2'] > 0:
        paths_with_cycles.append('s3_t2')

    fig = plt.figure(figsize=(18, 10))

    # LEFT subplot: Complete original graph
    ax1 = fig.add_subplot(121)

    # RIGHT subplot: Only paths with cycles (stacked vertically)
    ax2 = fig.add_subplot(122)

    # Original positions for complete graph
    pos_orig = {
        ('s', 1): (-0.3, 5.3), ('s', 2): (0, 5), ('s', 3): (-0.3, 4.7),
        ('t', 1): (10.3, 5.3), ('t', 2): (10, 5), ('t', 3): (10.3, 4.7),
        ('a', 1): (4, 8), ('a', 2): (3.5, 7), ('a', 3): (4.5, 7),
        ('b', 1): (5, 5), ('b', 2): (4.5, 4), ('b', 3): (5.5, 4),
        ('c', 1): (4, 2), ('c', 2): (3.5, 1), ('c', 3): (4.5, 1),
    }

    # Merged positions - separate section for each path with cycles
    # Each path gets its own vertical section with proper 2D layout
    # Calculate vertical offset for each path
    num_paths = len(paths_with_cycles)
    if num_paths == 1:
        y_offsets = [0]
        section_height = 10
    elif num_paths == 2:
        y_offsets = [5, -5]
        section_height = 10
    else:  # 3 paths
        y_offsets = [10, 0, -10]
        section_height = 10

    path_y_map = {}
    for i, path_name in enumerate(paths_with_cycles):
        path_y_map[path_name] = y_offsets[i]

    # ===== LEFT: Complete graph with all three paths color-coded =====

    # Extract edges for each path
    s1_t1_edge_set = {frozenset([u, v]) for u, v in result['s1_t1_path']}
    s2_t3_edge_set = {frozenset([u, v]) for u, v in result['s2_t3_path']}
    s3_t2_edge_set = {frozenset([u, v]) for u, v in result['s3_t2_path']}

    # Find removed cycle edges
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

            # Color priority: removed cycles (red) > paths (color-coded) > other (gray)
            if edge in removed_cycle_edges:
                color = 'red'
                lw = 3
                alpha = 0.9
            elif edge in s1_t1_edge_set:
                color = 'green'
                lw = 2.5
                alpha = 0.7
            elif edge in s2_t3_edge_set:
                color = 'blue'
                lw = 2.5
                alpha = 0.7
            elif edge in s3_t2_edge_set:
                color = 'purple'
                lw = 2.5
                alpha = 0.7
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
            if v[0] == 's':
                color = 'lightblue'
            elif v[0] == 't':
                color = 'plum'
            elif v[0] == 'a':
                color = 'lightgreen'
            elif v[0] == 'b':
                color = 'lightyellow'
            else:
                color = 'lightcoral'

            ax1.add_patch(plt.Circle((x, y), 0.15, color=color, ec='black', lw=2, zorder=3))
            ax1.text(x, y, f"{v[0]}{v[1]}", ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    ax1.set_xlim(-1.5, 11.5)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    title_text = f"Complete Graph ({result['cycles_removed']} cycle removed)\nGreen=s1-t1, Blue=s2-t3, Purple=s3-t2"
    ax1.set_title(title_text, fontsize=12, fontweight='bold')

    # ===== RIGHT: Only merged+split paths with cycles in 2D layout =====

    # Colors for vertices in merged view
    vcolors_merged = {
        ('s', 1): 'lightblue', ('s', 2): 'lightblue', ('s', 3): 'lightblue',
        ('t', 1): 'plum', ('t', 2): 'plum', ('t', 3): 'plum',
        ('a',): 'lightgreen', ('b',): 'lightyellow', ('c',): 'lightcoral',
    }

    # Path info: name, split_edges, color, s_node, t_node
    path_info_map = {
        's1_t1': (result['split_s1_t1'], 'green', ('s', 1), ('t', 1)),
        's2_t3': (result['split_s2_t3'], 'blue', ('s', 2), ('t', 3)),
        's3_t2': (result['split_s3_t2'], 'purple', ('s', 3), ('t', 2)),
    }

    # Get cycles and draw only paths with cycles
    for path_name in paths_with_cycles:
        split_edges, color, s_node, t_node = path_info_map[path_name]
        y_offset = path_y_map[path_name]

        # Get cycles for this path
        cycles = []
        remaining = split_edges[:]
        while True:
            cycle = find_cycle(remaining)
            if not cycle:
                break
            cycles.append(cycle)
            cycle_set = {frozenset(e) for e in cycle}
            remaining = [e for e in remaining if frozenset(e) not in cycle_set]

        # Identify cycle edges
        cycle_edges = set()
        for cycle in cycles:
            for edge in cycle:
                cycle_edges.add(frozenset(edge))

        # Define 2D positions for this path section (like v2/v3 - triangular layout)
        # Base positions with y_offset applied
        pos_section = {
            s_node: (0, 5 + y_offset),
            t_node: (10, 5 + y_offset),
            ('a',): (3, 7 + y_offset),
            ('b',): (5, 5 + y_offset),
            ('c',): (5, 3 + y_offset),
        }

        # Draw edges for this path
        for u, v in split_edges:
            if u in pos_section and v in pos_section:
                x1, y1 = pos_section[u]
                x2, y2 = pos_section[v]
                edge = frozenset([u, v])

                if edge in cycle_edges:
                    edge_color = 'red'
                    lw = 3
                    alpha = 0.9
                else:
                    edge_color = color
                    lw = 2
                    alpha = 0.6

                ax2.plot([x1, x2], [y1, y2], color=edge_color, lw=lw, alpha=alpha, zorder=1)

        # Draw vertices for this path
        all_vertices_in_path = set()
        for u, v in split_edges:
            all_vertices_in_path.add(u)
            all_vertices_in_path.add(v)

        for v in all_vertices_in_path:
            if v in pos_section:
                x, y = pos_section[v]
                ax2.add_patch(plt.Circle((x, y), 0.15, color=vcolors_merged[v], ec='black', lw=2, zorder=3))

                # Label formatting
                if v[0] in ['s', 't']:
                    label = f"{v[0]}{v[1]}"
                else:
                    label = f"{v[0]}"

                ax2.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

        # Add path label on the left
        if path_name == 's1_t1':
            label_text = "s1→t1:"
        elif path_name == 's2_t3':
            label_text = "s2→t3:"
        else:
            label_text = "s3→t2:"
        ax2.text(-1.5, 5 + y_offset, label_text, ha='right', va='center', fontsize=11, fontweight='bold', color=color)

    # Adjust y limits based on number of paths
    if num_paths == 1:
        ax2.set_ylim(1, 9)
    elif num_paths == 2:
        ax2.set_ylim(-1, 11)
    else:  # 3 paths
        ax2.set_ylim(-6, 16)

    ax2.set_xlim(-2, 11)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Build title showing only paths with cycles
    title_text = "Merged+Split Paths with Cycles (cycles in red)\n"
    cycle_info = []
    if result['cycles_s1_t1'] > 0:
        cycle_info.append(f"s1-t1: {result['cycles_s1_t1']} cycles")
    if result['cycles_s2_t3'] > 0:
        cycle_info.append(f"s2-t3: {result['cycles_s2_t3']} cycles")
    if result['cycles_s3_t2'] > 0:
        cycle_info.append(f"s3-t2: {result['cycles_s3_t2']} cycles")
    title_text += ", ".join(cycle_info)
    ax2.set_title(title_text, fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle(label, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def draw_all_v4_results(label, results, output_dir):
    """Draw all matchings from v4 results."""
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

        draw_three_paths_comparison(result, title, filename)

    print(f"\nSaved {len(results)} drawings to {output_dir}/")


def main():
    print("="*60)
    print("Drawing all SDcountnlo_v4 results")
    print("="*60)

    # Load results from v4
    from SDcountnlo_v4 import find_matchings_with_cycles_in_any_path

    # Process 6J
    results_6j = find_matchings_with_cycles_in_any_path(FIXED_6J, "6J")
    draw_all_v4_results("6J", results_6j, "nlo_v4_results/6J/drawings")

    # Process PILLOW
    results_pillow = find_matchings_with_cycles_in_any_path(FIXED_PILLOW, "PILLOW")
    draw_all_v4_results("PILLOW", results_pillow, "nlo_v4_results/PILLOW/drawings")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
