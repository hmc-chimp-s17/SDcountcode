"""
Draw all special case 0-cycle matchings that pass the untwisted test.
No twist parameters involved - just showing the graph structure.
Special case: all t's connect to the same type (a, b, or c).
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

# Import from main module
from SDcountnlo_twist import (
    FIXED_6J, FIXED_PILLOW, S, T, A, B, C, RHS_PORTS, NLO_PAIRING,
    generate_rhs_matchings, check_degrees, remove_cycles_with_zero_twist,
    check_three_paths_structure, all_t_same_type, get_removed_cycles
)


def draw_untwisted(combined, cycles, title, filename):
    """Draw graph with highlighted cycles (no twist labels)."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Positions
    pos = {
        ('s', 1): (-0.3, 5.3), ('s', 2): (0, 5), ('s', 3): (-0.3, 4.7),
        ('t', 1): (10.3, 5.3), ('t', 2): (10, 5), ('t', 3): (10.3, 4.7),
        ('a', 1): (4, 8), ('a', 2): (3.5, 7), ('a', 3): (4.5, 7),
        ('b', 1): (5, 5), ('b', 2): (4.5, 4), ('b', 3): (5.5, 4),
        ('c', 1): (4, 2), ('c', 2): (3.5, 1), ('c', 3): (4.5, 1),
    }

    # Vertex colors
    vcolors = {
        **{v: 'lightblue' for v in S},
        **{v: 'plum' for v in T},
        **{v: 'lightgreen' for v in A},
        **{v: 'lightyellow' for v in B},
        **{v: 'lightcoral' for v in C},
    }

    # Draw vertices first
    for v, (x, y) in pos.items():
        r = 0.15
        ax.add_patch(plt.Circle((x, y), r, color=vcolors[v], ec='black', lw=2, zorder=3))
        ax.text(x, y, f"{v[0]}{v[1]}", ha='center', va='center',
               fontsize=10, fontweight='bold', zorder=4)

    # Identify cycle edges
    cycle_edges = []
    colors = ['red', 'orange', 'purple']
    for i, cycle in enumerate(cycles):
        for edge, twist in cycle:
            cycle_edges.append((frozenset(edge), i))

    # Count parallel edges
    edge_counts = defaultdict(int)
    for u, v in combined:
        edge_counts[frozenset([u, v])] += 1

    # Draw edges
    edge_drawn = defaultdict(int)
    for u, v in combined:
        if u not in pos or v not in pos:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        key = frozenset([u, v])
        total_parallel = edge_counts[key]
        current_idx = edge_drawn[key]
        edge_drawn[key] += 1

        # Check if in cycle
        cycle_idx = None
        for ck, cidx in cycle_edges:
            if ck == key:
                cycle_idx = cidx
                break

        if cycle_idx is not None:
            color = colors[cycle_idx % len(colors)]
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
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add cycle information
    info_text = "SPECIAL: All t's → same type\n"
    if len(cycles) == 0:
        info_text += "0 cycles removed\n"
    else:
        for i, cyc in enumerate(cycles):
            if len(cyc) == 2:
                info_text += f"Cycle {i+1} (2-cycle)\n"
            else:
                info_text += f"Cycle {i+1} ({len(cyc)}-cycle)\n"

    info_text += f"\nPaths: s1-t1, s2-t3, s3-t2 (untwisted)"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def process_configuration(fixed_edges, config_label, output_dir):
    """Process a single configuration and draw all 0-cycle special matchings."""
    print("\n" + "="*60)
    print(f"Processing {config_label}")
    print("="*60)

    # Generate all RHS matchings
    all_rhs = generate_rhs_matchings()

    # Find matchings with valid degrees
    valid_matchings = []
    for idx, rhs in enumerate(all_rhs):
        combined = fixed_edges + rhs
        edges_no_twist = [((u, v), 0) for u, v in combined]
        if check_degrees(edges_no_twist):
            valid_matchings.append((idx, rhs, combined))

    print(f"Total matchings with valid degrees: {len(valid_matchings)}")

    # Find 0-cycle SPECIAL matchings that pass untwisted test
    zero_cycle_special = []

    for idx, rhs, combined in valid_matchings:
        # Only special cases
        if not all_t_same_type(rhs):
            continue

        edges_no_twist = [((u, v), 0) for u, v in combined]
        after, num_cycles, _ = remove_cycles_with_zero_twist(edges_no_twist)

        if num_cycles == 0 and check_three_paths_structure(after):
            # Get the cycle information (should be empty for 0-cycle)
            cycles = get_removed_cycles(edges_no_twist)

            # Determine which type all t's connect to
            t_type = None
            for u, v in rhs:
                if u[0] == 't':
                    t_type = v[0]
                    break
                elif v[0] == 't':
                    t_type = u[0]
                    break

            zero_cycle_special.append((idx, rhs, combined, cycles, t_type))

    print(f"Found {len(zero_cycle_special)} special 0-cycle matchings")

    # Group by t_type
    by_type = defaultdict(list)
    for idx, rhs, combined, cycles, t_type in zero_cycle_special:
        by_type[t_type].append((idx, rhs, combined, cycles))

    print(f"  By type: a={len(by_type.get('a', []))}, b={len(by_type.get('b', []))}, c={len(by_type.get('c', []))}")

    # Draw all of them
    for i, (idx, rhs, combined, cycles, t_type) in enumerate(zero_cycle_special, 1):
        title = f"{config_label} SPECIAL (all t→{t_type}) - 0 cycles - Matching {i}/{len(zero_cycle_special)} (index {idx})"
        filename = f"{output_dir}/matching_{i:03d}_idx{idx}_type{t_type}.png"
        draw_untwisted(combined, cycles, title, filename)

    # Save summary
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"{config_label} - All 0-cycle SPECIAL matchings (untwisted)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total matchings: {len(zero_cycle_special)}\n")
        f.write(f"  Type a: {len(by_type.get('a', []))}\n")
        f.write(f"  Type b: {len(by_type.get('b', []))}\n")
        f.write(f"  Type c: {len(by_type.get('c', []))}\n\n")

        for i, (idx, rhs, combined, cycles, t_type) in enumerate(zero_cycle_special, 1):
            f.write(f"Matching {i} (index {idx}, type {t_type}):\n")
            f.write("  RHS edges:\n")
            for u, v in rhs:
                f.write(f"    {u[0]}{u[1]}-{v[0]}{v[1]}\n")
            f.write("\n")

    print(f"Saved to {output_dir}/")


def main():
    print("="*60)
    print("Drawing all 0-cycle SPECIAL untwisted matchings")
    print("="*60)

    # Create main output directory
    base_dir = "untwisted_special_0cycle_all"
    os.makedirs(base_dir, exist_ok=True)

    # Process 6J configuration
    output_dir_6j = f"{base_dir}/6J"
    os.makedirs(output_dir_6j, exist_ok=True)
    process_configuration(FIXED_6J, "6J", output_dir_6j)

    # Process PILLOW configuration
    output_dir_pillow = f"{base_dir}/PILLOW"
    os.makedirs(output_dir_pillow, exist_ok=True)
    process_configuration(FIXED_PILLOW, "PILLOW", output_dir_pillow)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
