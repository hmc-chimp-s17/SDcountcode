"""
Draw examples of matchings that pass the untwisted test (not special cases),
showing all valid twist assignments for each.
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

# Import from main module
from SDcountnlo_twist import (
    FIXED_6J, FIXED_PILLOW, S, T, A, B, C, RHS_PORTS, NLO_PAIRING, REQUIRED_TWIST,
    generate_rhs_matchings, check_degrees, remove_cycles_with_zero_twist,
    check_three_paths_structure, check_three_paths_with_twist, find_cycle,
    all_t_same_type, get_removed_cycles
)


def draw_with_twist(combined, twists, cycles, title, filename):
    """Draw graph with highlighted cycles and twist values."""
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

    # Draw edges with twist labels
    edge_drawn = defaultdict(int)
    for i, (u, v) in enumerate(combined):
        if u not in pos or v not in pos:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        twist = twists[i]

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

            # Twist label
            twist_str = f"{twist:+d}" if twist != 0 else "0"
            ax.text(ctrl_x, ctrl_y, twist_str, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.9),
                   ha='center', va='center', zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Twist label
            twist_str = f"{twist:+d}" if twist != 0 else "0"
            ax.text(mid_x, mid_y, twist_str, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.9),
                   ha='center', va='center', zorder=2)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add cycle information
    info_text = ""
    for i, cyc in enumerate(cycles):
        total_twist = sum(t for e, t in cyc)
        if len(cyc) == 2:
            info_text += f"Cycle {i+1} (2-cycle): twist sum = {total_twist}\n"
        else:
            info_text += f"Cycle {i+1} ({len(cyc)}-cycle): twist sum = {total_twist}\n"

    # Path info
    info_text += f"\nPaths: s1-t1 (tw=0), s2-t3 (tw=1), s3-t2 (tw=0)"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def process_matchings(matchings_by_cycles, output_dir, config_label, is_special=False):
    """Process matchings and generate all twist variants."""
    for n_cycles in sorted(matchings_by_cycles.keys()):
        if not matchings_by_cycles[n_cycles]:
            continue

        # Pick the first matching
        idx, rhs, combined = matchings_by_cycles[n_cycles][0]

        print(f"\n{'='*60}")
        if is_special:
            # Determine which type all t's connect to
            t_type = None
            for u, v in rhs:
                if u[0] == 't':
                    t_type = v[0]
                    break
                elif v[0] == 't':
                    t_type = u[0]
                    break
            print(f"SPECIAL Matching with {n_cycles} cycles (index {idx}, all t→{t_type})")
        else:
            print(f"Matching with {n_cycles} cycles (index {idx})")
        print(f"{'='*60}")

        # Print the matching
        print("\nRHS edges:")
        for u, v in rhs:
            print(f"  {u[0]}{u[1]}-{v[0]}{v[1]}")

        # Find all valid twist assignments for this matching
        num_fixed = 6
        num_rhs = 6
        twist_values = [0, 1, -1]

        valid_twist_assignments = []

        for rhs_twists in product(twist_values, repeat=num_rhs):
            # Create edges with twist
            fixed_with_twist = [((u, v), 0) for u, v in combined[:num_fixed]]
            rhs_with_twist = [((u, v), t) for (u, v), t in zip(combined[num_fixed:], rhs_twists)]
            edges_with_twist = fixed_with_twist + rhs_with_twist

            # Remove cycles with twist sum = 0
            after, removed_cycles, _ = remove_cycles_with_zero_twist(edges_with_twist)

            # Check if there are remaining cycles
            if find_cycle(after) is not None:
                continue

            # Check if valid paths with correct twists
            valid, path_info = check_three_paths_with_twist(after)

            if valid:
                cycles = get_removed_cycles(edges_with_twist)
                full_twists = (0,) * num_fixed + rhs_twists
                valid_twist_assignments.append((full_twists, cycles, removed_cycles))

        print(f"\nValid twist assignments: {len(valid_twist_assignments)}")

        # Draw all valid twist assignments
        if is_special:
            subdir = f"{output_dir}/special_{n_cycles}_cycles"
        else:
            subdir = f"{output_dir}/{n_cycles}_cycles"
        os.makedirs(subdir, exist_ok=True)

        for i, (twists, cycles, num_removed) in enumerate(valid_twist_assignments, 1):
            if is_special:
                title = f"{config_label} SPECIAL - {n_cycles} orig cycles - Twist variant {i}/{len(valid_twist_assignments)}"
            else:
                title = f"{config_label} - {n_cycles} orig cycles - Twist variant {i}/{len(valid_twist_assignments)}"
            filename = f"{subdir}/twist_variant_{i}.png"
            draw_with_twist(combined, twists, cycles, title, filename)

        # Also save a summary
        with open(f"{subdir}/summary.txt", 'w') as f:
            if is_special:
                f.write(f"SPECIAL Matching with {n_cycles} original cycles (index {idx})\n")
            else:
                f.write(f"Matching with {n_cycles} original cycles (index {idx})\n")
            f.write("="*50 + "\n\n")

            f.write("RHS edges:\n")
            for u, v in rhs:
                f.write(f"  {u[0]}{u[1]}-{v[0]}{v[1]}\n")

            f.write(f"\nTotal valid twist assignments: {len(valid_twist_assignments)}\n\n")

            for i, (twists, cycles, num_removed) in enumerate(valid_twist_assignments, 1):
                f.write(f"Twist variant {i}:\n")
                f.write(f"  RHS twists: {twists[6:]}\n")
                f.write(f"  Cycles removed: {num_removed}\n")
                f.write("\n")

        print(f"Saved to {subdir}/")


def process_configuration(fixed_edges, config_label, output_dir):
    """Process a single configuration (6J or PILLOW)."""
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

    # Find matchings that pass untwisted test, organized by cycles
    # Separate non-special and special cases
    matchings_by_cycles = defaultdict(list)
    special_by_cycles = defaultdict(list)

    for idx, rhs, combined in valid_matchings:
        edges_no_twist = [((u, v), 0) for u, v in combined]
        after, num_cycles, _ = remove_cycles_with_zero_twist(edges_no_twist)

        if check_three_paths_structure(after):
            if all_t_same_type(rhs):
                special_by_cycles[num_cycles].append((idx, rhs, combined))
            else:
                matchings_by_cycles[num_cycles].append((idx, rhs, combined))

    print("\nNon-special matchings that pass untwisted test:")
    for n in sorted(matchings_by_cycles.keys()):
        print(f"  {n} cycles: {len(matchings_by_cycles[n])} matchings")

    print("\nSPECIAL matchings that pass untwisted test:")
    for n in sorted(special_by_cycles.keys()):
        print(f"  {n} cycles: {len(special_by_cycles[n])} matchings")

    # Process non-special matchings
    print("\n" + "="*60)
    print(f"Processing {config_label} NON-SPECIAL matchings")
    print("="*60)
    process_matchings(matchings_by_cycles, output_dir, config_label, is_special=False)

    # Process special matchings
    print("\n" + "="*60)
    print(f"Processing {config_label} SPECIAL matchings (all t's → same type)")
    print("="*60)
    process_matchings(special_by_cycles, output_dir, config_label, is_special=True)


def main():
    print("="*60)
    print("Finding matchings and their twist assignments")
    print("="*60)

    # Create main output directory
    base_dir = "twist_examples_by_matching"
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
