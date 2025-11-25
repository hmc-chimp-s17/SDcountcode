"""
Output all untwisted 1-cycle and special 0-cycle matchings to text files.
No drawings - just text output.
"""

from collections import defaultdict

# Import from main module
from SDcountnlo_twist import (
    FIXED_6J, FIXED_PILLOW, S, T, A, B, C,
    generate_rhs_matchings, check_degrees, remove_cycles_with_zero_twist,
    check_three_paths_structure, all_t_same_type
)


def process_configuration(fixed_edges, config_label, output_file):
    """Process a single configuration and output to text file."""
    all_rhs = generate_rhs_matchings()

    # Find matchings with valid degrees
    valid_matchings = []
    for idx, rhs in enumerate(all_rhs):
        combined = fixed_edges + rhs
        edges_no_twist = [((u, v), 0) for u, v in combined]
        if check_degrees(edges_no_twist):
            valid_matchings.append((idx, rhs, combined))

    # Find 1-cycle non-special matchings
    one_cycle_nonspecial = []
    for idx, rhs, combined in valid_matchings:
        if all_t_same_type(rhs):
            continue

        edges_no_twist = [((u, v), 0) for u, v in combined]
        after, num_cycles, _ = remove_cycles_with_zero_twist(edges_no_twist)

        if num_cycles == 1 and check_three_paths_structure(after):
            one_cycle_nonspecial.append((idx, rhs, combined))

    # Find 0-cycle special matchings
    zero_cycle_special = []
    for idx, rhs, combined in valid_matchings:
        if not all_t_same_type(rhs):
            continue

        edges_no_twist = [((u, v), 0) for u, v in combined]
        after, num_cycles, _ = remove_cycles_with_zero_twist(edges_no_twist)

        if num_cycles == 0 and check_three_paths_structure(after):
            # Determine which type all t's connect to
            t_type = None
            for u, v in rhs:
                if u[0] == 't':
                    t_type = v[0]
                    break
                elif v[0] == 't':
                    t_type = u[0]
                    break
            zero_cycle_special.append((idx, rhs, combined, t_type))

    # Write to file
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{config_label} - Untwisted Matchings (No Twist Parameters)\n")
        f.write("="*70 + "\n\n")

        # 1-cycle non-special
        f.write("="*70 + "\n")
        f.write("1-CYCLE NON-SPECIAL MATCHINGS\n")
        f.write("="*70 + "\n")
        f.write(f"Total: {len(one_cycle_nonspecial)} matchings\n\n")

        for i, (idx, rhs, combined) in enumerate(one_cycle_nonspecial, 1):
            f.write(f"Matching {i} (index {idx}):\n")
            f.write("  RHS edges:\n")
            for u, v in rhs:
                f.write(f"    {u[0]}{u[1]}-{v[0]}{v[1]}\n")
            f.write("\n")

        # 0-cycle special
        f.write("\n" + "="*70 + "\n")
        f.write("0-CYCLE SPECIAL MATCHINGS (all t's → same type)\n")
        f.write("="*70 + "\n")
        f.write(f"Total: {len(zero_cycle_special)} matchings\n")

        # Count by type
        by_type = defaultdict(int)
        for idx, rhs, combined, t_type in zero_cycle_special:
            by_type[t_type] += 1
        f.write(f"  Type a: {by_type.get('a', 0)}\n")
        f.write(f"  Type b: {by_type.get('b', 0)}\n")
        f.write(f"  Type c: {by_type.get('c', 0)}\n\n")

        for i, (idx, rhs, combined, t_type) in enumerate(zero_cycle_special, 1):
            f.write(f"Matching {i} (index {idx}, type {t_type}):\n")
            f.write("  RHS edges:\n")
            for u, v in rhs:
                f.write(f"    {u[0]}{u[1]}-{v[0]}{v[1]}\n")
            f.write("\n")

    print(f"Saved {config_label} to {output_file}")
    print(f"  1-cycle non-special: {len(one_cycle_nonspecial)}")
    print(f"  0-cycle special: {len(zero_cycle_special)}")


def main():
    print("="*60)
    print("Generating text files for untwisted matchings")
    print("="*60 + "\n")

    # Process 6J
    process_configuration(FIXED_6J, "6J", "untwisted_matchings_6J.txt")

    # Process PILLOW
    process_configuration(FIXED_PILLOW, "PILLOW", "untwisted_matchings_PILLOW.txt")

    # Create combined file
    print("\nCreating combined file...")
    with open("untwisted_matchings_combined.txt", 'w') as outfile:
        with open("untwisted_matchings_6J.txt", 'r') as infile:
            outfile.write(infile.read())

        outfile.write("\n\n" + "="*70 + "\n")
        outfile.write("="*70 + "\n")
        outfile.write("="*70 + "\n\n")

        with open("untwisted_matchings_PILLOW.txt", 'r') as infile:
            outfile.write(infile.read())

    print("Saved combined file to untwisted_matchings_combined.txt")
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
