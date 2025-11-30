"""
SDcountnlo_v4.py - Split a-c Edges After Merging

Find matchings where ANY of the three paths (s1-t1, s2-t3, s3-t2) has at least one cycle
after merging a,b,c vertices AND splitting a-c edges into a-b and b-c.
Self-loops are NOT counted as cycles.

Steps:
1. Find non-special 1-cycle and special 0-cycle matchings with NLO pairing (s1-t1, s2-t3, s3-t2)
2. Remove cycles
3. Extract ALL three paths: s1-t1, s2-t3, s3-t2
4. Merge vertices: a1,a2,a3 → a; b1,b2,b3 → b; c1,c2,c3 → c
5. Split a-c edges: each a-c edge becomes a-b and b-c
6. Count cycles in each path (excluding self-loops)
7. Keep only matchings where ANY path has >=1 cycles
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


def extract_path(edges, s_vertex):
    """Get all edges in the path starting from s_vertex."""
    adj = build_adj(edges)

    # BFS from s_vertex to find component
    component = {s_vertex}
    queue = [s_vertex]
    visited = {s_vertex}

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


def split_ac_edges(edges):
    """Split any a-c edge into a-b and b-c."""
    result = []
    for u, v in edges:
        # Check if this is an a-c edge (in either direction)
        if (u == ('a',) and v == ('c',)) or (u == ('c',) and v == ('a',)):
            # Split into a-b and b-c
            result.append((('a',), ('b',)))
            result.append((('b',), ('c',)))
        else:
            # Keep edge as is
            result.append((u, v))
    return result


def find_cycle(edges):
    """Find one cycle in the graph."""
    if not edges:
        return None

    # Check self-loops (but don't return them - skip to next)
    non_selfloop_edges = [(u, v) for u, v in edges if u != v]

    if not non_selfloop_edges:
        return None

    # Check parallel edges (2-cycles)
    edge_count = defaultdict(int)
    for u, v in non_selfloop_edges:
        edge_count[frozenset([u, v])] += 1
    for endpoints, count in edge_count.items():
        if count >= 2:
            u, v = tuple(endpoints)
            return [(u, v), (v, u)]

    # Find longer cycles with DFS
    adj = build_adj(non_selfloop_edges)
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


def find_matchings_with_cycles_in_any_path(fixed_edges, label):
    """Find all matchings where ANY path has >=1 cycles after merging and splitting a-c."""
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

    # Check which have >=1 cycles in ANY path after merging and splitting (excluding self-loops)
    results = []

    print(f"\nChecking all three paths after merging and splitting a-c edges...")

    for idx, rhs, combined, after_removal, cycles_removed in one_cycle_nonspecial:
        # Extract all three paths
        s1_t1_path = extract_path(after_removal, ('s', 1))
        s2_t3_path = extract_path(after_removal, ('s', 2))
        s3_t2_path = extract_path(after_removal, ('s', 3))

        # Merge and split for each path
        merged_s1_t1 = merge_vertices(s1_t1_path)
        merged_s2_t3 = merge_vertices(s2_t3_path)
        merged_s3_t2 = merge_vertices(s3_t2_path)

        split_s1_t1 = split_ac_edges(merged_s1_t1)
        split_s2_t3 = split_ac_edges(merged_s2_t3)
        split_s3_t2 = split_ac_edges(merged_s3_t2)

        cycles_s1_t1 = count_cycles(split_s1_t1)
        cycles_s2_t3 = count_cycles(split_s2_t3)
        cycles_s3_t2 = count_cycles(split_s3_t2)

        # Check if ANY path has >=1 cycles
        total_cycles = cycles_s1_t1 + cycles_s2_t3 + cycles_s3_t2

        if total_cycles >= 1:
            results.append({
                'type': '1-cycle non-special',
                'idx': idx,
                'rhs': rhs,
                'combined': combined,
                'after_removal': after_removal,
                'cycles_removed': cycles_removed,
                's1_t1_path': s1_t1_path,
                's2_t3_path': s2_t3_path,
                's3_t2_path': s3_t2_path,
                'merged_s1_t1': merged_s1_t1,
                'merged_s2_t3': merged_s2_t3,
                'merged_s3_t2': merged_s3_t2,
                'split_s1_t1': split_s1_t1,
                'split_s2_t3': split_s2_t3,
                'split_s3_t2': split_s3_t2,
                'cycles_s1_t1': cycles_s1_t1,
                'cycles_s2_t3': cycles_s2_t3,
                'cycles_s3_t2': cycles_s3_t2,
                'total_cycles': total_cycles,
                't_type': None
            })

    for idx, rhs, combined, after_removal, cycles_removed, t_type in zero_cycle_special:
        # Extract all three paths
        s1_t1_path = extract_path(after_removal, ('s', 1))
        s2_t3_path = extract_path(after_removal, ('s', 2))
        s3_t2_path = extract_path(after_removal, ('s', 3))

        # Merge and split for each path
        merged_s1_t1 = merge_vertices(s1_t1_path)
        merged_s2_t3 = merge_vertices(s2_t3_path)
        merged_s3_t2 = merge_vertices(s3_t2_path)

        split_s1_t1 = split_ac_edges(merged_s1_t1)
        split_s2_t3 = split_ac_edges(merged_s2_t3)
        split_s3_t2 = split_ac_edges(merged_s3_t2)

        cycles_s1_t1 = count_cycles(split_s1_t1)
        cycles_s2_t3 = count_cycles(split_s2_t3)
        cycles_s3_t2 = count_cycles(split_s3_t2)

        # Check if ANY path has >=1 cycles
        total_cycles = cycles_s1_t1 + cycles_s2_t3 + cycles_s3_t2

        if total_cycles >= 1:
            results.append({
                'type': '0-cycle special',
                'idx': idx,
                'rhs': rhs,
                'combined': combined,
                'after_removal': after_removal,
                'cycles_removed': cycles_removed,
                's1_t1_path': s1_t1_path,
                's2_t3_path': s2_t3_path,
                's3_t2_path': s3_t2_path,
                'merged_s1_t1': merged_s1_t1,
                'merged_s2_t3': merged_s2_t3,
                'merged_s3_t2': merged_s3_t2,
                'split_s1_t1': split_s1_t1,
                'split_s2_t3': split_s2_t3,
                'split_s3_t2': split_s3_t2,
                'cycles_s1_t1': cycles_s1_t1,
                'cycles_s2_t3': cycles_s2_t3,
                'cycles_s3_t2': cycles_s3_t2,
                'total_cycles': total_cycles,
                't_type': t_type
            })

    # Count by type
    one_cycle_count = sum(1 for r in results if r['type'] == '1-cycle non-special')
    zero_cycle_count = sum(1 for r in results if r['type'] == '0-cycle special')

    print(f"\nResults (ANY path with >=1 cycles after merging+splitting, excluding self-loops):")
    print(f"  1-cycle non-special: {one_cycle_count}")
    print(f"  0-cycle special: {zero_cycle_count}")
    print(f"  Total: {len(results)}")

    return results


def save_results(label, results, output_dir):
    """Save results and summary."""
    os.makedirs(output_dir, exist_ok=True)

    # Text summary
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"{label} - ANY Path with >=1 Cycles After Merging+Splitting a-c\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total: {len(results)} matchings\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"Matching {i} - {r['type']} (idx {r['idx']})\n")
            if r['t_type']:
                f.write(f"  Type: all t→{r['t_type']}\n")
            f.write(f"  Cycles in s1-t1: {r['cycles_s1_t1']}\n")
            f.write(f"  Cycles in s2-t3: {r['cycles_s2_t3']}\n")
            f.write(f"  Cycles in s3-t2: {r['cycles_s3_t2']}\n")
            f.write(f"  Total cycles: {r['total_cycles']}\n")
            f.write(f"  RHS edges: ")
            f.write(", ".join([f"{u[0]}{u[1]}-{v[0]}{v[1]}" for u, v in r['rhs']]))
            f.write("\n\n")

    print(f"\nResults saved to {output_dir}/")


def main():
    print("="*60)
    print("SDcountnlo_v4: ANY Path Cycle Analysis (with a-c splitting)")
    print("="*60)

    # Process 6J
    results_6j = find_matchings_with_cycles_in_any_path(FIXED_6J, "6J")
    save_results("6J", results_6j, "nlo_v4_results/6J")

    # Process PILLOW
    results_pillow = find_matchings_with_cycles_in_any_path(FIXED_PILLOW, "PILLOW")
    save_results("PILLOW", results_pillow, "nlo_v4_results/PILLOW")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
