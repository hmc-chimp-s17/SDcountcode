from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

# ============================================================
# GRAPH SETUP
# ============================================================

S = [("s", i) for i in (1, 2, 3)]
T = [("t", i) for i in (1, 2, 3)]
A = [("a", i) for i in (1, 2, 3)]
B = [("b", i) for i in (1, 2, 3)]
C = [("c", i) for i in (1, 2, 3)]

RHS_PORTS = T + A + B + C

# Two different FIXED edge sets
FIXED_6J = [
    (("s", 1), ("a", 1)), (("s", 2), ("b", 2)), (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)), (("b", 3), ("c", 2)), (("c", 1), ("a", 3)),
]

FIXED_PILLOW = [
    (("s", 1), ("a", 1)), (("s", 2), ("a", 2)), (("s", 3), ("c", 3)),
    (("a", 3), ("b", 1)), (("b", 2), ("c", 2)), (("b", 3), ("c", 1)),
]

# NLO path pairing: s1-t1, s2-t3, s3-t2
# Required twist for each path
NLO_PAIRING = {1: 1, 2: 3, 3: 2}
REQUIRED_TWIST = {1: 0, 2: 1, 3: 0}  # s1-t1: twist 0, s2-t3: twist 1, s3-t2: twist 0


# ============================================================
# UTILITIES
# ============================================================

def build_adj(edges_with_twist):
    """Build adjacency list (ignores parallel edges for traversal)."""
    adj = defaultdict(set)
    for (u, v), twist in edges_with_twist:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def count_degrees(edges_with_twist):
    """Count degree of each vertex (includes parallel edges)."""
    deg = defaultdict(int)
    for (u, v), twist in edges_with_twist:
        deg[u] += 1
        deg[v] += 1
    return deg


def check_degrees(edges_with_twist):
    """S/T must have degree 1, A/B/C must have degree 2."""
    deg = count_degrees(edges_with_twist)
    for v in S + T:
        if deg[v] != 1:
            return False
    for v in A + B + C:
        if deg[v] != 2:
            return False
    return True


def is_connected(edges_with_twist):
    """Check if graph forms a single connected component."""
    if not edges_with_twist:
        return True

    adj = build_adj(edges_with_twist)
    all_vertices = set()
    for (u, v), _ in edges_with_twist:
        all_vertices.add(u)
        all_vertices.add(v)

    if not all_vertices:
        return True

    # BFS from first vertex
    start = next(iter(all_vertices))
    visited = {start}
    queue = [start]

    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    return len(visited) == len(all_vertices)


# ============================================================
# CYCLE DETECTION WITH TWIST
# ============================================================

def find_cycle(edges_with_twist):
    """
    Find any cycle in the graph.
    Returns list of (edge, twist) pairs forming the cycle, or None.
    """
    # First check for parallel edges (2-cycles)
    edge_groups = defaultdict(list)
    for i, (edge, twist) in enumerate(edges_with_twist):
        key = frozenset(edge)
        edge_groups[key].append((edge, twist))
        if len(edge_groups[key]) >= 2:
            return edge_groups[key][:2]

    # Otherwise use DFS to find a cycle
    adj = defaultdict(list)
    for i, (edge, twist) in enumerate(edges_with_twist):
        u, v = edge
        adj[u].append((v, i))
        adj[v].append((u, i))

    visited = set()
    parent = {}
    parent_edge_idx = {}

    for start in adj:
        if start in visited:
            continue

        stack = [(start, None, None)]
        while stack:
            node, par, par_idx = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            parent[node] = par
            parent_edge_idx[node] = par_idx

            for nb, edge_idx in adj[node]:
                if edge_idx == par_idx:
                    continue
                if nb in visited:
                    # Found cycle - reconstruct it
                    cycle = [edges_with_twist[edge_idx]]
                    cur = node
                    while cur != nb:
                        cycle.append(edges_with_twist[parent_edge_idx[cur]])
                        cur = parent[cur]
                    return cycle
                stack.append((nb, node, edge_idx))

    return None


def remove_cycles_with_zero_twist(edges_with_twist):
    """
    Repeatedly remove cycles whose total twist = 0.
    Returns: (remaining_edges, num_cycles_removed, list_of_removed_cycles)
    """
    edges = list(edges_with_twist)
    num_removed = 0
    removed_cycles = []

    while True:
        cycle = find_cycle(edges)
        if not cycle:
            break

        # Sum twist values in cycle
        total_twist = sum(twist for edge, twist in cycle)

        # Only remove if twist sums to 0
        if total_twist != 0:
            break

        # Remove cycle edges from graph
        cycle_edges = {frozenset(edge) for edge, twist in cycle}
        edges = [(e, t) for e, t in edges if frozenset(e) not in cycle_edges]
        num_removed += 1
        removed_cycles.append(cycle)

    return edges, num_removed, removed_cycles


def get_removed_cycles(edges_with_twist):
    """Get all cycles that would be removed (for visualization)."""
    _, _, cycles = remove_cycles_with_zero_twist(edges_with_twist)
    return cycles


def has_remaining_cycles(edges_with_twist):
    """Check if graph still has cycles after removing zero-twist cycles."""
    remaining, _, _ = remove_cycles_with_zero_twist(edges_with_twist)
    # Check if remaining graph has any cycles
    return find_cycle(remaining) is not None


# ============================================================
# PATH CHECKING WITH TWIST (NLO version: s1-t1, s2-t3, s3-t2)
# ============================================================

def check_three_paths_structure(edges_with_twist):
    """
    Check if graph forms 3 disjoint paths with NLO pairing: s1→t1, s2→t3, s3→t2.
    Only checks graph structure, not twist requirements.
    Returns True if valid structure, False otherwise.
    """
    adj = defaultdict(list)
    for (u, v), twist in edges_with_twist:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()

    def get_component(start):
        """BFS to find connected component."""
        comp = {start}
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    comp.add(nb)
                    queue.append(nb)
        return comp

    # Get all components starting from s vertices
    components = []
    for s in S:
        if s not in visited:
            comp = get_component(s)
            components.append(comp)

    if len(components) != 3:
        return False

    # Check each component is a valid path with NLO pairing
    for comp in components:
        s_in_comp = [v for v in comp if v in S]
        t_in_comp = [v for v in comp if v in T]

        # Must have exactly one s and one t
        if len(s_in_comp) != 1 or len(t_in_comp) != 1:
            return False

        s_vertex = s_in_comp[0]
        t_vertex = t_in_comp[0]

        # Check NLO pairing: s1-t1, s2-t3, s3-t2
        expected_t = NLO_PAIRING[s_vertex[1]]
        if t_vertex[1] != expected_t:
            return False

        # Check path structure (all internal vertices have degree 2)
        for v in comp:
            deg = len([nb for nb in adj[v] if nb in comp])
            if v in S or v in T:
                if deg != 1:
                    return False
            else:
                if deg != 2:
                    return False

    return True


def check_three_paths_with_twist(edges_with_twist):
    """
    Check if graph forms 3 disjoint paths: s1→t1 (twist 0), s2→t3 (twist 1), s3→t2 (twist 0).
    Returns True and path info if valid, False otherwise.
    """
    adj = defaultdict(list)
    for (u, v), twist in edges_with_twist:
        adj[u].append((v, twist))
        adj[v].append((u, twist))

    visited = set()

    def get_component_with_twist(start):
        """BFS to find connected component and compute total twist from start."""
        comp = {start}
        queue = [(start, 0)]  # (node, cumulative_twist)
        visited.add(start)
        twist_to_node = {start: 0}

        while queue:
            node, cum_twist = queue.pop(0)
            for nb, edge_twist in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    comp.add(nb)
                    new_twist = cum_twist + edge_twist
                    twist_to_node[nb] = new_twist
                    queue.append((nb, new_twist))

        return comp, twist_to_node

    # Get all components starting from s vertices
    components = []
    for s in S:
        if s not in visited:
            comp, twist_to_node = get_component_with_twist(s)
            components.append((comp, twist_to_node))

    if len(components) != 3:
        return False, None

    # Check each component is a valid path with NLO pairing and correct twist
    path_info = []
    for comp, twist_to_node in components:
        s_in_comp = [v for v in comp if v in S]
        t_in_comp = [v for v in comp if v in T]

        # Must have exactly one s and one t
        if len(s_in_comp) != 1 or len(t_in_comp) != 1:
            return False, None

        s_vertex = s_in_comp[0]
        t_vertex = t_in_comp[0]
        s_idx = s_vertex[1]
        t_idx = t_vertex[1]

        # Check NLO pairing: s1-t1, s2-t3, s3-t2
        if NLO_PAIRING[s_idx] != t_idx:
            return False, None

        # Check twist requirement
        path_twist = twist_to_node[t_vertex]
        required = REQUIRED_TWIST[s_idx]
        if path_twist != required:
            return False, None

        # All vertices in path must have degree 1 or 2
        degrees = [len([nb for nb, tw in adj[v] if nb in comp]) for v in comp]
        if not all(d in (1, 2) for d in degrees):
            return False, None
        if sum(1 for d in degrees if d == 1) != 2:
            return False, None

        path_info.append((s_idx, t_idx, path_twist))

    return True, path_info


# ============================================================
# MATCHING GENERATION WITH TWIST
# ============================================================

def generate_rhs_matchings():
    """Generate all perfect matchings on RHS (no T-T edges)."""
    matchings = []

    def recurse(remaining, current):
        if len(current) == 6:
            matchings.append(current[:])
            return
        if not remaining:
            return

        v = remaining[0]
        for i, w in enumerate(remaining[1:], 1):
            # No T-T edges allowed
            if v in T and w in T:
                continue
            new_remaining = [remaining[j] for j in range(len(remaining)) if j not in (0, i)]
            recurse(new_remaining, current + [(v, w)])

    recurse(RHS_PORTS, [])
    return matchings


# ============================================================
# T-CONNECTION CHECK
# ============================================================

def all_t_same_type(rhs):
    """Check if all t vertices connect to same intermediate type (a/b/c)."""
    t_to_type = {}

    for u, v in rhs:
        if u[0] == 't' and v[0] in ['a', 'b', 'c']:
            t_to_type[u] = v[0]
        elif v[0] == 't' and u[0] in ['a', 'b', 'c']:
            t_to_type[v] = u[0]

    if len(t_to_type) == 3:
        types = set(t_to_type.values())
        if len(types) == 1:
            return types.pop()
    return None


# ============================================================
# MAIN COUNTING WITH TWIST
# ============================================================

def check_three_paths_basic(edges):
    """Basic check if edges form 3 disjoint paths (without twist consideration)."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    visited = set()

    def get_component(start):
        comp = {start}
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    comp.add(nb)
                    queue.append(nb)
        return comp

    components = [get_component(s) for s in S if s not in visited]

    if len(components) != 3:
        return False

    for comp in components:
        s_in_comp = [v for v in comp if v in S]
        t_in_comp = [v for v in comp if v in T]

        if len(s_in_comp) != 1 or len(t_in_comp) != 1:
            return False

        s_idx = s_in_comp[0][1]
        t_idx = t_in_comp[0][1]
        if NLO_PAIRING[s_idx] != t_idx:
            return False

        degrees = [len([nb for nb in adj[v] if nb in comp]) for v in comp]
        if not all(d in (1, 2) for d in degrees):
            return False
        if sum(1 for d in degrees if d == 1) != 2:
            return False

    return True


def count_all_with_twist(fixed_edges, label):
    """Count all valid configurations with twist assignments."""
    print(f"\n{'='*60}")
    print(f"Analyzing {label} configuration (NLO with twist)")
    print(f"Path requirements: s1-t1 (twist 0), s2-t3 (twist 1), s3-t2 (twist 0)")
    print(f"{'='*60}")
    print("Generating all RHS matchings...")
    all_rhs = generate_rhs_matchings()
    print(f"Total matchings: {len(all_rhs)}")

    # First pass: find matchings with valid degrees
    print("First pass: finding matchings with valid degrees...")
    valid_matchings = []
    for idx, rhs in enumerate(all_rhs):
        combined = fixed_edges + rhs
        edges_no_twist = [((u, v), 0) for u, v in combined]
        if not check_degrees(edges_no_twist):
            continue
        valid_matchings.append((rhs, combined))

    print(f"Found {len(valid_matchings)} matchings with valid degrees")

    valid_configs = []
    by_cycles = defaultdict(list)

    # First: find matchings that pass WITHOUT twist (original test)
    # These give 3 disconnected paths after removing cycles
    matchings_pass_no_twist = defaultdict(list)  # cycles -> list of (idx, rhs, combined)
    special_no_twist = defaultdict(list)  # cycles -> list of (idx, rhs, combined) for t_same_type
    for idx, (rhs, combined) in enumerate(valid_matchings):
        edges_no_twist = [((u, v), 0) for u, v in combined]
        after, num_cycles, _ = remove_cycles_with_zero_twist(edges_no_twist)
        # Check if gives 3 disconnected paths (structure only, ignoring twist requirements)
        if check_three_paths_structure(after):
            matchings_pass_no_twist[num_cycles].append((idx, rhs, combined))
            # Track special case
            if all_t_same_type(rhs):
                special_no_twist[num_cycles].append((idx, rhs, combined))

    print(f"\nOriginal matchings that pass without twist:")
    for n in sorted(matchings_pass_no_twist.keys()):
        print(f"  {n} cycles: {len(matchings_pass_no_twist[n])} matchings")
    total_no_twist = sum(len(v) for v in matchings_pass_no_twist.values())
    print(f"  Total: {total_no_twist} matchings")

    # Track which original matchings also pass WITH twist
    # organized by number of cycles in the original (no-twist) case
    matchings_that_pass_with_twist = defaultdict(set)  # original_cycles -> set of matching indices
    special_pass_with_twist = defaultdict(set)  # original_cycles -> set of matching indices for t_same_type

    total_checked = 0

    print("\nSecond pass: checking twist assignments (only RHS edges have twist)...")
    for idx, (rhs, combined) in enumerate(valid_matchings):
        if (idx + 1) % 50 == 0:
            print(f"  Checked {idx + 1}/{len(valid_matchings)} matchings...")

        # Generate all possible twist assignments for RHS edges only (0, +1, -1)
        # Fixed edges (first 6) have twist 0
        num_fixed = 6
        num_rhs = 6
        twist_values = [0, 1, -1]

        for rhs_twists in product(twist_values, repeat=num_rhs):
            total_checked += 1

            # Create edges with twist: fixed edges have twist 0, RHS edges have assigned twist
            fixed_with_twist = [((u, v), 0) for u, v in combined[:num_fixed]]
            rhs_with_twist = [((u, v), t) for (u, v), t in zip(combined[num_fixed:], rhs_twists)]
            edges_with_twist = fixed_with_twist + rhs_with_twist

            # Remove cycles with twist sum = 0
            after, num_cycles, removed_cycles = remove_cycles_with_zero_twist(edges_with_twist)

            # Check if there are remaining cycles that couldn't be removed
            if find_cycle(after) is not None:
                continue  # Skip if there are non-removable cycles

            # Check if we get valid paths with correct twists
            valid, path_info = check_three_paths_with_twist(after)

            if valid:  # Include all cases (0 or more cycles removed)
                cycles = get_removed_cycles(edges_with_twist)
                # Full twist list: 0s for fixed edges, then RHS twists
                full_twists = (0,) * num_fixed + rhs_twists
                # Check if all t's connect to same type
                t_same = all_t_same_type(rhs)
                config = {
                    'rhs': rhs,
                    'combined': combined,
                    'twists': full_twists,
                    'edges_with_twist': edges_with_twist,
                    'num_cycles': num_cycles,
                    'cycles': cycles,
                    'path_info': path_info,
                    'remaining': after,
                    't_same_type': t_same
                }
                valid_configs.append(config)
                by_cycles[num_cycles].append(config)

                # Track that this original matching passes WITH twist
                # Find which original cycle count this matching had
                for orig_cycles, matching_list in matchings_pass_no_twist.items():
                    for m_idx, m_rhs, m_combined in matching_list:
                        if m_idx == idx:
                            matchings_that_pass_with_twist[orig_cycles].add(idx)
                            # Also track special case
                            for s_cycles, s_matching_list in special_no_twist.items():
                                for s_m_idx, s_m_rhs, s_m_combined in s_matching_list:
                                    if s_m_idx == idx:
                                        special_pass_with_twist[s_cycles].add(idx)
                                        break
                            break

    print(f"\nTotal twist assignments checked: {total_checked}")

    # Print original matchings that also pass WITH twist
    print(f"\nOriginal matchings that also pass WITH twist (by original cycles):")
    for n in sorted(matchings_that_pass_with_twist.keys()):
        orig_count = len(matchings_pass_no_twist[n])
        pass_count = len(matchings_that_pass_with_twist[n])
        print(f"  {n} cycles: {pass_count}/{orig_count} matchings")

    # Total
    total_pass_with_twist = sum(len(s) for s in matchings_that_pass_with_twist.values())
    print(f"  Total: {total_pass_with_twist}/{total_no_twist} matchings")

    # Print SPECIAL case comparison
    print(f"\nSPECIAL (All t's → same type) - Original matchings that pass:")
    for n in sorted(special_no_twist.keys()):
        print(f"  {n} cycles: {len(special_no_twist[n])} matchings")
    total_special_no_twist = sum(len(v) for v in special_no_twist.values())
    print(f"  Total: {total_special_no_twist} matchings")

    print(f"\nSPECIAL (All t's → same type) - Also pass WITH twist:")
    for n in sorted(special_pass_with_twist.keys()):
        orig_count = len(special_no_twist[n])
        pass_count = len(special_pass_with_twist[n])
        print(f"  {n} cycles: {pass_count}/{orig_count} matchings")
    total_special_with_twist = sum(len(s) for s in special_pass_with_twist.values())
    print(f"  Total: {total_special_with_twist}/{total_special_no_twist} matchings")

    return valid_configs, by_cycles, matchings_pass_no_twist, matchings_that_pass_with_twist, special_no_twist, special_pass_with_twist


# ============================================================
# VISUALIZATION
# ============================================================

def draw_with_twist(config, title, filename):
    """Draw graph with highlighted cycles and twist values."""
    fig, ax = plt.subplots(figsize=(14, 10))

    combined = config['combined']
    twists = config['twists']
    cycles = config['cycles']

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
            (u1, v1), t1 = cyc[0]
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


# ============================================================
# MAIN
# ============================================================

def print_results(label, valid_configs, by_cycles):
    """Print results for one configuration."""
    total = len(valid_configs)

    print("\n" + "="*60)
    print(f"{label} (NLO with twist) - TOTAL VALID: {total}")
    print("="*60)

    print("\nBy cycles removed:")
    for n in sorted(by_cycles.keys()):
        print(f"  {n} cycles: {len(by_cycles[n])}")

    # Special case: all t's connect to same type
    t_same_configs = [c for c in valid_configs if c['t_same_type']]
    print("\n" + "="*60)
    print("SPECIAL: All t's → same type (a, b, or c)")
    print("="*60)
    print(f"Total: {len(t_same_configs)} configs")

    # Breakdown by cycle count
    t_cycle_dist = defaultdict(int)
    for c in t_same_configs:
        t_cycle_dist[c['num_cycles']] += 1

    print("\nCycle breakdown:")
    for n in sorted(t_cycle_dist.keys()):
        print(f"  {n} cycles: {t_cycle_dist[n]}")


def save_results_to_file(all_results, summary_filename, details_filename):
    """Save results: summary to one file, detailed configurations to another."""
    # Write summary file
    with open(summary_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NLO Configuration with Twist - Summary\n")
        f.write("Path requirements: s1-t1 (twist 0), s2-t3 (twist 1), s3-t2 (twist 0)\n")
        f.write("="*60 + "\n\n")

        for label, (valid_configs, by_cycles, no_twist, with_twist, special_no, special_with) in all_results.items():
            total = len(valid_configs)

            f.write(f"{label}:\n")
            f.write(f"  Total valid configurations (with twist): {total}\n")
            f.write(f"  By cycles removed (with twist):\n")
            for n in sorted(by_cycles.keys()):
                f.write(f"    {n} cycles: {len(by_cycles[n])}\n")

            # Original matchings without twist
            f.write(f"\n  Original matchings (without twist):\n")
            for n in sorted(no_twist.keys()):
                f.write(f"    {n} cycles: {len(no_twist[n])} matchings\n")
            total_no_twist = sum(len(v) for v in no_twist.values())
            f.write(f"    Total: {total_no_twist} matchings\n")

            # Matchings that also pass WITH twist
            f.write(f"\n  Original matchings that also pass WITH twist (by original cycles):\n")
            for n in sorted(with_twist.keys()):
                orig_count = len(no_twist[n])
                pass_count = len(with_twist[n])
                f.write(f"    {n} cycles: {pass_count}/{orig_count} matchings\n")
            total_with_twist = sum(len(s) for s in with_twist.values())
            f.write(f"    Total: {total_with_twist}/{total_no_twist} matchings\n")

            # Special case: all t's connect to same type (configs with twist)
            t_same_configs = [c for c in valid_configs if c['t_same_type']]
            f.write(f"\n  SPECIAL: All t's → same type (configs with twist): {len(t_same_configs)} configs\n")
            t_cycle_dist = defaultdict(int)
            for c in t_same_configs:
                t_cycle_dist[c['num_cycles']] += 1
            for n in sorted(t_cycle_dist.keys()):
                f.write(f"    {n} cycles: {t_cycle_dist[n]}\n")

            # Special case: without twist vs with twist comparison
            f.write(f"\n  SPECIAL (All t's → same type) - Original matchings:\n")
            for n in sorted(special_no.keys()):
                f.write(f"    {n} cycles: {len(special_no[n])} matchings\n")
            total_special_no = sum(len(v) for v in special_no.values())
            f.write(f"    Total: {total_special_no} matchings\n")

            f.write(f"\n  SPECIAL (All t's → same type) - Also pass WITH twist:\n")
            for n in sorted(special_with.keys()):
                orig_count = len(special_no[n])
                pass_count = len(special_with[n])
                f.write(f"    {n} cycles: {pass_count}/{orig_count} matchings\n")
            total_special_with = sum(len(s) for s in special_with.values())
            f.write(f"    Total: {total_special_with}/{total_special_no} matchings\n")
            f.write("\n")

    print(f"Summary saved to {summary_filename}")

    # Write detailed configurations file
    with open(details_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("NLO Configuration with Twist - Detailed Configurations\n")
        f.write("Path requirements: s1-t1 (twist 0), s2-t3 (twist 1), s3-t2 (twist 0)\n")
        f.write("="*60 + "\n\n")

        for label, (valid_configs, by_cycles, no_twist, with_twist, special_no, special_with) in all_results.items():
            f.write("="*60 + "\n")
            f.write(f"{label} CONFIGURATIONS\n")
            f.write("="*60 + "\n\n")

            for i, config in enumerate(valid_configs, 1):
                f.write(f"Configuration {i}:\n")
                f.write(f"  Cycles removed: {config['num_cycles']}\n")
                f.write(f"  Edges with twist:\n")
                for j, ((u, v), twist) in enumerate(zip(config['combined'], config['twists'])):
                    twist_str = f"{twist:+d}" if twist != 0 else " 0"
                    f.write(f"    {u[0]}{u[1]}-{v[0]}{v[1]}: {twist_str}\n")
                f.write("\n")

    print(f"Details saved to {details_filename}")


def draw_results(label, valid_configs, by_cycles, base_dir):
    """Draw one example for each number of cycles, plus special cases."""
    print("\n" + "="*60)
    print(f"DRAWING {label} (NLO with twist)")
    print("="*60)

    output_dir = f"{base_dir}_nlo_twist_examples"
    os.makedirs(output_dir, exist_ok=True)

    # Draw one example for each number of cycles
    print("\nDrawing one example for each cycle count...")
    for n_cycles in sorted(by_cycles.keys()):
        config = by_cycles[n_cycles][0]  # First example with this cycle count
        title = f"{label} NLO Twist - {n_cycles} Cycle(s) Removed"
        filename = f"{output_dir}/{n_cycles}_cycles_example.png"
        draw_with_twist(config, title, filename)

    # Draw one example for each cycle count in SPECIAL case (all t's → same type)
    print("\nDrawing SPECIAL case examples (all t's → same type)...")
    special_configs = [c for c in valid_configs if c['t_same_type']]

    # Group special configs by cycle count
    special_by_cycles = defaultdict(list)
    for c in special_configs:
        special_by_cycles[c['num_cycles']].append(c)

    for n_cycles in sorted(special_by_cycles.keys()):
        config = special_by_cycles[n_cycles][0]  # First special example with this cycle count
        t_type = config['t_same_type']
        title = f"{label} SPECIAL (all t→{t_type}) - {n_cycles} Cycle(s) Removed"
        filename = f"{output_dir}/special_{n_cycles}_cycles_example.png"
        draw_with_twist(config, title, filename)

    print("\n" + "="*60)
    print(f"Done! Saved to {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("6j Symbol Configuration Counter - NLO with Twist")
    print("Path requirements: s1-t1 (twist 0), s2-t3 (twist 1), s3-t2 (twist 0)")
    print("="*60)

    # Analyze 6J configuration
    valid_6j, by_cycles_6j, no_twist_6j, with_twist_6j, special_no_6j, special_with_6j = count_all_with_twist(FIXED_6J, "6J")
    print_results("6J", valid_6j, by_cycles_6j)

    # Analyze PILLOW configuration
    valid_pillow, by_cycles_pillow, no_twist_pillow, with_twist_pillow, special_no_pillow, special_with_pillow = count_all_with_twist(FIXED_PILLOW, "PILLOW")
    print_results("PILLOW", valid_pillow, by_cycles_pillow)

    # Save results to separate files
    all_results = {
        "6J": (valid_6j, by_cycles_6j, no_twist_6j, with_twist_6j, special_no_6j, special_with_6j),
        "PILLOW": (valid_pillow, by_cycles_pillow, no_twist_pillow, with_twist_pillow, special_no_pillow, special_with_pillow)
    }
    save_results_to_file(all_results, "nlo_twist_summary.txt", "nlo_twist_details.txt")

    # Draw examples
    if valid_6j:
        draw_results("6J", valid_6j, by_cycles_6j, "6j")
    if valid_pillow:
        draw_results("PILLOW", valid_pillow, by_cycles_pillow, "pillow")
