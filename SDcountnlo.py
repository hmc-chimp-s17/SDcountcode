from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

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
NLO_PAIRING = {1: 1, 2: 3, 3: 2}


# ============================================================
# UTILITIES
# ============================================================

def build_adj(edges):
    """Build adjacency list (ignores parallel edges for traversal)."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def count_degrees(edges):
    """Count degree of each vertex (includes parallel edges)."""
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def check_degrees(edges):
    """S/T must have degree 1, A/B/C must have degree 2."""
    deg = count_degrees(edges)
    for v in S + T:
        if deg[v] != 1:
            return False
    for v in A + B + C:
        if deg[v] != 2:
            return False
    return True


# ============================================================
# CYCLE DETECTION
# ============================================================

def find_parallel_edges(edges):
    """Find parallel edges (same endpoints)."""
    edge_count = defaultdict(int)
    for u, v in edges:
        edge_count[frozenset([u, v])] += 1

    for endpoints, count in edge_count.items():
        if count >= 2:
            u, v = tuple(endpoints)
            return [(u, v), (v, u)]
    return None


def find_cycle_dfs(edges):
    """Find any cycle using DFS."""
    adj = build_adj(edges)
    visited, parent = set(), {}

    def dfs(node, par):
        visited.add(node)
        for nb in adj[node]:
            if nb == par:
                continue
            if nb in visited:
                # Reconstruct cycle
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


def find_cycle(edges):
    """Find any cycle (parallel edges have priority)."""
    # Check parallel edges first (form 2-cycles)
    parallel = find_parallel_edges(edges)
    if parallel:
        return parallel
    # Otherwise find any cycle via DFS
    return find_cycle_dfs(edges)


def remove_cycles(edges):
    """Remove all cycles and count how many."""
    edges = [e for e in edges]  # Copy
    num_cycles = 0

    while True:
        cycle = find_cycle(edges)
        if not cycle:
            break
        # Remove cycle edges
        cycle_set = {frozenset(e) for e in cycle}
        edges = [e for e in edges if frozenset(e) not in cycle_set]
        num_cycles += 1

    return edges, num_cycles


def get_all_cycles(edges):
    """Get list of all cycles (for visualization)."""
    edges = [e for e in edges]
    cycles = []

    while True:
        cycle = find_cycle(edges)
        if not cycle:
            break
        cycles.append(cycle)
        cycle_set = {frozenset(e) for e in cycle}
        edges = [e for e in edges if frozenset(e) not in cycle_set]

    return cycles


# ============================================================
# PATH CHECKING (NLO version: s1-t1, s2-t3, s3-t2)
# ============================================================

def check_three_paths(edges):
    """
    Check if graph forms 3 disjoint paths: s1→t1, s2→t3, s3→t2.
    Each path must be simple (all vertices have degree 1 or 2, exactly 2 endpoints).
    """
    adj = build_adj(edges)
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
    components = [get_component(s) for s in S if s not in visited]

    if len(components) != 3:
        return False

    # Check each component is a valid path with NLO pairing
    for comp in components:
        s_in_comp = [v for v in comp if v in S]
        t_in_comp = [v for v in comp if v in T]

        # Must have exactly one s and one t
        if len(s_in_comp) != 1 or len(t_in_comp) != 1:
            return False

        # Check NLO pairing: s1-t1, s2-t3, s3-t2
        s_idx = s_in_comp[0][1]
        t_idx = t_in_comp[0][1]
        if NLO_PAIRING[s_idx] != t_idx:
            return False

        # All vertices in path must have degree 1 or 2
        # Exactly 2 vertices have degree 1 (the endpoints)
        degrees = [len([nb for nb in adj[v] if nb in comp]) for v in comp]
        if not all(d in (1, 2) for d in degrees):
            return False
        if sum(1 for d in degrees if d == 1) != 2:
            return False

    return True


# ============================================================
# MATCHING GENERATION
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
# MAIN COUNTING
# ============================================================

def count_all(fixed_edges, label):
    """Count all valid configurations and categorize them."""
    print(f"\n{'='*60}")
    print(f"Analyzing {label} configuration (NLO: s1-t1, s2-t3, s3-t2)")
    print(f"{'='*60}")
    print("Generating all RHS matchings...")
    all_rhs = generate_rhs_matchings()
    print(f"Total: {len(all_rhs)}")

    by_cycles = defaultdict(list)
    t_same_type = []

    for idx, rhs in enumerate(all_rhs):
        if (idx + 1) % 1000 == 0:
            print(f"  Checked {idx + 1}/{len(all_rhs)}...")

        combined = fixed_edges + rhs

        if not check_degrees(combined):
            continue

        after, num_cycles = remove_cycles(combined)

        if check_three_paths(after):
            cycles = get_all_cycles(combined)
            by_cycles[num_cycles].append((rhs, combined, cycles))

            # Check special case
            if all_t_same_type(rhs):
                t_same_type.append((rhs, combined, cycles, num_cycles))

    return by_cycles, t_same_type


# ============================================================
# VISUALIZATION
# ============================================================

def draw(combined, cycles, title, filename):
    """Draw graph with highlighted cycles."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Positions (same as original SDcount.py)
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
        for edge in cycle:
            cycle_edges.append((frozenset(edge), i))

    # Count parallel edges
    edge_counts = defaultdict(int)
    for u, v in combined:
        edge_counts[frozenset([u, v])] += 1

    # Draw edges
    edge_drawn = defaultdict(int)
    for e in combined:
        u, v = tuple(e) if isinstance(e, frozenset) else e
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

            if cycle_idx is not None:
                ax.text(ctrl_x, ctrl_y, f"{u[0]}{u[1]}-{v[0]}{v[1]}", fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='pink', alpha=0.9),
                       ha='center', va='center', zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            if cycle_idx is not None:
                ax.text(mid_x, mid_y, f"{u[0]}{u[1]}-{v[0]}{v[1]}", fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='pink', alpha=0.9),
                       ha='center', va='center', zorder=2)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add cycle information
    cycle_text = ""
    for i, cyc in enumerate(cycles):
        if len(cyc) == 2:
            u, v = cyc[0]
            cycle_text += f"Cycle {i+1} (2-cycle): {u[0]}{u[1]} ⇄ {v[0]}{v[1]}\n"
        else:
            vertices = [u for u, _ in cyc]
            vertices.append(vertices[0])
            cycle_text += f"Cycle {i+1} ({len(cyc)}-cycle): {' → '.join([f'{v[0]}{v[1]}' for v in vertices])}\n"

    ax.text(0.02, 0.98, cycle_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def print_results(label, by_cycles, t_same_type):
    """Print results for one configuration."""
    total = sum(len(v) for v in by_cycles.values())

    print("\n" + "="*60)
    print(f"{label} (NLO) - TOTAL VALID: {total}")
    print("="*60)

    print("\nBy cycles removed:")
    for n in sorted(by_cycles.keys()):
        print(f"  {n} cycles: {len(by_cycles[n])} configs")

    print("\n" + "="*60)
    print("SPECIAL: All t's → same type (a, b, or c)")
    print("="*60)
    print(f"Total: {len(t_same_type)} configs")

    t_cycle_dist = defaultdict(int)
    for _, _, _, nc in t_same_type:
        t_cycle_dist[nc] += 1

    print("\nCycle breakdown:")
    for n in sorted(t_cycle_dist.keys()):
        print(f"  {n} cycles: {t_cycle_dist[n]} configs")


def draw_results(label, by_cycles, t_same_type, base_dir):
    """Draw all examples for one configuration."""
    print("\n" + "="*60)
    print(f"DRAWING {label} (NLO)")
    print("="*60)

    cycle_dir = f"{base_dir}_nlo_cycle_examples"
    t_dir = f"{base_dir}_nlo_t_same_type_all"
    os.makedirs(cycle_dir, exist_ok=True)
    os.makedirs(t_dir, exist_ok=True)

    # Draw special cases with >= 1 cycle (all t's connect to same type)
    t_same_with_cycles = [(rhs, combined, cycles, nc) for rhs, combined, cycles, nc in t_same_type if nc >= 1]
    if t_same_with_cycles:
        print(f"\nDrawing {len(t_same_with_cycles)} t-same-type examples with ≥1 cycle...")
        for i, (rhs, combined, cycles, nc) in enumerate(t_same_with_cycles):
            t_type = all_t_same_type(rhs)
            title = f"{label} NLO - {nc} Cycle(s) (all t→{t_type}) - Example {i+1}/{len(t_same_with_cycles)}"
            draw(combined, cycles, title, f"{cycle_dir}/t_same_{nc}cycles_{i+1}.png")

    # Draw 2-cycle examples (general, not special)
    if 2 in by_cycles:
        print(f"\nDrawing {len(by_cycles[2])} examples with 2 cycles...")
        for i, (rhs, combined, cycles) in enumerate(by_cycles[2]):
            title = f"{label} NLO - 2 Cycles - Example {i+1}/{len(by_cycles[2])}"
            draw(combined, cycles, title, f"{cycle_dir}/2cycles_{i+1}.png")

    # Draw 3-cycle examples (general, not special)
    if 3 in by_cycles:
        print(f"\nDrawing {len(by_cycles[3])} example(s) with 3 cycles...")
        for i, (rhs, combined, cycles) in enumerate(by_cycles[3]):
            title = f"{label} NLO - 3 Cycles - Example {i+1}/{len(by_cycles[3])}"
            draw(combined, cycles, title, f"{cycle_dir}/3cycles_{i+1}.png")

    # Draw all t-same-type examples (to separate folder for reference)
    print(f"\nDrawing all {len(t_same_type)} t-same-type examples to {t_dir}/...")
    for i, (rhs, combined, cycles, nc) in enumerate(t_same_type):
        t_type = all_t_same_type(rhs)
        title = f"{label} NLO - All t→{t_type}, {nc} cycles - #{i+1}/{len(t_same_type)}"
        draw(combined, cycles, title, f"{t_dir}/ex{i+1}.png")

    print("\n" + "="*60)
    print(f"Done! Saved to {cycle_dir}/ and {t_dir}/")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("6j Symbol Configuration Counter - NLO (s1-t1, s2-t3, s3-t2)")
    print("="*60)

    # Analyze 6J configuration
    by_cycles_6j, t_same_type_6j = count_all(FIXED_6J, "6J")
    print_results("6J", by_cycles_6j, t_same_type_6j)

    # Analyze PILLOW configuration
    by_cycles_pillow, t_same_type_pillow = count_all(FIXED_PILLOW, "PILLOW")
    print_results("PILLOW", by_cycles_pillow, t_same_type_pillow)

    # Draw both
    draw_results("6J", by_cycles_6j, t_same_type_6j, "6j")
    draw_results("PILLOW", by_cycles_pillow, t_same_type_pillow, "pillow")
