from collections import defaultdict

# Define ports
S = [("s", i) for i in (1, 2, 3)]
T = [("t", i) for i in (1, 2, 3)]
A = [("a", i) for i in (1, 2, 3)]
B = [("b", i) for i in (1, 2, 3)]
C = [("c", i) for i in (1, 2, 3)]

RHS_PORTS = T + A + B + C

# Fixed LHS edges
FIXED = [
    (("s", 1), ("a", 1)),
    (("s", 2), ("b", 2)),
    (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)),
    (("b", 3), ("c", 2)),
    (("c", 1), ("a", 3)),
]


def build_adjacency(edges):
    """Build adjacency list (ignores parallel edges)."""
    adj = defaultdict(set)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        adj[u].add(v)
        adj[v].add(u)
    return adj


def check_port_degrees(edges):
    """Check S/T have degree 1, A/B/C have degree 2 (counting parallel edges)."""
    adj = defaultdict(list)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        adj[u].append(v)
        adj[v].append(u)

    for port in S + T:
        if len(adj[port]) != 1:
            return False
    for port in A + B + C:
        if len(adj[port]) != 2:
            return False
    return True


def find_cycle(edges):
    """Find any cycle, including 2-cycles from parallel edges."""
    # Check for parallel edges first
    edge_counts = defaultdict(int)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        edge_counts[frozenset([u, v])] += 1

    for key, count in edge_counts.items():
        if count >= 2:
            u, v = tuple(key)
            return [(u, v), (v, u)]

    # Find regular cycles using DFS
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
    """Remove all cycles. Returns (remaining_edges, num_cycles_removed)."""
    E = [frozenset(e) if not isinstance(e, frozenset) else e for e in edges]
    num_cycles = 0

    while True:
        cyc = find_cycle(E)
        if not cyc:
            break
        E = [e for e in E if frozenset(e) not in {frozenset(c) for c in cyc}]
        num_cycles += 1

    return E, num_cycles


def check_three_paths(edges):
    """Check for 3 disjoint paths s_i → t_i."""
    adj = build_adjacency(edges)
    visited = set()

    def bfs(start):
        q, comp = [start], {start}
        visited.add(start)
        while q:
            x = q.pop(0)
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    comp.add(y)
                    q.append(y)
        return comp

    comps = [bfs(sp) for sp in S if sp not in visited]
    if len(comps) != 3:
        return False

    for comp in comps:
        s_list = [p for p in comp if p in S]
        t_list = [p for p in comp if p in T]
        if len(s_list) != 1 or len(t_list) != 1 or s_list[0][1] != t_list[0][1]:
            return False
        degs = [len([n for n in adj[v] if n in comp]) for v in comp]
        if not all(d in (1, 2) for d in degs) or sum(1 for d in degs if d == 1) != 2:
            return False
    return True


def generate_all_rhs_matchings():
    """Generate all perfect matchings on RHS_PORTS (no T-T edges)."""
    all_matchings = []

    def rec(remaining, cur):
        if len(cur) == 6:
            if not remaining:
                all_matchings.append(list(cur))
            return
        if len(remaining) < 2:
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            second = remaining[i]
            if first in T and second in T:
                continue
            rec([p for j, p in enumerate(remaining) if j not in (0, i)], cur + [(first, second)])

    rec(RHS_PORTS, [])
    return all_matchings


def count_valid_configurations():
    """Count all valid configurations."""
    print("Generating all RHS matchings...")
    all_rhs = generate_all_rhs_matchings()
    print(f"Total RHS matchings: {len(all_rhs)}")

    cycle_stats = defaultdict(list)

    for idx, rhs in enumerate(all_rhs):
        if (idx + 1) % 1000 == 0:
            print(f"  Checked {idx + 1}/{len(all_rhs)}...")

        combined = FIXED + rhs

        if not check_port_degrees(combined):
            continue

        after, num_cycles = remove_all_cycles(combined)

        if check_three_paths(after):
            cycle_stats[num_cycles].append((rhs, combined, after))

    return cycle_stats


if __name__ == "__main__":
    print("="*60)
    print("6j Symbol Configuration Counter")
    print("="*60)

    cycle_stats = count_valid_configurations()

    total = sum(len(v) for v in cycle_stats.values())

    print()
    print("="*60)
    print(f"TOTAL VALID CONFIGURATIONS: {total}")
    print("="*60)

    print("\nCycle removal statistics:")
    for num in sorted(cycle_stats.keys()):
        print(f"  {num} cycle(s) removed: {len(cycle_stats[num])} configurations")

    # Show examples
    print("\nExamples (prioritized by cycle count):")
    count = 0
    for num in sorted(cycle_stats.keys(), reverse=True):
        for rhs, combined, final in cycle_stats[num][:3]:
            count += 1
            print(f"\nExample {count} ({num} cycle(s) removed):")
            print(f"  RHS edges:")
            for e in rhs:
                print(f"    {e}")
            print(f"  Total edges: {len(combined)} → {len(final)} after removal")

            if count >= 5:
                break
        if count >= 5:
            break
