from collections import defaultdict

# Define ports
S = [("s", i) for i in (1, 2, 3)]
T = [("t", i) for i in (1, 2, 3)]
A = [("a", i) for i in (1, 2, 3)]
B = [("b", i) for i in (1, 2, 3)]
C = [("c", i) for i in (1, 2, 3)]

RHS_PORTS = T + A + B + C

FIXED = [
    (("s", 1), ("a", 1)),
    (("s", 2), ("b", 2)),
    (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)),
    (("b", 3), ("c", 2)),
    (("c", 1), ("a", 3)),
]


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


def build_adjacency(edges):
    """Build adjacency list (ignores multiplicity for cycle/path checking)."""
    adj = defaultdict(set)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_cycle(edges):
    """Find any cycle using DFS."""
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
    """Repeatedly remove cycles and track them."""
    E = [frozenset(e) if not isinstance(e, frozenset) else e for e in edges]
    removed_cycles = []

    while True:
        cyc = find_cycle(E)
        if not cyc:
            break
        removed_cycles.append(cyc)
        E = [e for e in E if frozenset(e) not in {frozenset(c) for c in cyc}]

    return E, removed_cycles


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


def has_parallel_edges(edges):
    """Check if edges contain parallel edges."""
    edge_counts = defaultdict(int)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        key = frozenset([u, v])
        edge_counts[key] += 1
    return any(count > 1 for count in edge_counts.values()), edge_counts


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


print("Searching for configurations with parallel edges and multiple cycles...")
all_rhs = generate_all_rhs_matchings()

stats = {
    'parallel_no_cycles': [],
    'parallel_1_cycle': [],
    'parallel_multiple_cycles': [],
    'no_parallel_1_cycle': [],
    'no_parallel_multiple_cycles': [],
}

for rhs in all_rhs:
    combined = FIXED + rhs

    if not check_port_degrees(combined):
        continue

    has_parallel, edge_counts = has_parallel_edges(combined)
    after, removed = remove_all_cycles(combined)

    if not check_three_paths(after):
        continue

    num_cycles = len(removed)

    # Categorize
    if has_parallel and num_cycles == 0:
        stats['parallel_no_cycles'].append((rhs, combined, removed))
    elif has_parallel and num_cycles == 1:
        stats['parallel_1_cycle'].append((rhs, combined, removed))
    elif has_parallel and num_cycles >= 2:
        stats['parallel_multiple_cycles'].append((rhs, combined, removed))
    elif not has_parallel and num_cycles == 1:
        stats['no_parallel_1_cycle'].append((rhs, combined, removed))
    elif not has_parallel and num_cycles >= 2:
        stats['no_parallel_multiple_cycles'].append((rhs, combined, removed))

print("\n" + "="*60)
print("STATISTICS")
print("="*60)
print(f"Parallel edges, 0 cycles: {len(stats['parallel_no_cycles'])}")
print(f"Parallel edges, 1 cycle: {len(stats['parallel_1_cycle'])}")
print(f"Parallel edges, 2+ cycles: {len(stats['parallel_multiple_cycles'])}")
print(f"No parallel, 1 cycle: {len(stats['no_parallel_1_cycle'])}")
print(f"No parallel, 2+ cycles: {len(stats['no_parallel_multiple_cycles'])}")

if stats['parallel_multiple_cycles']:
    print("\n" + "="*60)
    print("EXAMPLES WITH PARALLEL EDGES AND MULTIPLE CYCLES")
    print("="*60)
    for i, (rhs, combined, removed) in enumerate(stats['parallel_multiple_cycles'][:3], 1):
        print(f"\nExample {i}:")
        print(f"  Cycles removed: {len(removed)}")

        has_parallel, edge_counts = has_parallel_edges(combined)
        print(f"\n  Parallel edges:")
        for key, count in edge_counts.items():
            if count > 1:
                u, v = tuple(key)
                print(f"    {u} - {v}: appears {count} times")

        print(f"\n  Cycles:")
        for j, cyc in enumerate(removed, 1):
            print(f"    Cycle {j}: {' → '.join([f'{u[0]}{u[1]}' for u, v in cyc] + [f'{cyc[0][0][0]}{cyc[0][0][1]}'])}")

elif stats['no_parallel_multiple_cycles']:
    print("\n" + "="*60)
    print("EXAMPLES WITH MULTIPLE CYCLES (NO PARALLEL EDGES)")
    print("="*60)
    for i, (rhs, combined, removed) in enumerate(stats['no_parallel_multiple_cycles'][:3], 1):
        print(f"\nExample {i}:")
        print(f"  Cycles removed: {len(removed)}")
        print(f"\n  Cycles:")
        for j, cyc in enumerate(removed, 1):
            print(f"    Cycle {j}: {' → '.join([f'{u[0]}{u[1]}' for u, v in cyc] + [f'{cyc[0][0][0]}{cyc[0][0][1]}'])}")
else:
    print("\nNo configurations found with 2 or more cycles!")
    print("\nThis is because parallel edges form a 2-cycle only if detected properly,")
    print("but the current cycle detection using DFS skips back-to-parent edges,")
    print("so it doesn't detect parallel edges as cycles!")
