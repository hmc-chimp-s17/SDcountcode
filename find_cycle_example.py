from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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


def build_adjacency(edges):
    """Build adjacency list (ignores multiplicity for cycle/path checking)."""
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
    edge_set = defaultdict(int)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        key = frozenset([u, v])
        edge_set[key] += 1
    return any(count > 1 for count in edge_set.values())


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


# Find examples with cycles AND parallel edges
print("Searching for examples...")
all_rhs = generate_all_rhs_matchings()

examples_with_cycles = []
examples_with_parallel = []

for rhs in all_rhs:
    combined = FIXED + rhs

    if not check_port_degrees(combined):
        continue

    after, removed = remove_all_cycles(combined)
    if not check_three_paths(after):
        continue

    has_parallel = has_parallel_edges(combined)
    has_cycles = len(removed) > 0

    if has_cycles:
        examples_with_cycles.append((rhs, combined, after, removed, has_parallel))

    if has_parallel:
        examples_with_parallel.append((rhs, combined, after, removed, has_cycles))

print(f"\nTotal valid configurations: {len(examples_with_cycles) + len([x for x in examples_with_parallel if x not in examples_with_cycles])}")
print(f"Configurations with cycles: {len(examples_with_cycles)}")
print(f"Configurations with parallel edges: {len(examples_with_parallel)}")
print(f"Configurations with BOTH cycles AND parallel edges: {len([x for x in examples_with_cycles if x[4]])}")

print("\n" + "="*60)
print("EXAMPLE WITH CYCLES (no parallel edges):")
print("="*60)

if examples_with_cycles:
    # Find one without parallel edges
    ex = next((x for x in examples_with_cycles if not x[4]), examples_with_cycles[0])
    rhs, combined, after, removed, has_parallel = ex

    print(f"\nHas parallel edges: {has_parallel}")
    print(f"Cycles removed: {len(removed)}")

    print("\nFIXED edges:")
    for e in FIXED:
        print(f"  {e}")

    print("\nRHS edges:")
    for e in rhs:
        print(f"  {e}")

    print(f"\nBefore cycle removal: {len(combined)} edges")
    print(f"After cycle removal: {len(after)} edges")

    print("\nCycles found:")
    for i, cyc in enumerate(removed):
        print(f"\nCycle {i+1}:")
        for u, v in cyc:
            print(f"  {u} - {v}")

print("\n" + "="*60)
print("EXAMPLE WITH PARALLEL EDGES (no cycles):")
print("="*60)

if examples_with_parallel:
    # Find one without cycles
    ex = next((x for x in examples_with_parallel if not x[4]), examples_with_parallel[0])
    rhs, combined, after, removed, has_cycles = ex

    print(f"\nHas cycles: {has_cycles}")
    print(f"Cycles removed: {len(removed)}")

    print("\nFIXED edges:")
    for e in FIXED:
        print(f"  {e}")

    print("\nRHS edges:")
    for e in rhs:
        print(f"  {e}")

    # Check for parallel edges
    edge_counts = defaultdict(int)
    for e in combined:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        key = frozenset([u, v])
        edge_counts[key] += 1

    print("\nParallel edges:")
    for key, count in edge_counts.items():
        if count > 1:
            u, v = tuple(key)
            print(f"  {u} - {v}: appears {count} times")

print("\n" + "="*60)
if any(x[4] for x in examples_with_cycles):
    print("EXAMPLE WITH BOTH CYCLES AND PARALLEL EDGES:")
    print("="*60)

    ex = next(x for x in examples_with_cycles if x[4])
    rhs, combined, after, removed, has_parallel = ex

    print(f"\nCycles removed: {len(removed)}")

    print("\nFIXED edges:")
    for e in FIXED:
        print(f"  {e}")

    print("\nRHS edges:")
    for e in rhs:
        print(f"  {e}")

    print(f"\nBefore cycle removal: {len(combined)} edges")
    print(f"After cycle removal: {len(after)} edges")

    print("\nCycles found:")
    for i, cyc in enumerate(removed):
        print(f"\nCycle {i+1}:")
        for u, v in cyc:
            print(f"  {u} - {v}")

    # Check for parallel edges
    edge_counts = defaultdict(int)
    for e in combined:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        key = frozenset([u, v])
        edge_counts[key] += 1

    print("\nParallel edges:")
    for key, count in edge_counts.items():
        if count > 1:
            u, v = tuple(key)
            print(f"  {u} - {v}: appears {count} times")
