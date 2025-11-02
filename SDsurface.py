from collections import defaultdict
import matplotlib.pyplot as plt

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
    """Build adjacency list (ignores multiplicity)."""
    adj = defaultdict(set)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        adj[u].add(v)
        adj[v].add(u)
    return adj


def check_port_degrees(edges):
    """Check S/T have degree 1, A/B/C have degree 2."""
    adj = build_adjacency(edges)
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
    """Repeatedly remove cycles."""
    E = [frozenset(e) if not isinstance(e, frozenset) else e for e in edges]
    while True:
        cyc = find_cycle(E)
        if not cyc:
            break
        E = [e for e in E if frozenset(e) not in {frozenset(c) for c in cyc}]
    return E


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


def merge_vertices_with_rotation(edges):
    """
    Merge s1,s2,s3 → S and t1,t2,t3 → T with rotation system.
    S/T rotation orders by port index (1,2,3).
    Returns: (merged_edges, rotations)
    """
    edges_with_ids, v_darts = [], defaultdict(list)

    for eid, e in enumerate(edges):
        u, v = tuple(e) if isinstance(e, frozenset) else e
        uV = 'S' if u in S else ('T' if u in T else u)
        vV = 'S' if v in S else ('T' if v in T else v)
        edges_with_ids.append((uV, vV, eid, u, v))
        v_darts[uV].append((vV, eid, u))
        v_darts[vV].append((uV, eid, v))

    rotations = {}
    for v in ['S', 'T'] + A + B + C:
        darts = v_darts.get(v, [])
        if v == 'S':
            for (_, _, orig) in darts:
                assert orig in S
            darts = sorted(darts, key=lambda d: d[2][1])
        elif v == 'T':
            for (_, _, orig) in darts:
                assert orig in T
            darts = sorted(darts, key=lambda d: d[2][1])
        else:
            if darts:
                assert len(darts) == 2
        rotations[v] = [(nb, eid) for nb, eid, _ in darts]

    return [(uV, vV, eid) for uV, vV, eid, _, _ in edges_with_ids], rotations


def compute_euler_characteristic(edges):
    """Compute χ = V - E + F for ribbon graph."""
    if not edges:
        return 0, 0, 0, 0

    merged_edges, rotations = merge_vertices_with_rotation(edges)
    V = len({v for u, v, _ in merged_edges for v in (u, v)})
    E = len(merged_edges)
    F = count_ribbon_faces_with_darts(merged_edges, rotations)
    return V - E + F, V, E, F


def count_ribbon_faces_with_darts(merged_edges, rotations):
    """Count faces using dart permutations: φ = σ ∘ α."""
    # Build darts
    darts, dart_map = [], {}
    for u, v, eid in merged_edges:
        idx = len(darts)
        darts.extend([(u, v, eid), (v, u, eid)])
        dart_map[(u, v, eid)] = idx
        dart_map[(v, u, eid)] = idx + 1

    D = len(darts)

    # α: edge flip
    alpha = [dart_map[(v, u, eid)] for u, v, eid in darts]
    assert all(alpha[alpha[i]] == i for i in range(D))

    # σ: vertex rotation (cycles outgoing darts)
    sigma = [None] * D
    for u, cyc in rotations.items():
        if not cyc:
            continue
        n = len(cyc)
        for k in range(n):
            nb_k, eid_k = cyc[k]
            nb_next, eid_next = cyc[(k + 1) % n]
            out_dart = (u, nb_k, eid_k)
            out_next = (u, nb_next, eid_next)
            if out_dart in dart_map and out_next in dart_map:
                sigma[dart_map[out_dart]] = dart_map[out_next]

    # Verify σ
    for i, (u, v, eid) in enumerate(darts):
        if u in rotations and rotations[u]:
            assert sigma[i] is not None

    # φ = σ ∘ α
    phi = [sigma[alpha[i]] for i in range(D)]
    assert all(p is not None for p in phi)

    # Count φ-cycles
    visited, faces = [False] * D, 0
    for i in range(D):
        if not visited[i]:
            j = i
            while not visited[j]:
                visited[j] = True
                j = phi[j]
            faces += 1
    return faces


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


def draw_ribbon_graph(edges, title="Ribbon Graph", filename=None):
    """Draw ribbon graph with merged S/T vertices."""
    merged_edges, _ = merge_vertices_with_rotation(edges)

    positions = {
        'S': (0, 5), 'T': (10, 5),
        ('a', 1): (4, 8), ('a', 2): (3.5, 7), ('a', 3): (4.5, 7),
        ('b', 1): (5, 5), ('b', 2): (4.5, 4), ('b', 3): (5.5, 4),
        ('c', 1): (4, 2), ('c', 2): (3.5, 1), ('c', 3): (4.5, 1),
    }

    colors = {
        'S': 'lightblue', 'T': 'plum',
        **{p: 'lightgreen' for p in A},
        **{p: 'lightyellow' for p in B},
        **{p: 'lightcoral' for p in C},
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    for v, (x, y) in positions.items():
        r = 0.3 if v in ['S', 'T'] else 0.2
        ax.add_patch(plt.Circle((x, y), r, color=colors[v], ec='black', lw=2, zorder=3))
        label = v if v in ['S', 'T'] else f"{v[0]}{v[1]}"
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

    for u, v, _ in merged_edges:
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], 'k-', lw=3, alpha=0.6, zorder=1)

    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.close()


def count_with_euler_characteristic():
    """Count valid configurations and compute Euler characteristics."""
    all_rhs = generate_all_rhs_matchings()
    print(f"Checking {len(all_rhs)} RHS matchings...")

    euler_chars = defaultdict(list)

    for idx, rhs in enumerate(all_rhs):
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(all_rhs)}...")

        combined = FIXED + rhs
        if not check_port_degrees(combined):
            continue

        after = remove_all_cycles(combined)
        if not check_three_paths(after):
            continue

        chi, V, E, F = compute_euler_characteristic(combined)
        euler_chars[chi].append({
            'rhs': rhs, 'combined': combined, 'final': after,
            'V': V, 'E': E, 'F': F, 'chi': chi
        })

    return sum(len(v) for v in euler_chars.values()), euler_chars


if __name__ == "__main__":
    print("="*60)
    print("6j Symbol Surface Analysis")
    print("="*60)

    count, euler_chars = count_with_euler_characteristic()

    print(f"\nTOTAL VALID CONFIGURATIONS: {count}")
    print("="*60)
    print("\nEuler Characteristic Distribution:")
    for chi in sorted(euler_chars.keys()):
        ex = euler_chars[chi][0]
        print(f"  χ = {chi:2d}: {len(euler_chars[chi]):4d} configs  (V={ex['V']}, E={ex['E']}, F={ex['F']})")

    print("\n" + "="*60)
    for chi in sorted(euler_chars.keys()):
        ex = euler_chars[chi][0]
        print(f"\nχ={chi} example (V={ex['V']}, E={ex['E']}, F={ex['F']}):")
        for e in ex['rhs']:
            print(f"  {e}")
        draw_ribbon_graph(ex['combined'], f"χ={chi}", f"ribbon_chi_{chi}.png")

    print("\nDone!")
