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

# Mode edges: three cuffs (A, B, C triangles) + one cross-group edge
MODE_EDGES = [
    (("a", 1), ("a", 2)), (("a", 2), ("a", 3)), (("a", 3), ("a", 1)),
    (("b", 1), ("b", 2)), (("b", 2), ("b", 3)), (("b", 3), ("b", 1)),
    (("c", 1), ("c", 2)), (("c", 2), ("c", 3)), (("c", 3), ("c", 1)),
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
    adj = defaultdict(list)  # Use list to count multiplicities
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


def merge_ST(p):
    """Merge s1,s2,s3 → 'S' and t1,t2,t3 → 'T'."""
    return 'S' if p in S else ('T' if p in T else p)


def key_label(p):
    """Sorting key for port labels."""
    if isinstance(p, str):
        return (0, p, 0)
    name, idx = p
    return (1, name, idx)


def next_prev_in_group(port):
    """Get next/prev port in triangle (a1→a2→a3→a1)."""
    if isinstance(port, tuple) and port[0] in ('a', 'b', 'c'):
        name, i = port
        nxt = (name, 1 if i == 3 else i + 1)
        prv = (name, 3 if i == 1 else i - 1)
        return nxt, prv
    return None, None


def merge_vertices_with_rotation(edges):
    """
    Build rotation system with PHYSICAL edges + MODE edges (cuffs).
    Returns: (merged_edges, rotations)
    """
    # Physical edges with merged endpoints
    phys_m = [(merge_ST(u), merge_ST(v), u, v) for (u, v) in edges]
    mode_m = [(merge_ST(u), merge_ST(v)) for (u, v) in MODE_EDGES]

    # Assign unique edge IDs
    merged_edges = []
    for eid, (U, V, u_orig, v_orig) in enumerate(phys_m):
        merged_edges.append((U, V, eid, 'phys', u_orig, v_orig))
    base = len(phys_m)
    for j, (U, V) in enumerate(mode_m):
        merged_edges.append((U, V, base + j, 'mode', None, None))

    # Build incident darts per vertex
    incident = defaultdict(list)
    vertices = set(['S', 'T'])
    for (U, V, eid, tag, u_orig, v_orig) in merged_edges:
        vertices.add(U)
        vertices.add(V)
        incident[U].append((V, eid, tag, u_orig if tag == 'phys' else None))
        incident[V].append((U, eid, tag, v_orig if tag == 'phys' else None))

    rotations = {}

    for v in vertices:
        darts = incident.get(v, [])

        if v in ('S', 'T'):
            # S/T: physical darts by port index, mode darts by label
            phys = [(nb, eid, tag, orig) for (nb, eid, tag, orig) in darts if tag == 'phys']
            mode = [(nb, eid, tag, orig) for (nb, eid, tag, orig) in darts if tag == 'mode']

            phys_sorted = sorted(phys, key=lambda d: d[3][1] if d[3] in S or d[3] in T else 0)
            mode_sorted = sorted(mode, key=lambda d: key_label(d[0]))

            # Interleave for stability
            cyc = []
            for i in range(max(len(mode_sorted), len(phys_sorted))):
                if i < len(mode_sorted):
                    cyc.append((mode_sorted[i][0], mode_sorted[i][1]))
                if i < len(phys_sorted):
                    cyc.append((phys_sorted[i][0], phys_sorted[i][1]))

            rotations[v] = cyc
            continue

        # Intermediate ports: interleave [tri_next, phys_1, tri_prev, phys_2]
        phys = [(nb, eid) for (nb, eid, tag, _) in darts if tag == 'phys']
        mode = [(nb, eid) for (nb, eid, tag, _) in darts if tag == 'mode']

        nxt, prv = next_prev_in_group(v)
        d_nxt = next(((nb, eid) for (nb, eid) in mode if nb == nxt), None)
        d_prv = next(((nb, eid) for (nb, eid) in mode if nb == prv), None)

        phys_sorted = sorted(phys, key=lambda x: key_label(x[0]))

        if d_nxt and d_prv and len(phys_sorted) >= 2:
            cyc = [d_nxt, phys_sorted[0], d_prv, phys_sorted[1]]
        else:
            # Fallback: stable mix
            mode_sorted = sorted(mode, key=lambda x: key_label(x[0]))
            cyc = []
            for i in range(max(len(mode_sorted), len(phys_sorted))):
                if i < len(mode_sorted):
                    cyc.append(mode_sorted[i])
                if i < len(phys_sorted):
                    cyc.append(phys_sorted[i])

        rotations[v] = cyc

    # Return simplified edge list
    edges_all = [(U, V, eid) for (U, V, eid, _, _, _) in merged_edges]
    return edges_all, rotations


def count_ribbon_faces_with_darts(merged_edges, rotations):
    """Count faces using dart permutations: φ = σ ∘ α."""
    darts, dart_map = [], {}
    for u, v, eid in merged_edges:
        idx = len(darts)
        darts.extend([(u, v, eid), (v, u, eid)])
        dart_map[(u, v, eid)] = idx
        dart_map[(v, u, eid)] = idx + 1

    D = len(darts)

    # α: edge flip
    alpha = [dart_map[(v, u, eid)] for u, v, eid in darts]

    # σ: vertex rotation
    sigma = [None] * D
    for u, cyc in rotations.items():
        n = len(cyc)
        if n == 0:
            continue
        for k in range(n):
            nb_k, eid_k = cyc[k]
            nb_n, eid_n = cyc[(k + 1) % n]
            out_dart = (u, nb_k, eid_k)
            out_next = (u, nb_n, eid_n)
            if out_dart in dart_map and out_next in dart_map:
                sigma[dart_map[out_dart]] = dart_map[out_next]

    # φ = σ ∘ α
    phi = [sigma[alpha[i]] if alpha[i] is not None else None for i in range(D)]

    # Count φ-cycles
    visited, faces = [False] * D, 0
    for i in range(D):
        if visited[i] or phi[i] is None:
            continue
        j = i
        while not visited[j] and phi[j] is not None:
            visited[j] = True
            j = phi[j]
        faces += 1
    return faces


def compute_euler_characteristic(edges):
    """Compute χ = V - E + F including mode edges (cuffs)."""
    if not edges:
        return 0, 0, 0, 0

    merged_edges, rotations = merge_vertices_with_rotation(edges)

    # V includes S, T and all vertices touched by edges
    Vset = set(['S', 'T'])
    for u, v, _ in merged_edges:
        Vset.add(u)
        Vset.add(v)
    V = len(Vset)

    # E counts ALL ribbons (physical + mode)
    E = len(merged_edges)

    # F counts boundary faces
    F = count_ribbon_faces_with_darts(merged_edges, rotations)

    return V - E + F, V, E, F


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
    print("6j Symbol Surface Analysis (with Mode Edges)")
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
        draw_ribbon_graph(ex['combined'], f"χ={chi} (with cuffs)", f"ribbon2_chi_{chi}.png")

    print("\nDone!")
