from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

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
            darts = sorted(darts, key=lambda d: d[2][1])
        elif v == 'T':
            darts = sorted(darts, key=lambda d: d[2][1])
        rotations[v] = [(nb, eid) for nb, eid, _ in darts]

    return [(uV, vV, eid) for uV, vV, eid, _, _ in edges_with_ids], rotations


def count_ribbon_faces_with_darts(merged_edges, rotations):
    """Count faces using dart permutations and return face cycles."""
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

    # φ = σ ∘ α
    phi = [sigma[alpha[i]] for i in range(D)]

    # Count φ-cycles and collect them
    visited, face_cycles = [False] * D, []
    for i in range(D):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(darts[j])
                j = phi[j]
            face_cycles.append(cycle)

    return len(face_cycles), face_cycles


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


def has_parallel_edges(edges):
    """Check if edges contain parallel edges."""
    edge_set = defaultdict(int)
    for e in edges:
        u, v = tuple(e) if isinstance(e, frozenset) else e
        key = frozenset([u, v])
        edge_set[key] += 1
    return any(count > 1 for count in edge_set.values())


def draw_faces_visualization(edges, title="Ribbon Graph with Faces", filename=None):
    """Draw ribbon graph with face boundaries highlighted."""
    merged_edges, rotations = merge_vertices_with_rotation(edges)
    num_faces, face_cycles = count_ribbon_faces_with_darts(merged_edges, rotations)

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

    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw vertices
    for v, (x, y) in positions.items():
        r = 0.3 if v in ['S', 'T'] else 0.2
        ax.add_patch(plt.Circle((x, y), r, color=colors[v], ec='black', lw=2, zorder=3))
        label = v if v in ['S', 'T'] else f"{v[0]}{v[1]}"
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

    # Count parallel edges for offset
    edge_counts = defaultdict(int)
    for u, v, eid in merged_edges:
        if u in positions and v in positions:
            key = tuple(sorted([u, v], key=str))
            edge_counts[key] += 1

    # Draw edges with offsets for parallel edges
    edge_drawn = defaultdict(int)
    for u, v, eid in merged_edges:
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            key = tuple(sorted([u, v], key=str))
            total_parallel = edge_counts[key]
            current_idx = edge_drawn[key]
            edge_drawn[key] += 1

            if total_parallel > 1:
                # Offset parallel edges
                offset = (current_idx - (total_parallel - 1) / 2) * 0.15
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                norm = np.sqrt(dx**2 + dy**2)
                perp_x, perp_y = -dy / norm, dx / norm

                ctrl_x = mid_x + perp_x * offset
                ctrl_y = mid_y + perp_y * offset

                # Draw curved edge
                t = np.linspace(0, 1, 50)
                bx = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
                by = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
                ax.plot(bx, by, 'k-', lw=2, alpha=0.5, zorder=1)

                # Label edge ID
                ax.text(ctrl_x, ctrl_y, f"e{eid}", fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       ha='center', va='center', zorder=2)
            else:
                ax.plot([x1, x2], [y1, y2], 'k-', lw=2, alpha=0.5, zorder=1)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, f"e{eid}", fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       ha='center', va='center', zorder=2)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add face information
    face_colors = plt.cm.tab10(np.linspace(0, 1, num_faces))
    face_text = f"Faces = {num_faces}\n\n"
    for i, cycle in enumerate(face_cycles):
        face_text += f"Face {i+1}: "
        dart_str = " → ".join([f"{u}→{v}(e{eid})" for u, v, eid in cycle[:min(3, len(cycle))]])
        if len(cycle) > 3:
            dart_str += "..."
        face_text += f"{dart_str}\n"

    ax.text(0.02, 0.98, face_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.close()

    return num_faces, face_cycles


# Find example with parallel edges
print("Finding example with parallel edges...")
all_rhs = generate_all_rhs_matchings()

for rhs in all_rhs:
    combined = FIXED + rhs

    if not has_parallel_edges(combined):
        continue

    if not check_port_degrees(combined):
        continue

    after = remove_all_cycles(combined)
    if not check_three_paths(after):
        continue

    # Found one!
    print("\n" + "="*60)
    print("Example with PARALLEL EDGES:")
    print("="*60)

    print("\nFIXED edges:")
    for e in FIXED:
        print(f"  {e}")

    print("\nRHS edges:")
    for e in rhs:
        print(f"  {e}")

    # Check for overlaps
    fixed_set = [frozenset(e) for e in FIXED]
    rhs_set = [frozenset(e) for e in rhs]
    overlaps = [e for e in rhs_set if e in fixed_set]

    print(f"\nParallel edges (FIXED ∩ RHS): {len(overlaps)}")
    for e in overlaps:
        print(f"  {tuple(e)}")

    print(f"\nTotal edges: {len(combined)}")
    print(f"Unique edges: {len(set(frozenset(e) for e in combined))}")

    merged_edges, rotations = merge_vertices_with_rotation(combined)
    V = len({v for u, v, _ in merged_edges for v in (u, v)})
    E = len(merged_edges)
    num_faces, face_cycles = count_ribbon_faces_with_darts(merged_edges, rotations)
    chi = V - E + num_faces

    print(f"\nTopology: V={V}, E={E}, F={num_faces}, χ={chi}")

    print(f"\nFace boundaries (dart cycles):")
    for i, cycle in enumerate(face_cycles):
        print(f"\nFace {i+1} (length {len(cycle)}):")
        for u, v, eid in cycle:
            print(f"  {u} → {v} (edge {eid})")

    # Draw visualization
    draw_faces_visualization(combined,
                            f"Parallel Edges Example (χ={chi}, F={num_faces})",
                            "parallel_edges_faces.png")

    break
