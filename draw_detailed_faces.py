from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use the same example
S = [("s", i) for i in (1, 2, 3)]
T = [("t", i) for i in (1, 2, 3)]
A = [("a", i) for i in (1, 2, 3)]
B = [("b", i) for i in (1, 2, 3)]
C = [("c", i) for i in (1, 2, 3)]

FIXED = [
    (("s", 1), ("a", 1)),
    (("s", 2), ("b", 2)),
    (("s", 3), ("c", 3)),
    (("a", 2), ("b", 1)),
    (("b", 3), ("c", 2)),
    (("c", 1), ("a", 3)),
]

RHS = [
    (("t", 1), ("a", 1)),
    (("t", 2), ("a", 2)),
    (("t", 3), ("a", 3)),
    (("b", 1), ("b", 2)),
    (("b", 3), ("c", 2)),  # PARALLEL with FIXED!
    (("c", 1), ("c", 3)),
]


def merge_vertices_with_rotation(edges):
    """Merge s1,s2,s3 → S and t1,t2,t3 → T with rotation system."""
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


combined = FIXED + RHS
merged_edges, rotations = merge_vertices_with_rotation(combined)
num_faces, face_cycles = count_ribbon_faces_with_darts(merged_edges, rotations)

positions = {
    'S': (0, 5), 'T': (10, 5),
    ('a', 1): (4, 8), ('a', 2): (3.5, 7), ('a', 3): (4.5, 7),
    ('b', 1): (5, 5), ('b', 2): (4.5, 4), ('b', 3): (5.5, 4),
    ('c', 1): (4, 2), ('c', 2): (3.5, 1), ('c', 3): (4.5, 1),
}

fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()

# Face colors
face_colors_list = ['lightcoral', 'lightgreen', 'lightblue']

# Plot 0: Original graph with all edges
ax = axes[0]
vertex_colors = {
    'S': 'lightblue', 'T': 'plum',
    **{p: 'lightgreen' for p in A},
    **{p: 'lightyellow' for p in B},
    **{p: 'lightcoral' for p in C},
}

for v, (x, y) in positions.items():
    r = 0.3 if v in ['S', 'T'] else 0.2
    ax.add_patch(plt.Circle((x, y), r, color=vertex_colors[v], ec='black', lw=2, zorder=3))
    label = v if v in ['S', 'T'] else f"{v[0]}{v[1]}"
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

# Draw edges with offset for parallel edges
edge_counts = defaultdict(int)
for u, v, eid in merged_edges:
    if u in positions and v in positions:
        key = tuple(sorted([u, v], key=str))
        edge_counts[key] += 1

edge_drawn = defaultdict(int)
for u, v, eid in merged_edges:
    if u in positions and v in positions:
        x1, y1 = positions[u]
        x2, y2 = positions[v]

        key = tuple(sorted([u, v], key=str))
        total_parallel = edge_counts[key]
        current_idx = edge_drawn[key]
        edge_drawn[key] += 1

        # Check if this is the parallel edge
        is_parallel = (eid in [4, 10])  # edges 4 and 10 are the parallel ones
        color = 'red' if is_parallel else 'black'
        lw = 3 if is_parallel else 2

        if total_parallel > 1:
            offset = (current_idx - (total_parallel - 1) / 2) * 0.2
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            norm = np.sqrt(dx**2 + dy**2)
            perp_x, perp_y = -dy / norm, dx / norm

            ctrl_x = mid_x + perp_x * offset
            ctrl_y = mid_y + perp_y * offset

            t = np.linspace(0, 1, 50)
            bx = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
            by = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
            ax.plot(bx, by, color=color, lw=lw, alpha=0.7, zorder=1)

            ax.text(ctrl_x, ctrl_y, f"e{eid}", fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                   ha='center', va='center', zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=0.7, zorder=1)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, f"e{eid}", fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                   ha='center', va='center', zorder=2)

ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Graph with Parallel Edges (red = parallel)", fontsize=14, fontweight='bold')

# Plot 1-3: Individual faces
for face_idx in range(min(3, num_faces)):
    ax = axes[face_idx + 1]

    # Draw all vertices
    for v, (x, y) in positions.items():
        r = 0.3 if v in ['S', 'T'] else 0.2
        ax.add_patch(plt.Circle((x, y), r, color=vertex_colors[v], ec='black', lw=2, zorder=3))
        label = v if v in ['S', 'T'] else f"{v[0]}{v[1]}"
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

    # Draw all edges lightly
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
                offset = (current_idx - (total_parallel - 1) / 2) * 0.2
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                norm = np.sqrt(dx**2 + dy**2)
                perp_x, perp_y = -dy / norm, dx / norm

                ctrl_x = mid_x + perp_x * offset
                ctrl_y = mid_y + perp_y * offset

                t = np.linspace(0, 1, 50)
                bx = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
                by = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
                ax.plot(bx, by, 'gray', lw=1, alpha=0.3, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], 'gray', lw=1, alpha=0.3, zorder=1)

    # Highlight this face's boundary
    cycle = face_cycles[face_idx]
    face_color = face_colors_list[face_idx]

    for step, (u, v, eid) in enumerate(cycle):
        if u in positions and v in positions:
            x1, y1 = positions[u]
            x2, y2 = positions[v]

            # Check if parallel edge
            key = tuple(sorted([u, v], key=str))
            total_parallel = edge_counts[key]

            if total_parallel > 1:
                # Find which instance this is
                instances = [(uu, vv, ee) for (uu, vv, ee) in merged_edges
                            if tuple(sorted([uu, vv], key=str)) == key]
                current_idx = next(i for i, (uu, vv, ee) in enumerate(instances) if ee == eid)

                offset = (current_idx - (total_parallel - 1) / 2) * 0.2
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                norm = np.sqrt(dx**2 + dy**2)
                perp_x, perp_y = -dy / norm, dx / norm

                ctrl_x = mid_x + perp_x * offset
                ctrl_y = mid_y + perp_y * offset

                t = np.linspace(0, 1, 50)
                bx = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
                by = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
                ax.plot(bx, by, color=face_color, lw=4, alpha=0.8, zorder=2)

                # Arrow
                arrow_t = 0.5
                arrow_x = (1-arrow_t)**2 * x1 + 2*(1-arrow_t)*arrow_t * ctrl_x + arrow_t**2 * x2
                arrow_y = (1-arrow_t)**2 * y1 + 2*(1-arrow_t)*arrow_t * ctrl_y + arrow_t**2 * y2
                dx_t = 2*(1-arrow_t)*(ctrl_x - x1) + 2*arrow_t*(x2 - ctrl_x)
                dy_t = 2*(1-arrow_t)*(ctrl_y - y1) + 2*arrow_t*(y2 - ctrl_y)
                ax.arrow(arrow_x, arrow_y, dx_t*0.1, dy_t*0.1, head_width=0.15, head_length=0.1,
                        fc=face_color, ec=face_color, zorder=2)
            else:
                ax.plot([x1, x2], [y1, y2], color=face_color, lw=4, alpha=0.8, zorder=2)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                ax.arrow(mid_x - dx*0.05, mid_y - dy*0.05, dx*0.1, dy*0.1,
                        head_width=0.15, head_length=0.1, fc=face_color, ec=face_color, zorder=2)

    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Face {face_idx+1} (boundary length {len(cycle)})", fontsize=14, fontweight='bold')

plt.suptitle(f"Ribbon Graph Faces: V=11, E=12, F={num_faces}, χ={11-12+num_faces}",
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("detailed_faces_parallel.png", dpi=150, bbox_inches='tight')
print("Saved: detailed_faces_parallel.png")
plt.close()

# Print summary
print("\n" + "="*60)
print("PARALLEL EDGES EXAMPLE - FACE ANALYSIS")
print("="*60)
print(f"\nParallel edge: (b3, c2) appears twice (edge 4 and edge 10)")
print(f"\nTopology: V=11, E=12, F={num_faces}, χ={11-12+num_faces}")
print(f"\nFace structure:")
for i, cycle in enumerate(face_cycles):
    print(f"\nFace {i+1}: length {len(cycle)}")
    if len(cycle) <= 10:
        for u, v, eid in cycle:
            print(f"  {u} → {v} (e{eid})")
    else:
        for u, v, eid in cycle[:5]:
            print(f"  {u} → {v} (e{eid})")
        print(f"  ... ({len(cycle)-5} more darts)")
