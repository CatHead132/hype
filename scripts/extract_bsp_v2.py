import struct
import numpy as np
from srctools.bsp import BSP, SurfFlags
from srctools.math import Vec

BSP_PATH = "/home/cathead/Downloads/GarrysMod/garrysmod/maps/gm_construct.bsp"
OUT_NPZ = "/tmp/claude-1000/-home-cathead/e8d60c32-98bf-415b-b7be-871772b20419/scratchpad/gm_construct_mesh_v2.npz"

bsp = BSP(BSP_PATH)

# ---- parse DISPINFO + DISP_VERTS lumps manually ----
from srctools.bsp import BSP_LUMPS
raw_dispinfo = bsp.get_lump(BSP_LUMPS.DISPINFO)
raw_dispverts = bsp.get_lump(BSP_LUMPS.DISP_VERTS)

DISPINFO_FMT = '<3f4ifiH2x2i128x'
DISPINFO_SIZE = struct.calcsize(DISPINFO_FMT)
assert DISPINFO_SIZE == 176, DISPINFO_SIZE
n_dispinfo = len(raw_dispinfo) // DISPINFO_SIZE
dispinfos = []
for i in range(n_dispinfo):
    (sx, sy, sz, dispvertstart, disptristart, power, mintess, smoothangle,
     contents, mapface, lmalphastart, lmsamplestart) = struct.unpack_from(DISPINFO_FMT, raw_dispinfo, i * DISPINFO_SIZE)
    dispinfos.append(dict(start=Vec(sx, sy, sz), dispvertstart=dispvertstart, power=power, mapface=mapface))

DISPVERT_FMT = '<3f2f'
DISPVERT_SIZE = struct.calcsize(DISPVERT_FMT)
assert DISPVERT_SIZE == 20
n_dispverts = len(raw_dispverts) // DISPVERT_SIZE
dispverts = []
for i in range(n_dispverts):
    vx, vy, vz, dist, alpha = struct.unpack_from(DISPVERT_FMT, raw_dispverts, i * DISPVERT_SIZE)
    dispverts.append((Vec(vx, vy, vz), dist))

print(f"parsed {n_dispinfo} dispinfos, {n_dispverts} dispverts")

# map orig_face index -> its texinfo (needed for displacement UV projection) and dispinfo
face_list = list(bsp.faces)
mapface_to_dispinfo = {di['mapface']: di for di in dispinfos}

verts = []   # (x,y,z) source units
uvs = []     # (u,v)
tris = []    # (i0,i1,i2)
tri_mat = [] # material name per triangle
tri_src = [] # human-readable source label per triangle, e.g. "flat_face_1523" / "disp_42"

skipped_nodraw = 0
face_count = 0
disp_face_count = 0

for face_idx, face in enumerate(face_list):
    ti = face.texinfo
    if ti is None:
        continue
    flags = ti.flags
    if flags & SurfFlags.NODRAW or flags & SurfFlags.SKYBOX_2D or flags & SurfFlags.SKYBOX_3D or flags & SurfFlags.TRIGGER or flags & SurfFlags.HINT or flags & SurfFlags.SKIP:
        skipped_nodraw += 1
        continue
    info = ti._info
    if info is not None and ("water" in info.mat.lower() or "color_room" in info.mat.lower()):
        # water doesn't render correctly without proper $vertexcolor/flow-map support,
        # and color_room is a $vertexcolor-tinted effect surface we can't replicate
        # (no per-vertex color data extracted) -- drop both entirely rather than
        # show a broken/flat-white result
        skipped_nodraw += 1
        continue
    if info is None:
        continue
    w, h = info.width, info.height
    if not w or not h:
        w, h = 512, 512
    mat_name = info.mat.lower()

    def uv_of(v):
        u = (v.dot(ti.s_off) + ti.s_shift) / w
        vv = (v.dot(ti.t_off) + ti.t_shift) / h
        return (u, vv)

    di = mapface_to_dispinfo.get(face_idx)
    if di is not None:
        # ---- displacement surface: reconstruct full grid ----
        corners = [e.a for e in face.edges]
        if len(corners) != 4:
            skipped_nodraw += 1
            continue
        # find corner closest to start position -> becomes grid origin (0,0)
        start = di['start']
        dists = [(c - start).mag() for c in corners]
        start_idx = dists.index(min(dists))
        c0 = corners[start_idx]
        c1 = corners[(start_idx + 1) % 4]
        c2 = corners[(start_idx + 2) % 4]
        c3 = corners[(start_idx + 3) % 4]

        power = di['power']
        size = (1 << power) + 1
        vbase = di['dispvertstart']

        # base-plane normal, used below to sanity-check the reconstructed surface
        base_normal = (c1 - c0).cross(c3 - c0)
        base_normal_len = base_normal.mag()
        if base_normal_len == 0:
            skipped_nodraw += 1
            continue
        base_normal = base_normal / base_normal_len

        grid_pos = [[None] * size for _ in range(size)]
        for row in range(size):
            s = row / (size - 1)
            for col in range(size):
                t = col / (size - 1)
                bx = c0.x * (1 - s) * (1 - t) + c1.x * (1 - s) * t + c3.x * s * (1 - t) + c2.x * s * t
                by = c0.y * (1 - s) * (1 - t) + c1.y * (1 - s) * t + c3.y * s * (1 - t) + c2.y * s * t
                bz = c0.z * (1 - s) * (1 - t) + c1.z * (1 - s) * t + c3.z * s * (1 - t) + c2.z * s * t
                dv_idx = row * size + col
                dvec, dist = dispverts[vbase + dv_idx]
                grid_pos[row][col] = Vec(bx + dvec.x * dist, by + dvec.y * dist, bz + dvec.z * dist)

        # sanity check: a genuinely broken/twisted reconstruction (bad corner match,
        # degenerate patch) produces cells whose normal points opposite the overall
        # patch plane -- a real terrain bump doesn't flip that far. If too many cells
        # fail, distrust the whole patch and fall back to its flat undisplaced quad
        # instead of rendering the twisted "3D cross" mess.
        bad_cells = 0
        total_cells = (size - 1) * (size - 1)
        for row in range(size - 1):
            for col in range(size - 1):
                a = grid_pos[row][col]; b = grid_pos[row][col + 1]
                c = grid_pos[row + 1][col]
                n = (b - a).cross(c - a)
                if n.mag() > 0 and n.dot(base_normal) < 0:
                    bad_cells += 1

        disp_label = f"disp_{di['mapface']}"
        if total_cells == 0 or bad_cells / total_cells > 0.15:
            base_idx = len(verts)
            for v in corners:
                verts.append((v.x, v.y, v.z))
                uvs.append(uv_of(v))
            for i in range(1, 3):
                tris.append((base_idx, base_idx + i, base_idx + i + 1))
                tri_mat.append(mat_name)
                tri_src.append(disp_label + "_flatfallback")
            print(f"  displacement face {face_idx}: {bad_cells}/{total_cells} bad cells, falling back to flat quad")
            disp_face_count += 1
            continue

        grid_idx = [[0] * size for _ in range(size)]
        for row in range(size):
            for col in range(size):
                p = grid_pos[row][col]
                idx = len(verts)
                verts.append((p.x, p.y, p.z))
                uvs.append(uv_of(p))
                grid_idx[row][col] = idx

        for row in range(size - 1):
            for col in range(size - 1):
                a = grid_idx[row][col]
                b = grid_idx[row][col + 1]
                c = grid_idx[row + 1][col]
                d = grid_idx[row + 1][col + 1]
                tris.append((a, b, c)); tri_mat.append(mat_name); tri_src.append(disp_label)
                tris.append((b, d, c)); tri_mat.append(mat_name); tri_src.append(disp_label)
        disp_face_count += 1
        continue

    # ---- regular flat face ----
    loop = [e.a for e in face.edges]
    if len(loop) < 3:
        continue
    base_idx = len(verts)
    for v in loop:
        verts.append((v.x, v.y, v.z))
        uvs.append(uv_of(v))
    flat_label = f"flat_face_{face_idx}"
    for i in range(1, len(loop) - 1):
        tris.append((base_idx, base_idx + i, base_idx + i + 1))
        tri_mat.append(mat_name)
        tri_src.append(flat_label)
    face_count += 1

print(f"flat faces: {face_count}  displacement faces: {disp_face_count}  skipped: {skipped_nodraw}")
print(f"total verts: {len(verts)}  total tris: {len(tris)}")

unique_mats = sorted(set(tri_mat))
mat_to_id = {m: i for i, m in enumerate(unique_mats)}
tri_mat_id = np.array([mat_to_id[m] for m in tri_mat], dtype=np.int32)
tri_src_arr = np.array(tri_src)  # parallel to tri_mat_id, human-readable object id per triangle
print("unique materials used:", len(unique_mats))

verts_arr = np.array(verts, dtype=np.float32)
uvs_arr = np.array(uvs, dtype=np.float32)
tris_arr = np.array(tris, dtype=np.uint32)

# ---- weld vertices that sit at (nearly) the same 3D position AND have the same UV ----
# every face/displacement patch currently emits its own independent copies of
# shared-edge vertices; without welding, adjacent pieces don't share indices and
# any tiny float mismatch (or just being separate verts at all) shows as a seam/crack.
# UV *is* part of the weld key: welding by position alone silently discarded whichever
# UV was recorded first at a shared position, corrupting texturing on every triangle
# that touched a boundary between differently-UV'd faces (huge "stretched" artifacts).
# Keying on (position, uv) still merges true duplicates (continuous same-material
# neighbors) while correctly keeping separate copies at genuine UV seams -- standard
# and necessary hard-vertex-split behavior, and positions still coincide exactly so
# there's no reintroduced crack.
ROUND_DECIMALS = 2  # source units (inches); ~0.25mm after *0.0254 scale
UV_ROUND_DECIMALS = 4
weld_key = np.concatenate([
    np.round(verts_arr, ROUND_DECIMALS),
    np.round(uvs_arr, UV_ROUND_DECIMALS),
], axis=1)
_, unique_idx, inverse = np.unique(weld_key, axis=0, return_index=True, return_inverse=True)
inverse = inverse.reshape(-1)

welded_verts = verts_arr[unique_idx]
welded_uvs = uvs_arr[unique_idx]
welded_tris = inverse[tris_arr]

print(f"welded {len(verts_arr)} verts -> {len(welded_verts)} unique positions")

# ---- drop the 3D skybox: a miniature floating replica of the map built far away,
# only meant to be seen through a special skybox camera, not walkable geometry.
# Its faces aren't reliably flagged, but it sits in its own disconnected, isolated
# height band (source Z, i.e. up-axis) -- use connected components + a height
# threshold to cut it (and any other floating debris) out cleanly.
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

n_verts_w = len(welded_verts)
ii = np.concatenate([welded_tris[:, 0], welded_tris[:, 1], welded_tris[:, 2]])
jj = np.concatenate([welded_tris[:, 1], welded_tris[:, 2], welded_tris[:, 0]])
adj = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n_verts_w, n_verts_w))
n_comp, labels = connected_components(adj, directed=False)

Z_SKYBOX_THRESHOLD = 5000  # source units; confirmed a clean gap, no component straddles this
comp_zmin = np.full(n_comp, np.inf)
np.minimum.at(comp_zmin, labels, welded_verts[:, 2])
keep_component = comp_zmin <= Z_SKYBOX_THRESHOLD
vert_keep_mask = keep_component[labels]

tri_keep_mask = vert_keep_mask[welded_tris[:, 0]] & vert_keep_mask[welded_tris[:, 1]] & vert_keep_mask[welded_tris[:, 2]]
dropped_tris = (~tri_keep_mask).sum()
print(f"dropped {dropped_tris} triangles belonging to skybox/floating components")

kept_tri_mat_id = tri_mat_id[tri_keep_mask]
kept_tri_src = tri_src_arr[tri_keep_mask]
kept_tris_raw = welded_tris[tri_keep_mask]

# compact vertex list to only those actually referenced by kept triangles
used_verts = np.unique(kept_tris_raw)
remap = -np.ones(n_verts_w, dtype=np.int64)
remap[used_verts] = np.arange(len(used_verts))
final_verts = welded_verts[used_verts]
final_uvs = welded_uvs[used_verts]
final_tris = remap[kept_tris_raw]

print(f"final: {len(final_verts)} verts, {len(final_tris)} tris")

# ---- T-junction stitching ----
# Neighboring displacement patches can have different subdivision power (this map
# mixes 2/3/4), so a shared edge gets a different number of vertices on each side --
# no amount of position-only vertex snapping closes that gap, since the coarse side
# genuinely has no vertex at the fine side's intermediate positions. Fix: find
# boundary edges (used by only one triangle -- i.e. nothing on this mesh connects to
# them, which is exactly what a crack looks like), find any OTHER vertex lying
# exactly on that edge segment (the fine neighbor's intermediate point), and re-fan
# the boundary triangle through it so the coarse side actually has a vertex there too.
from collections import defaultdict

def stitch_t_junctions(verts, tris, tri_mat_id, tri_src, tol=0.5, cell=8.0):
    edge_count = defaultdict(int)
    for tri in tris:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            edge_count[key] += 1
    boundary_edges = {e for e, c in edge_count.items() if c == 1}
    print(f"  boundary edges: {len(boundary_edges)}")

    grid = defaultdict(list)
    for vi, p in enumerate(verts):
        grid[(int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))].append(vi)

    def candidates_near_segment(pa, pb):
        lo = np.minimum(pa, pb) - cell
        hi = np.maximum(pa, pb) + cell
        c0 = (lo // cell).astype(int)
        c1 = (hi // cell).astype(int)
        cand = []
        for cx in range(c0[0], c1[0] + 1):
            for cy in range(c0[1], c1[1] + 1):
                for cz in range(c0[2], c1[2] + 1):
                    cand.extend(grid.get((cx, cy, cz), []))
        return cand

    out_tris = []
    out_mat = []
    out_src = []
    stitched_count = 0
    for ti in range(len(tris)):
        p, q, r = tris[ti]
        tri_mat = tri_mat_id[ti]
        tri_src_val = tri_src[ti]
        # for each directed edge of this triangle, see if it's a boundary edge with
        # an intermediate point sitting exactly on it
        edges = [(p, q, r), (q, r, p), (r, p, q)]  # (edge_start, edge_end, opposite_vertex)
        replacement = None
        for a, b, c in edges:
            key = (a, b) if a < b else (b, a)
            if key not in boundary_edges:
                continue
            pa, pb = verts[a], verts[b]
            seg = pb - pa
            seg_len2 = seg.dot(seg)
            if seg_len2 < 1e-6:
                continue
            on_seg = []
            for v in candidates_near_segment(pa, pb):
                if v == a or v == b:
                    continue
                pv = verts[v]
                t = (pv - pa).dot(seg) / seg_len2
                if t <= 1e-4 or t >= 1 - 1e-4:
                    continue
                proj = pa + seg * t
                if np.linalg.norm(pv - proj) < tol:
                    on_seg.append((t, v))
            if on_seg:
                on_seg.sort()
                chain = [a] + [v for _, v in on_seg] + [b]
                replacement = [(chain[i], chain[i + 1], c) for i in range(len(chain) - 1)]
                break  # one stitched edge per triangle per pass; a second pass catches more if needed
        if replacement:
            stitched_count += 1
            for f in replacement:
                out_tris.append(f)
                out_mat.append(tri_mat)
                out_src.append(tri_src_val)
        else:
            out_tris.append((p, q, r))
            out_mat.append(tri_mat)
            out_src.append(tri_src_val)
    print(f"  stitched {stitched_count} boundary triangles")
    return np.array(out_tris, dtype=np.uint32), np.array(out_mat, dtype=np.int32), np.array(out_src), stitched_count

print("stitching T-junctions...")
for pass_num in range(3):  # a few passes: stitching one edge can reveal a triangle's other edge also needs it
    final_tris, kept_tri_mat_id, kept_tri_src, n_stitched = stitch_t_junctions(final_verts, final_tris, kept_tri_mat_id, kept_tri_src)
    if n_stitched == 0:
        break

# ---- remove genuinely duplicate/coincident triangles (real z-fighting cause) ----
# Source brushes that touch/overlap slightly can compile to two different faces
# occupying the exact same 3D position -- both get rendered, and since they're at
# identical depth, the GPU flickers between them (z-fighting). Detect by canonical
# (sorted, rounded) vertex position per triangle and keep only the first occurrence.
seen_keys = set()
dedup_tris = []
dedup_mat = []
dedup_src = []
n_dropped = 0
for i, t in enumerate(final_tris):
    pts = tuple(sorted(tuple(np.round(final_verts[v], 2)) for v in t))
    if pts in seen_keys:
        n_dropped += 1
        continue
    seen_keys.add(pts)
    dedup_tris.append(t)
    dedup_mat.append(kept_tri_mat_id[i])
    dedup_src.append(kept_tri_src[i])
print(f"removed {n_dropped} duplicate/coincident triangles (z-fighting sources)")
final_tris = np.array(dedup_tris, dtype=np.uint32)
kept_tri_mat_id = np.array(dedup_mat, dtype=np.int32)
kept_tri_src = np.array(dedup_src)

np.savez(OUT_NPZ, verts=final_verts, uvs=final_uvs, tris=final_tris, tri_mat_id=kept_tri_mat_id,
         mat_names=np.array(unique_mats), tri_src=kept_tri_src)
print("wrote", OUT_NPZ)
